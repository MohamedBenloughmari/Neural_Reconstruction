import numpy as np
import torch
import torch.nn as nn

from .gaussian import gaussian_blur_torch, img_to_encoder_torch
from .grid import generate_hex_grid
from .glm import make_glm_context, glm_rates_np, glm_rates_torch_batch
from .particles import propagate_particles, resample_particles, sample_positions
from .decoder import adam_update_A, update_hessian
from .animation import create_animation


class NeuralCortex:
    def __init__(self, dx, dy, dt, D_diff, ds=None, img_size=32, device=None):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.D_diff = D_diff
        self.ds = ds
        self.img_size = img_size
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.half_n = None
        self.optotype_np = None
        self.optotype_display = None
        self.optotype_torch = None
        self.n_steps = None
        self.walk = None
        self.ganglion_x = None
        self.ganglion_y = None
        self.spikes_on = None
        self.spikes_off = None
        self.r_on_all = None
        self.r_off_all = None
        self.A_hat_history = None
        self.S_hat_history = None
        self.q_particles = None
        self.q_weights = None
        self._mean_offset = 0.0
        self._glm_ctx = None
        self.lambda0 = 10.0
        self.lambda1 = 100.0

    def fit(self, optotype, blur_sigma=1.5):
        if optotype.dim() == 3:
            optotype = optotype.squeeze(0)
        h, _ = optotype.shape
        self.half_n = h // 2
        optotype_dev = optotype.to(self.device)
        blurred = gaussian_blur_torch(optotype_dev, blur_sigma)
        self.optotype_display = blurred.cpu().numpy()
        self.optotype_np = np.fliplr(blurred.cpu().numpy().T)
        self.optotype_torch = img_to_encoder_torch(blurred)

    def simulate_random_walk(self, T):
        self.n_steps = int(T / self.dt)
        sigma = np.sqrt(self.D_diff * self.dt)
        disp = np.random.normal(0.0, sigma, size=(self.n_steps, 2))
        self.walk = np.vstack([np.zeros((1, 2)), np.cumsum(disp, axis=0)])

    def compute_activations(self, grid_range=10.0, grid_resolution=30):
        self.grid_range = grid_range
        self.grid_resolution = grid_resolution
        self.ganglion_x, self.ganglion_y = generate_hex_grid(
            grid_range, grid_resolution
        )
        self.n_cells = len(self.ganglion_x)
        n_t = self.n_steps + 1
        self.spikes_on = np.zeros((n_t, self.n_cells), dtype=int)
        self.spikes_off = np.zeros((n_t, self.n_cells), dtype=int)
        for t in range(n_t):
            cx, cy = self.walk[t]
            lam_on, lam_off = glm_rates_np(
                self.optotype_np, cx, cy,
                self.ganglion_x, self.ganglion_y,
                self.half_n, self.dx, self.dy,
                self.ds, self.lambda0, self.lambda1,
            )
            self.spikes_on[t] = np.random.poisson(lam_on * self.dt)
            self.spikes_off[t] = np.random.poisson(lam_off * self.dt)

    def _init_decode_state(self):
        self._glm_ctx = make_glm_context(
            self.ganglion_x, self.ganglion_y,
            self.half_n, self.dx, self.dy,
            self.ds, self.lambda0, self.lambda1,
            self.dt, self.device,
        )
        dev = self.device
        self._t_diff_sigma = torch.tensor(
            np.sqrt(self.D_diff * self.dt), dtype=torch.float32, device=dev,
        )
        self.r_on_all = torch.as_tensor(
            self.spikes_on, dtype=torch.float32, device=dev,
        )
        self.r_off_all = torch.as_tensor(
            self.spikes_off, dtype=torch.float32, device=dev,
        )

    def decode(self, mean_offset=0.0, n_particles=80, n_samples=30,
               beta=0.05, gamma=0.1, adam_iter=20, lr=1e-2,
               anchor_weight=1.0, hessian_tau=0.5,
               hessian_every=1, verbose=True):
        self._mean_offset = mean_offset
        self._init_decode_state()
        dev = self.device
        glm_ctx = self._glm_ctx
        N_sp = self.img_size * self.img_size

        A_param = nn.Parameter(torch.zeros(N_sp, device=dev, dtype=torch.float32))
        Hessian = torch.zeros((N_sp, N_sp), device=dev, dtype=torch.float32)
        optimizer = torch.optim.Adam([A_param], lr=lr)

        T = self.n_steps + 1
        particles = torch.zeros(n_particles, 2, device=dev)
        weights = torch.ones(n_particles, device=dev) / n_particles

        self.q_particles = torch.zeros(T, n_particles, 2, device=dev)
        self.q_weights = torch.zeros(T, n_particles, device=dev)
        self.A_hat_history = np.zeros((T, N_sp))
        self.S_hat_history = np.zeros((T, self.img_size, self.img_size))

        for t in range(T):
            cur_anchor_w = 0.0 if t == 0 else anchor_weight

            if t > 0:
                A_flat = A_param.detach()
                img = A_flat.reshape(self.img_size, self.img_size) + self._mean_offset
                S_t = img_to_encoder_torch(img)

                get_rates = lambda cx, cy: glm_rates_torch_batch(
                    S_t, cx, cy, glm_ctx,
                )

                particles, weights = propagate_particles(
                    particles=particles,
                    r_on_t=self.r_on_all[t],
                    r_off_t=self.r_off_all[t],
                    get_rates=get_rates,
                    diff_sigma=self._t_diff_sigma,
                    dt=glm_ctx.t_dt,
                    eps=glm_ctx.t_eps,
                    device=dev,
                )
                particles, weights = resample_particles(particles, weights, dev)

            self.q_particles[t] = particles
            self.q_weights[t] = weights

            samples_t = sample_positions(particles, weights, n_samples, dev)
            samples_np = samples_t.cpu().numpy()
            r_on_t = self.r_on_all[t]
            r_off_t = self.r_off_all[t]
            A_anchor = A_param.detach().clone()

            loss_val = adam_update_A(
                glm_ctx=glm_ctx,
                A_param=A_param,
                Hessian=Hessian,
                optimizer=optimizer,
                samples_t=samples_np,
                r_on_t=r_on_t,
                r_off_t=r_off_t,
                A_anchor=A_anchor,
                beta=beta,
                gamma=gamma,
                n_iter=adam_iter,
                img_size=self.img_size,
                mean_offset=self._mean_offset,
                device=dev,
                anchor_weight=cur_anchor_w,
            )

            if (t % hessian_every) == 0:
                A_new = A_param.detach().clone()
                Hessian = update_hessian(
                    glm_ctx=glm_ctx,
                    H_prev=Hessian,
                    A_anchor=A_new,
                    samples_t=samples_t,
                    r_on_t=r_on_t,
                    r_off_t=r_off_t,
                    tau=hessian_tau,
                    dt=self.dt,
                    img_size=self.img_size,
                    mean_offset=self._mean_offset,
                    device=dev,
                )

            A_np = A_param.detach().cpu().numpy()
            self.A_hat_history[t] = A_np
            self.S_hat_history[t] = (
                A_np.reshape(self.img_size, self.img_size) + self._mean_offset
            )

            if verbose and t % max(T // 10, 1) == 0:
                print(
                    f"[decode] t={t}/{T - 1}  loss={loss_val:.3f}  "
                    f"||A||_1={np.abs(A_np).sum():.2f}"
                )
        return self.A_hat_history, self.S_hat_history

    def animate(self, interval=80, save_path=None):
        return create_animation(
            ganglion_x=self.ganglion_x,
            ganglion_y=self.ganglion_y,
            half_n=self.half_n,
            dx=self.dx,
            dy=self.dy,
            walk=self.walk,
            optotype_display=self.optotype_display,
            spikes_on=self.spikes_on,
            spikes_off=self.spikes_off,
            S_hat_history=self.S_hat_history,
            n_steps=self.n_steps,
            dt=self.dt,
            interval=interval,
            save_path=save_path,
        )
