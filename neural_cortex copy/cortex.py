import numpy as np
import torch
import torch.nn as nn

from .gaussian import img_to_encoder_torch
from .grid import generate_hex_grid
from .glm import make_glm_context, glm_rates
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

    def fit(self, optotype):
        if optotype.dim() == 3:
            optotype = optotype.squeeze(0)
        self.H, self.W = optotype.shape
        self.half_n = self.H // 2
        optotype = optotype.to(self.device)
        self.optotype=optotype

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
            lam_on, lam_off = glm_rates(
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
        self.diff_sigma = torch.tensor(
            np.sqrt(self.D_diff * self.dt), dtype=torch.float32, device=self.device,
        )
        self.spikes_on = torch.as_tensor(
            self.spikes_on, dtype=torch.float32, device=self.device,
        )
        self.spikes_off = torch.as_tensor(
            self.spikes_off, dtype=torch.float32, device=self.device,
        )
        #self.r_off_all = torch.as_tensor(
        #    self.spikes_off, dtype=torch.float32, device=self.device,
        #)

    def decode(self, n_particles=80, n_samples=30,
               beta=0.05, gamma=0.1, adam_iter=20, lr=1e-2,
               anchor_weight=1.0, hessian_tau=0.5,
               hessian_every=1, verbose=True):
        self._init_decode_state()
        glm_ctx = self._glm_ctx
        N_sp = self.H * self.W

        S_param = nn.Parameter(torch.zeros(N_sp, device=self.device, dtype=torch.float32))
        Hessian = torch.zeros((N_sp, N_sp), device=self.device, dtype=torch.float32)
        optimizer = torch.optim.Adam([S_param], lr=lr)

        NT = self.n_steps + 1
        particles = torch.zeros(n_particles, 2, device=self.device)
        weights = torch.ones(n_particles, device=self.device) / n_particles

        self.q_particles = torch.zeros(NT, n_particles, 2, device=self.device)
        self.q_weights = torch.zeros(NT, n_particles, device=self.device)
        self.S_hat_history = np.zeros((NT, N_sp))
        self.img_history = np.zeros((NT, self.img_size, self.img_size))

        for t in range(NT):
            if t > 0:
                S_flat = S_param.detach()
                img = S_flat.reshape(self.img_size, self.img_size)
                S_t = torch.tensor(img,dtype=torch.float32,device=self.device)

                get_rates= lambda cx,cy: glm_rates(S_t, cx, cy, self.ganglion_x, self.ganglion_y, self.half_n, self.dx, self.dy, self.ds,
                 self.lambda0, self.lambda1)

                particles, weights = propagate_particles(
                    particles=particles,
                    spikes_on_t=self.spikes_on[t],
                    spikes_off_t=self.spikes_off[t],
                    get_rates=get_rates,
                    diff_sigma=self.diff_sigma,
                    dt=glm_ctx.dt,
                    eps=glm_ctx.eps,
                    device=self.device,
                )
                particles, weights = resample_particles(particles, weights, self.device)

            self.q_particles[t] = particles
            self.q_weights[t] = weights

            samples_t = sample_positions(particles, weights, n_samples, self.device)
            samples_t = samples_t.cpu().numpy()
            spikes_on_t = self.spikes_on[t]
            spikes_off_t = self.spikes_off[t]
            S_anchor = S_param.detach().clone()

            loss_val = adam_update(
                glm_ctx=glm_ctx,
                A_param=S_param,
                Hessian=Hessian,
                optimizer=optimizer,
                samples_t=samples_t,
                spikes_on_t=spikes_on_t,
                spikes_off_t=spikes_off_t,
                A_anchor=S_anchor,
                beta=beta,
                gamma=gamma,
                n_iter=adam_iter,
                img_size=self.img_size,
                mean_offset=self._mean_offset,
                device=self.device,
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
