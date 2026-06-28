import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, jacrev
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random


def generate_letter_image(letter, angle=0.0, output_path=None,
                          font_size=50, bg_color=255, fg_color=0,
                          image_size=32):
    if len(letter) != 1:
        raise ValueError(f"Expected a single character, got: {repr(letter)}")

    pad = image_size * 2
    canvas = pad + image_size
    big = Image.new("L", (canvas, canvas), color=bg_color)
    draw = ImageDraw.Draw(big)

    font = None
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/System/Library/Fonts/Courier.ttc",
        "C:/Windows/Fonts/cour.ttf",
    ]:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size=font_size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), letter, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas - text_w) / 2 - bbox[0]
    y = (canvas - text_h) / 2 - bbox[1]
    draw.text((x, y), letter, fill=fg_color, font=font)

    big = big.rotate(angle, resample=Image.BICUBIC,
                     center=(canvas / 2, canvas / 2))
    left = pad // 2
    top = pad // 2
    img = big.crop((left, top, left + image_size, top + image_size))

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        img.save(output_path, format="PNG")

    return img


def letter_to_tensor(letter, angle=0.0, image_size=32, invert=True):
    img = generate_letter_image(letter, angle=angle, image_size=image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    return torch.from_numpy(arr).float()


def load_image_tensor(image_path, image_size=32, invert=True, deblur=True):
    img = Image.open(image_path).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    if deblur:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    return torch.from_numpy(arr).float()


def _gaussian_kernel_2d(sigma, kernel_size=None):
    if kernel_size is None:
        kernel_size = int(4 * sigma + 1) | 1
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    x, y = torch.meshgrid(ax, ax, indexing='ij')
    k = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k[None, None, :, :]


def _gaussian_blur_torch(img_2d, sigma):
    if sigma <= 0:
        return img_2d
    kernel = _gaussian_kernel_2d(sigma)
    kernel = kernel.to(img_2d.device)
    padding = kernel.shape[-1] // 2
    img_4d = img_2d[None, None, :, :]
    return F.conv2d(img_4d, kernel, padding=padding)[0, 0]


def _img_to_encoder_torch(img):
    return torch.flip(img.T, dims=[-1])


def generate_hex_grid(grid_range, grid_resolution):
    spacing = 1.0 * grid_range / grid_resolution
    dx = spacing
    dy = spacing * np.sqrt(3.0) / 2.0
    x_vals = np.arange(-grid_range, grid_range + dx, dx)
    y_vals = np.arange(-grid_range, grid_range + dy, dy)
    positions_x = []
    positions_y = []
    for i, y in enumerate(y_vals):
        offset = spacing / 2.0 if i % 2 == 1 else 0.0
        for x in x_vals:
            px = x + offset
            if abs(px) <= grid_range:
                positions_x.append(px)
                positions_y.append(y)
    return np.array(positions_x), np.array(positions_y)


class NeuralEncoder:
    def __init__(self, dx, dy, dt, D_diff, ds=None, img_size=32, device=None):
        self.dx, self.dy, self.dt = dx, dy, dt
        self.D_diff = D_diff
        self.ds = ds
        self.img_size = img_size
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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

        self.lambda0 = 10.0
        self.lambda1 = 100.0

    def fit(self, optotype, blur_sigma=1.5):
        if optotype.dim() == 3:
            optotype = optotype.squeeze(0)
        h, _ = optotype.shape
        self.half_n = h // 2
        optotype_dev = optotype.to(self.device)
        blurred = _gaussian_blur_torch(optotype_dev, blur_sigma)
        self.optotype_display = blurred.cpu().numpy()
        self.optotype_np = np.fliplr(blurred.cpu().numpy().T)
        self.optotype_torch = _img_to_encoder_torch(blurred)

    def simulate_random_walk(self, T):
        self.n_steps = int(T / self.dt)
        sigma = np.sqrt(self.D_diff * self.dt)
        disp = np.random.normal(0.0, sigma, size=(self.n_steps, 2))
        self.walk = np.vstack([np.zeros((1, 2)), np.cumsum(disp, axis=0)])

    def _glm_rates_np(self, S, cx, cy):
        H, W = S.shape
        sigma_s = 0.5 * self.ds
        sigma_e = 0.203 * self.ds
        sigma2 = sigma_s ** 2 + sigma_e ** 2
        px_x = (np.arange(H) + 0.5 - self.half_n) * self.dx
        px_y = (np.arange(W) + 0.5 - self.half_n) * self.dy
        xg, yg = self.ganglion_x, self.ganglion_y
        diff_x = (cx + px_x[None, :]) - xg[:, None]
        diff_y = (cy + px_y[None, :]) - yg[:, None]
        gx_w = np.exp(-0.5 * diff_x ** 2 / sigma2)
        gy_w = np.exp(-0.5 * diff_y ** 2 / sigma2)
        tmp = gx_w @ S
        c_raw = np.sum(tmp * gy_w, axis=1) / (2.0 * np.pi * sigma2)
        g_norm = max(c_raw.max(), 1e-9)
        c = np.clip(c_raw / g_norm, 0.0, 1.0)
        lam_on = self.lambda0 * np.exp(np.log(self.lambda1 / self.lambda0) * c)
        lam_off = self.lambda0 * np.exp(np.log(self.lambda1 / self.lambda0) * (1.0 - c))
        return lam_on, lam_off

    def _precompute_torch_constants(self):
        dev = self.device
        self.t_xg = torch.as_tensor(self.ganglion_x, dtype=torch.float32, device=dev)
        self.t_yg = torch.as_tensor(self.ganglion_y, dtype=torch.float32, device=dev)
        H = W = 2 * self.half_n
        self.t_px_x = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5 - self.half_n) * self.dx
        self.t_px_y = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5 - self.half_n) * self.dy
        sigma_s = 0.5 * self.ds
        sigma_e = 0.203 * self.ds
        self.t_sigma2 = torch.tensor(sigma_s ** 2 + sigma_e ** 2,
                                     dtype=torch.float32, device=dev)
        self.t_inv_sigma2 = 1.0 / self.t_sigma2
        self.t_neg_half_inv_sigma2 = -0.5 * self.t_inv_sigma2
        self.t_log_ratio = torch.tensor(np.log(self.lambda1 / self.lambda0),
                                        dtype=torch.float32, device=dev)
        self.t_two_pi_sigma2 = 2.0 * float(np.pi) * self.t_sigma2
        self.t_inv_two_pi_sigma2 = 1.0 / self.t_two_pi_sigma2
        self.t_lambda0 = torch.tensor(self.lambda0, dtype=torch.float32, device=dev)
        self.t_dt = torch.tensor(self.dt, dtype=torch.float32, device=dev)
        self.t_log_dt = torch.tensor(np.log(self.dt), dtype=torch.float32, device=dev)
        self.t_eps = torch.tensor(1e-12, dtype=torch.float32, device=dev)

    def _precompute_spatial_kernels(self, cx, cy):
        neg_half_div_s2 = self.t_neg_half_inv_sigma2
        diff_x = (cx[:, None, None] + self.t_px_x[None, None, :]) - self.t_xg[None, :, None]
        diff_y = (cy[:, None, None] + self.t_px_y[None, None, :]) - self.t_yg[None, :, None]
        gx_w = torch.exp(neg_half_div_s2 * diff_x * diff_x)
        gy_w = torch.exp(neg_half_div_s2 * diff_y * diff_y)
        return gx_w, gy_w

    def _glm_rates_torch_batch(self, S, cx, cy, gx_w=None, gy_w=None):
        if gx_w is None or gy_w is None:
            gx_w, gy_w = self._precompute_spatial_kernels(cx, cy)
        tmp = torch.matmul(gx_w, S)
        c_raw = (tmp * gy_w).sum(dim=2) * self.t_inv_two_pi_sigma2
        g_norm = c_raw.detach().amax(dim=1, keepdim=True).clamp(min=1e-9)
        c = (c_raw / g_norm).clamp(0.0, 1.0)
        lam_on = self.t_lambda0 * torch.exp(self.t_log_ratio * c)
        lam_off = self.t_lambda0 * torch.exp(self.t_log_ratio * (1.0 - c))
        return lam_on, lam_off

    def compute_activations(self, grid_range=10.0, grid_resolution=30):
        self.grid_range = grid_range
        self.grid_resolution = grid_resolution
        self.ganglion_x, self.ganglion_y = generate_hex_grid(grid_range, grid_resolution)
        self.n_cells = len(self.ganglion_x)
        n_t = self.n_steps + 1
        self.spikes_on = np.zeros((n_t, self.n_cells), dtype=int)
        self.spikes_off = np.zeros((n_t, self.n_cells), dtype=int)
        for t in range(n_t):
            cx, cy = self.walk[t]
            lam_on, lam_off = self._glm_rates_np(self.optotype_np, cx, cy)
            self.spikes_on[t] = np.random.poisson(lam_on * self.dt)
            self.spikes_off[t] = np.random.poisson(lam_off * self.dt)

    def _propagate_particles(self, particles, S_t, t):
        dev = self.device
        n_p = particles.shape[0]
        sigma = self._t_diff_sigma
        noise = torch.randn(n_p, 2, device=dev) * sigma
        new_p = particles + noise
        with torch.no_grad():
            lam_on, lam_off = self._glm_rates_torch_batch(S_t, new_p[:, 0], new_p[:, 1])
            r_on = self.r_on_all[t]
            r_off = self.r_off_all[t]
            dt = self.t_dt
            log_w = (r_on * torch.log(lam_on * dt + self.t_eps) - lam_on * dt
                     + r_off * torch.log(lam_off * dt + self.t_eps) - lam_off * dt
                     ).sum(dim=1)
            log_w = log_w - log_w.max()
            w = torch.exp(log_w)
            w = w / w.sum()
        return new_p, w

    def _resample(self, particles, weights):
        dev = self.device
        n_p = weights.shape[0]
        csum = torch.cumsum(weights, dim=0)
        u = torch.rand(n_p, device=dev)
        idx = torch.searchsorted(csum, u)
        idx = idx.clamp(0, n_p - 1)
        uniform_w = torch.ones(n_p, device=dev) / n_p
        return particles[idx], uniform_w

    def _sample_positions(self, particles, weights, n_samples):
        dev = self.device
        n_p = weights.shape[0]
        csum = torch.cumsum(weights, dim=0)
        u = torch.rand(n_samples, device=dev)
        idx = torch.searchsorted(csum, u)
        idx = idx.clamp(0, n_p - 1)
        return particles[idx]

    def _Er_per_sample_fn(self, S, cx, cy, r_on_t, r_off_t, gx_w, gy_w):
        lam_on, lam_off = self._glm_rates_torch_batch(S, cx, cy, gx_w, gy_w)
        dt = self.t_dt
        return ((lam_on * dt - r_on_t * torch.log(lam_on * dt + self.t_eps)).sum(dim=1)
                + (lam_off * dt - r_off_t * torch.log(lam_off * dt + self.t_eps)).sum(dim=1))

    def Er_fn(self, S, samples_t, r_on_t, r_off_t, gx_w=None, gy_w=None):
        cx = torch.as_tensor(samples_t[:, 0], dtype=torch.float32, device=self.device)
        cy = torch.as_tensor(samples_t[:, 1], dtype=torch.float32, device=self.device)
        per_sample = self._Er_per_sample_fn(S, cx, cy, r_on_t, r_off_t, gx_w, gy_w)
        return per_sample.sum() / samples_t.shape[0]

    def Ep_fn(self, A_param, A_anchor, beta):
        log_p_anchor = -beta * torch.sum(torch.abs(A_anchor))
        grad_log_p = -beta * torch.sign(A_anchor)
        neg_Ep_lin = log_p_anchor - (grad_log_p * (A_param - A_anchor)).sum()
        return -neg_Ep_lin

    def _adam_update_A(self, A_param, Hessian, optimizer,
                       samples_t, r_on_t, r_off_t,
                       A_anchor, beta, gamma, n_iter, anchor_weight=1.0):
        dev = self.device
        cx = torch.as_tensor(samples_t[:, 0], dtype=torch.float32, device=dev)
        cy = torch.as_tensor(samples_t[:, 1], dtype=torch.float32, device=dev)
        gx_w, gy_w = self._precompute_spatial_kernels(cx, cy)
        last_loss = 0.0
        for _ in range(n_iter):
            optimizer.zero_grad()
            S = _img_to_encoder_torch(A_param.reshape(self.img_size, self.img_size) + self._mean_offset)
            Er = self.Er_fn(S, samples_t, r_on_t, r_off_t, gx_w, gy_w)

            diff = A_param - A_anchor
            Eg = 0.5 * anchor_weight * (diff @ (Hessian @ diff))
            Ep = self.Ep_fn(A_param, A_anchor, beta)

            range_pen = gamma * (
                torch.clamp(S - 1.0, min=0.0)
                + torch.clamp(-S, min=0.0)
            ).sum()

            loss = Er + Eg + Ep + range_pen
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
        return last_loss

    def _compute_gauss_newton_hessian(self, A, samples_t, r_on_t, r_off_t, gx_w, gy_w):
        dev = self.device
        n_samples = samples_t.shape[0]
        N_sp = A.numel()

        def Er_single(A_vec, cx1, cy1, r_on1, r_off1, gx1, gy1):
            S = _img_to_encoder_torch(A_vec.reshape(self.img_size, self.img_size) + self._mean_offset)
            lam_on, lam_off = self._glm_rates_torch_batch(S, cx1, cy1, gx1, gy1)
            dt = self.t_dt
            loss = ((lam_on * dt - r_on1 * torch.log(lam_on * dt + self.t_eps)).sum()
                    + (lam_off * dt - r_off1 * torch.log(lam_off * dt + self.t_eps)).sum())
            return loss / n_samples

        J = torch.zeros(n_samples, N_sp, device=dev, dtype=torch.float32)
        for i in range(n_samples):
            cx_i = samples_t[i:i+1, 0:1]
            cy_i = samples_t[i:i+1, 1:2]
            gx_i = gx_w[i:i+1]
            gy_i = gy_w[i:i+1]
            r_on_i = r_on_t.unsqueeze(0)
            r_off_i = r_off_t.unsqueeze(0)

            def bound_fn(a):
                return Er_single(a, cx_i, cy_i, r_on_i, r_off_i, gx_i, gy_i)

            J[i] = grad(bound_fn)(A).detach()

        H_gn = J.T @ J
        return 0.5 * (H_gn + H_gn.T)

    def _update_hessian(self, H_prev, A_anchor, samples_t,
                        r_on_t, r_off_t, tau):
        dev = self.device
        cx = torch.as_tensor(samples_t[:, 0], dtype=torch.float32, device=dev)
        cy = torch.as_tensor(samples_t[:, 1], dtype=torch.float32, device=dev)
        gx_w, gy_w = self._precompute_spatial_kernels(cx, cy)

        H_new = self._compute_gauss_newton_hessian(A_anchor, samples_t, r_on_t, r_off_t, gx_w, gy_w)
        decay = float(np.exp(-self.dt / tau))
        return decay * H_prev + H_new

    def decode(self, mean_offset=0.0, n_particles=80, n_samples=30,
               beta=0.05, gamma=0.1, adam_iter=20, lr=1e-2,
               anchor_weight=1.0, hessian_tau=0.5,
               hessian_every=1, verbose=True):
        self._mean_offset = mean_offset
        self._precompute_torch_constants()
        dev = self.device
        N_sp = self.img_size * self.img_size

        self._t_diff_sigma = torch.tensor(np.sqrt(self.D_diff * self.dt),
                                          dtype=torch.float32, device=dev)

        self.r_on_all = torch.as_tensor(self.spikes_on, dtype=torch.float32, device=dev)
        self.r_off_all = torch.as_tensor(self.spikes_off, dtype=torch.float32, device=dev)

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
                S_t = _img_to_encoder_torch(img)
                particles, weights = self._propagate_particles(particles, S_t, t)
                particles, weights = self._resample(particles, weights)

            self.q_particles[t] = particles
            self.q_weights[t] = weights

            samples_t = self._sample_positions(particles, weights, n_samples)
            samples_np = samples_t.cpu().numpy()
            r_on_t = self.r_on_all[t]
            r_off_t = self.r_off_all[t]
            A_anchor = A_param.detach().clone()

            loss_val = self._adam_update_A(
                A_param, Hessian, optimizer, samples_np, r_on_t, r_off_t,
                A_anchor=A_anchor, anchor_weight=cur_anchor_w,
                beta=beta, gamma=gamma, n_iter=adam_iter,
            )

            if (t % hessian_every) == 0:
                A_new = A_param.detach().clone()
                Hessian = self._update_hessian(
                    Hessian, A_new, samples_t, r_on_t, r_off_t,
                    tau=hessian_tau,
                )

            A_np = A_param.detach().cpu().numpy()
            self.A_hat_history[t] = A_np
            self.S_hat_history[t] = A_np.reshape(self.img_size, self.img_size) + self._mean_offset

            if verbose and t % max(T // 10, 1) == 0:
                print(f"[decode] t={t}/{T-1}  loss={loss_val:.3f}  "
                      f"||A||_1={np.abs(A_np).sum():.2f}")
        return self.A_hat_history, self.S_hat_history

    def animate(self, interval=80, save_path=None):
        xg, yg = self.ganglion_x, self.ganglion_y
        half_x = self.half_n * self.dx
        half_y = self.half_n * self.dy
        pts_x, pts_y = xg, yg
        n_cells = len(pts_x)

        has_decoder = self.S_hat_history is not None
        n_panels = 4 if has_decoder else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        ax_opt, ax_on, ax_off = axes[0], axes[1], axes[2]
        ax_rec = axes[3] if has_decoder else None
        for ax in axes:
            ax.set_aspect('equal')

        cx0, cy0 = self.walk[0]
        opt_img = ax_opt.imshow(
            self.optotype_display,
            extent=[cx0 - half_x, cx0 + half_x, cy0 - half_y, cy0 + half_y],
            origin='upper', cmap='gray_r', alpha=0.9,
        )
        walk_line, = ax_opt.plot([], [], color='steelblue', lw=0.7)
        fovea, = ax_opt.plot([], [], 'b+', ms=12, mew=2)
        ax_opt.set_xlim(xg.min(), xg.max())
        ax_opt.set_ylim(yg.min(), yg.max())
        ax_opt.set_title('Stimulus + eye path')
        time_text = ax_opt.text(0.02, 0.96, '', transform=ax_opt.transAxes, va='top')

        on_sc = ax_on.scatter(pts_x, pts_y, c=np.zeros(n_cells),
                              cmap='YlOrRd', vmin=0, vmax=1, s=5)
        ax_on.set_xlim(xg.min(), xg.max())
        ax_on.set_ylim(yg.min(), yg.max())
        ax_on.set_title('ON spikes')

        off_sc = ax_off.scatter(pts_x, pts_y, c=np.zeros(n_cells),
                                cmap='YlGnBu', vmin=0, vmax=1, s=5)
        ax_off.set_xlim(xg.min(), xg.max())
        ax_off.set_ylim(yg.min(), yg.max())
        ax_off.set_title('OFF spikes')

        on_global = max(self.spikes_on.max(), 1)
        off_global = max(self.spikes_off.max(), 1)

        if has_decoder:
            S0 = self.S_hat_history[0]
            vmax0 = max(abs(S0).max(), 1e-3)
            rec_img = ax_rec.imshow(S0, cmap='gray_r', vmin=0, vmax=vmax0, origin='upper')
            ax_rec.set_title(r'Reconstruction $\hat{optotype}$')
            ax_rec.set_xticks([]); ax_rec.set_yticks([])

        def _update(frame):
            cx, cy = self.walk[frame]
            opt_img.set_extent([cx - half_x, cx + half_x, cy - half_y, cy + half_y])
            walk_line.set_data(self.walk[:frame + 1, 0], self.walk[:frame + 1, 1])
            fovea.set_data([cx], [cy])
            time_text.set_text(f't = {frame * self.dt:.3f} s')
            on_sc.set_array(self.spikes_on[frame].astype(float) / on_global)
            off_sc.set_array(self.spikes_off[frame].astype(float) / off_global)
            if has_decoder:
                S_hat = self.S_hat_history[frame]
                rec_img.set_data(S_hat)
                vmin, vmax = float(S_hat.min()), float(S_hat.max())
                if vmax - vmin < 1e-6:
                    vmax = vmin + 1e-3
                rec_img.set_clim(vmin, vmax)
            return opt_img, walk_line, fovea, on_sc, off_sc

        anim = animation.FuncAnimation(fig, _update,
                                       frames=self.n_steps + 1,
                                       interval=interval, blit=False)
        plt.tight_layout()
        if save_path:
            writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
            anim.save(save_path, writer=writer, fps=1000 // interval, dpi=120)
        return anim


if __name__ == "__main__":
    image_path = "images_c/myopia_0p0D.png"


    optotype = load_image_tensor(image_path, image_size=32, invert=True, deblur=False)
    print(f"Loaded image from: {image_path}")

    seed = 55
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    mean_offset = float(optotype.mean())
    optotype_centered = optotype - mean_offset

    sim = NeuralEncoder(dx=0.05, dy=0.05, dt=0.01, ds=0.3, D_diff=5.0, img_size=32)
    sim.fit(optotype_centered, blur_sigma=0.0)
    sim.simulate_random_walk(T=0.750)
    sim.compute_activations(grid_range=10, grid_resolution=30)
    sim.decode(
        mean_offset=mean_offset,
        n_particles=100, n_samples=100,
        beta=0.001, gamma=10.0,
        adam_iter=20, lr=1e-2,
        anchor_weight=1.0,
        hessian_tau=10.0,
        hessian_every=1,
        verbose=True,
    )

    anim = sim.animate(interval=100, save_path="C_de_Landolt.gif")
    plt.show()
