"""
NeuralCortexSimple_single_var.py — Single-variable θ decoding via Hessian-preconditioned optimisation.

Encoding:  Image S → ON/OFF ganglion-cell Poisson spikes via GLM on hex grid.
Decoding:  Spikes → reconstructed S(θ) via particle-filtered,
           Hessian-preconditioned Poisson maximum-likelihood over θ alone.
           θ = 2·arctan(α), optimise α ∈ ℝ  →  θ ∈ (-π, π).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import random
from scipy.ndimage import rotate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def load_image_tensor(image_path, image_size=32, invert=True):
    img = Image.open(image_path).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    return torch.from_numpy(arr).float()


def S_theta(S_base, theta_rad):
    """
    Differentiable rotation of S_base by theta_rad (radians) CCW.

    Args:
        S_base: (H, W) tensor
        theta_rad: scalar tensor (fully differentiable)

    Returns:
        (H, W) tensor
    """
    if S_base.dim() == 2:
        S_base = S_base.unsqueeze(0).unsqueeze(0)
    _, _, H, W = S_base.shape
    cos = torch.cos(theta_rad)
    sin = torch.sin(theta_rad)
    rot_mat = torch.stack([
        cos, -sin, torch.zeros_like(cos),
        sin,  cos, torch.zeros_like(cos),
    ], dim=-1).reshape(1, 2, 3)
    grid = F.affine_grid(rot_mat, S_base.size(), align_corners=False)
    out = F.grid_sample(S_base, grid, mode='bilinear',
                        padding_mode='zeros', align_corners=False)
    return out.squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# HexEncoder
# ---------------------------------------------------------------------------

class HexEncoder:
    def __init__(self, dx=0.05, dy=0.05, dt=0.01, D_diff=5.0, ds=0.3,
                 img_size=32, lambda0=10.0, lambda1=100.0, device=None):
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.D_diff = D_diff
        self.ds = ds
        self.img_size = img_size
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.half_n = img_size // 2
        self.device = device or torch.device("cpu")

        self.S_true = None
        self.S_base = None          # fixed unrotated reference image
        self.n_steps = None
        self.walk = None
        self.ganglion_x = None
        self.ganglion_y = None
        self.n_cells = None
        self.spikes_on = None
        self.spikes_off = None
        self.S_hat_history = None
        self.theta_history = None   # (T,) decoded θ at each step
        self.q_particles = None
        self.q_weights = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def fit(self, optotype):
        if optotype.dim() == 3:
            optotype = optotype.squeeze(0)
        h, w = optotype.shape
        self.half_n = h // 2
        self.S_true = optotype.to(self.device)
        self.S_base = self.S_true.detach().clone()

    def simulate_random_walk(self, T):
        self.n_steps = int(T / self.dt)
        sigma = np.sqrt(self.D_diff * self.dt)
        disp = np.random.normal(0.0, sigma, size=(self.n_steps, 2))
        self.walk = np.vstack([np.zeros((1, 2)), np.cumsum(disp, axis=0)])

    # ------------------------------------------------------------------
    # Core encoding computation
    # ------------------------------------------------------------------

    def _spatial_kernels(self, cx, cy):
        B = cx.shape[0]
        H = W = self.img_size
        sigma_s = 0.5 * self.ds
        sigma_e = 0.203 * self.ds
        sigma2 = sigma_s ** 2 + sigma_e ** 2
        neg_half_inv_sigma2 = -0.5 / sigma2
        px_y = (torch.arange(H, device=self.device, dtype=torch.float32)
                + 0.5 - self.half_n) * self.dy
        px_x = (torch.arange(W, device=self.device, dtype=torch.float32)
                + 0.5 - self.half_n) * self.dx
        xg = torch.as_tensor(self.ganglion_x, device=self.device, dtype=torch.float32)
        yg = torch.as_tensor(self.ganglion_y, device=self.device, dtype=torch.float32)
        diff_x = cx[:, None, None] + px_x[None, None, :] - xg[None, :, None]
        diff_y = cy[:, None, None] + px_y[None, None, :] - yg[None, :, None]
        gx_w = torch.exp((-0.5 / sigma2) * diff_x * diff_x)
        gy_w = torch.exp((-0.5 / sigma2) * diff_y * diff_y)
        return gx_w, gy_w

    def _glm_rates(self, S, cx, cy, gx_w=None, gy_w=None):
        if gx_w is None or gy_w is None:
            gx_w, gy_w = self._spatial_kernels(cx, cy)
        sigma_s = 0.5 * self.ds
        sigma_e = 0.203 * self.ds
        sigma2 = sigma_s ** 2 + sigma_e ** 2
        inv_two_pi_sigma2 = 1.0 / (2.0 * np.pi * sigma2)
        log_ratio = np.log(self.lambda1 / self.lambda0)
        tmp = torch.matmul(gy_w, S)
        c_raw = (tmp * gx_w).sum(dim=2) * inv_two_pi_sigma2
        g_norm = c_raw.detach().amax(dim=1, keepdim=True).clamp(min=1e-9)
        c = (c_raw / g_norm).clamp(0.0, 1.0)
        lam_on  = self.lambda0 * torch.exp(log_ratio * c)
        lam_off = self.lambda0 * torch.exp(log_ratio * (1.0 - c))
        return lam_on, lam_off

    # ------------------------------------------------------------------
    # Encoding: image → spikes
    # ------------------------------------------------------------------

    def encode_spikes(self, grid_range=10.0, grid_resolution=30):
        self.ganglion_x, self.ganglion_y = generate_hex_grid(grid_range, grid_resolution)
        self.n_cells = len(self.ganglion_x)
        T = self.n_steps + 1
        all_cx = torch.as_tensor(self.walk[:, 0], device=self.device, dtype=torch.float32)
        all_cy = torch.as_tensor(self.walk[:, 1], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            lam_on, lam_off = self._glm_rates(self.S_true, all_cx, all_cy)
        self.spikes_on  = np.zeros((T, self.n_cells), dtype=int)
        self.spikes_off = np.zeros((T, self.n_cells), dtype=int)
        for t in range(T):
            self.spikes_on[t]  = np.random.poisson(lam_on[t].cpu().numpy() * self.dt)
            self.spikes_off[t] = np.random.poisson(lam_off[t].cpu().numpy() * self.dt)

    # ------------------------------------------------------------------
    # Particle filter
    # ------------------------------------------------------------------

    def _propagate_particles(self, particles, S, spikes_on_t, spikes_off_t):
        dev = self.device
        n_p = particles.shape[0]
        sigma = np.sqrt(self.D_diff * self.dt)
        noise = torch.randn(n_p, 2, device=dev) * sigma
        new_p = particles + noise
        with torch.no_grad():
            lam_on, lam_off = self._glm_rates(S, new_p[:, 0], new_p[:, 1])
            dt = self.dt
            eps = 1e-12
            log_w = (spikes_on_t * torch.log(lam_on * dt + eps) - lam_on * dt
                     + spikes_off_t * torch.log(lam_off * dt + eps) - lam_off * dt
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
        idx = torch.searchsorted(csum, u).clamp(0, n_p - 1)
        return particles[idx], torch.ones(n_p, device=dev) / n_p

    def _sample_positions(self, particles, weights, n_samples):
        dev = self.device
        n_p = weights.shape[0]
        csum = torch.cumsum(weights, dim=0)
        u = torch.rand(n_samples, device=dev)
        idx = torch.searchsorted(csum, u).clamp(0, n_p - 1)
        return particles[idx]

    # ------------------------------------------------------------------
    # Hessian (Gauss-Newton per-sample loop — first derivatives only)
    # ------------------------------------------------------------------

    def compute_hessian(self, alpha_val, samples, r_on, r_off):
        M = samples.shape[0]
        N = 1
        dt = self.dt
        eps = 1e-12

        cx = samples[:, 0]
        cy = samples[:, 1]
        gx_w, gy_w = self._spatial_kernels(cx, cy)

        J = torch.zeros(M, N, device=self.device, dtype=torch.float32)

        for i in range(M):
            cx_i = cx[i:i + 1]
            cy_i = cy[i:i + 1]
            gx_i = gx_w[i:i + 1]
            gy_i = gy_w[i:i + 1]

            def loss_fn(a, _cx=cx_i, _cy=cy_i, _gx=gx_i, _gy=gy_i):
                theta = 2.0 * torch.atan(a)
                img = S_theta(self.S_base, theta)
                lo, lf = self._glm_rates(img, _cx, _cy, _gx, _gy)
                return ((lo * dt - r_on * torch.log(lo * dt + eps)).sum()
                        + (lf * dt - r_off * torch.log(lf * dt + eps)).sum()) / M

            J[i] = grad(loss_fn)(alpha_val).detach()

        H = J.T @ J
        return 0.5 * (H + H.T)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_sequential(self, n_particles=80, n_samples=30,
                          n_iter=20, lr=1e-2, beta=0.05,
                          hessian_every=1, hessian_tau=0.5,
                          anchor_weight=1.0, True_Path=True,
                          verbose=True):
        dev = self.device
        T = self.n_steps + 1
        dt = self.dt
        eps = 1e-12

        spikes_on_t = torch.as_tensor(self.spikes_on, dtype=torch.float32, device=dev)
        spikes_off_t = torch.as_tensor(self.spikes_off, dtype=torch.float32, device=dev)

        # Scalar parameter  α ∈ ℝ  →  θ = 2·arctan(α) ∈ (-π, π)
        alpha_param = nn.Parameter(torch.zeros(1, device=dev))
        Hessian = torch.zeros((1, 1), device=dev, dtype=torch.float32)
        optimizer = torch.optim.Adam([alpha_param], lr=lr)

        particles = torch.zeros(n_particles, 2, device=dev)
        weights = torch.ones(n_particles, device=dev) / n_particles

        self.q_particles = torch.zeros(T, n_particles, 2, device=dev)
        self.q_weights = torch.zeros(T, n_particles, device=dev)
        self.S_hat_history = np.zeros((T, self.img_size, self.img_size))
        self.theta_history = np.zeros(T)

        for t in range(T):
            cur_aw = 0.0 if t == 0 else anchor_weight

            # ---- particle filter ----
            if t > 0:
                theta_prev = 2.0 * torch.atan(alpha_param.detach())
                S_prev = S_theta(self.S_base, theta_prev)
                r_on_t  = spikes_on_t[t]
                r_off_t = spikes_off_t[t]
                if not True_Path:
                    particles, weights = self._propagate_particles(
                        particles, S_prev, r_on_t, r_off_t)
                    particles, weights = self._resample(particles, weights)

            self.q_particles[t] = particles
            self.q_weights[t]  = weights

            # ---- sample candidate eye positions ----
            if True_Path:
                pos = torch.as_tensor(self.walk[t], device=dev, dtype=torch.float32)
                samples = pos.unsqueeze(0).repeat(n_samples, 1)
            else:
                samples = self._sample_positions(particles, weights, n_samples)
            cx = samples[:, 0]
            cy = samples[:, 1]
            gx_w, gy_w = self._spatial_kernels(cx, cy)

            alpha_anchor = alpha_param.detach().clone().flatten()
            theta_anchor = 2.0 * torch.atan(alpha_anchor)
            r_on_t  = spikes_on_t[t]
            r_off_t = spikes_off_t[t]

            # ---- Adam optimisation ----
            last_loss = 0.0
            for _ in range(n_iter):
                optimizer.zero_grad()

                theta_rad = 2.0 * torch.atan(alpha_param)
                S_cur = S_theta(self.S_base, theta_rad)

                lam_on, lam_off = self._glm_rates(S_cur, cx, cy, gx_w, gy_w)

                # Poisson NLL
                Er = ((lam_on * dt - r_on_t * torch.log(lam_on * dt + eps)).sum(dim=1)
                      + (lam_off * dt - r_off_t * torch.log(lam_off * dt + eps)).sum(dim=1)
                      ).mean()

                # Trust-region (scalar)
                alpha_flat = alpha_param.reshape(-1)
                diff = alpha_flat - alpha_anchor
                Eg = 0.5 * cur_aw * (diff @ (Hessian @ diff))

                # L1 prior on θ:  log p(θ) = -β |θ|
                # Taylor around θ_anchor:
                #   -log p(θ) ≈ β|θ_a| + β·sign(θ_a)·(θ - θ_a)
                log_p_a = -beta * torch.abs(theta_anchor)
                grad_lp = -beta * torch.sign(theta_anchor)
                Ep = -(-log_p_a + (grad_lp * diff).sum())

                loss = Er + Eg + Ep
                loss.backward()
                optimizer.step()
                last_loss = loss.item()

            # ---- update Hessian (EMA) ----
            if t % hessian_every == 0:
                alpha_new = alpha_param.detach().clone().flatten()
                H_new = self.compute_hessian(alpha_new, samples, r_on_t, r_off_t)
                decay = float(np.exp(-self.dt / hessian_tau))
                Hessian = decay * Hessian + H_new

            # ---- record ----
            theta_out = 2.0 * torch.atan(alpha_param.detach())
            S_out = S_theta(self.S_base, theta_out)
            self.S_hat_history[t] = S_out.cpu().numpy()
            self.theta_history[t] = float(theta_out.cpu().item())

            if verbose and t % max(T // 10, 1) == 0:
                print(f"[decode seq] t={t}/{T - 1}  θ={self.theta_history[t]:+.3f} rad  loss={last_loss:.3f}")

        return self.S_hat_history

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def animate(self, interval=80, save_path=None):
        xg, yg = self.ganglion_x, self.ganglion_y
        half_x = self.half_n * self.dx
        half_y = self.half_n * self.dy
        n_cells = len(xg)
        has_decoder = self.S_hat_history is not None
        n_panels = 4 if has_decoder else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        ax_opt, ax_on, ax_off = axes[0], axes[1], axes[2]
        ax_rec = axes[3] if has_decoder else None
        for ax in axes:
            ax.set_aspect("equal")
        S_disp = self.S_true.cpu().numpy()
        cx0, cy0 = self.walk[0]
        opt_img = ax_opt.imshow(
            S_disp,
            extent=[cx0 - half_x, cx0 + half_x, cy0 - half_y, cy0 + half_y],
            origin="upper", cmap="gray_r", alpha=0.9,
        )
        walk_line, = ax_opt.plot([], [], color="steelblue", lw=0.7)
        fovea, = ax_opt.plot([], [], "b+", ms=12, mew=2)
        ax_opt.set_xlim(xg.min(), xg.max())
        ax_opt.set_ylim(yg.min(), yg.max())
        ax_opt.set_title("Stimulus + eye path")
        time_text = ax_opt.text(0.02, 0.96, "", transform=ax_opt.transAxes, va="top")
        on_sc = ax_on.scatter(xg, yg, c=np.zeros(n_cells),
                              cmap="YlOrRd", vmin=0, vmax=1, s=5)
        ax_on.set_xlim(xg.min(), xg.max())
        ax_on.set_ylim(yg.min(), yg.max())
        ax_on.set_title("ON spikes")
        off_sc = ax_off.scatter(xg, yg, c=np.zeros(n_cells),
                                cmap="YlGnBu", vmin=0, vmax=1, s=5)
        ax_off.set_xlim(xg.min(), xg.max())
        ax_off.set_ylim(yg.min(), yg.max())
        ax_off.set_title("OFF spikes")
        on_global  = max(self.spikes_on.max(), 1)
        off_global = max(self.spikes_off.max(), 1)
        if has_decoder:
            S0 = self.S_hat_history[0]
            vmax0 = max(abs(S0).max(), 1e-3)
            rec_img = ax_rec.imshow(S0, cmap="gray_r", vmin=0, vmax=vmax0, origin="upper")
            ax_rec.set_title(r"Reconstruction $\hat{S}$")
            ax_rec.set_xticks([])
            ax_rec.set_yticks([])

        def _update(frame):
            cx, cy = self.walk[frame]
            opt_img.set_extent([cx - half_x, cx + half_x, cy - half_y, cy + half_y])
            walk_line.set_data(self.walk[:frame + 1, 0], self.walk[:frame + 1, 1])
            fovea.set_data([cx], [cy])
            time_text.set_text(f"t = {frame * self.dt:.3f} s")
            on_sc.set_array(self.spikes_on[frame].astype(float) / on_global)
            off_sc.set_array(self.spikes_off[frame].astype(float) / off_global)
            artists = [opt_img, walk_line, fovea, on_sc, off_sc]
            if has_decoder:
                S_hat = self.S_hat_history[frame]
                rec_img.set_data(S_hat)
                vmin, vmax = float(S_hat.min()), float(S_hat.max())
                if vmax - vmin < 1e-6:
                    vmax = vmin + 1e-3
                rec_img.set_clim(vmin, vmax)
                artists.append(rec_img)
            return artists

        anim = animation.FuncAnimation(
            fig, _update, frames=self.n_steps + 1,
            interval=interval, blit=False,
        )
        plt.tight_layout()
        if save_path:
            writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
            anim.save(save_path, writer=writer, fps=1000 // interval, dpi=120)
        return anim


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    image_path = "images_c/myopia_0p0D.png"

    optotype = load_image_tensor(image_path, image_size=32, invert=True)
    print(f"Loaded image from: {image_path}")

    seed = 55
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    enc = HexEncoder(dx=0.07, dy=0.07, dt=0.01, ds=0.4, D_diff=10.0, img_size=32)
    enc.fit(optotype)
    enc.simulate_random_walk(T=0.750)
    enc.encode_spikes(grid_range=10, grid_resolution=25)

    enc.decode_sequential(
        n_particles=100, n_samples=100,
        n_iter=20, lr=1e-2,
        beta=0.001,
        anchor_weight=1.0, hessian_tau=10.0,
        hessian_every=1, verbose=True,
    )

    anim = enc.animate(interval=100, save_path="reconstruction_single_var.gif")
    plt.show()
