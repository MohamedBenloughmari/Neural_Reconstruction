"""
NeuralCortexSimple.py — Hex-grid retinal encoding/decoding (core calculations).

Encoding:  Image S → ON/OFF ganglion-cell Poisson spikes via GLM on hex grid.
Decoding:  Spikes → reconstructed S via particle-filtered,
           Hessian-preconditioned Poisson maximum-likelihood.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.func import grad
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_hex_grid(grid_range, grid_resolution):
    """Hexagonal lattice of ganglion-cell positions."""
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


# ---------------------------------------------------------------------------
# HexEncoder
# ---------------------------------------------------------------------------

class HexEncoder:
    """
    Neural encoding/decoding on a hexagonal ganglion-cell lattice.

    The image S (img_size × img_size, values in [0, 1]) stays in its natural
    orientation throughout — no flips, no transposes.  Rows map to y,
    columns to x in the physical visual field.
    """

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
        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        # Populated by fit / encode / decode
        self.S_true = None          # ground-truth image (H, W) torch tensor
        self.n_steps = None
        self.walk = None           # (T, 2) eye path
        self.ganglion_x = None     # (n_cells,) np array
        self.ganglion_y = None     # (n_cells,) np array
        self.n_cells = None
        self.spikes_on = None      # (T, n_cells) np int
        self.spikes_off = None     # (T, n_cells) np int
        self.S_hat_history = None  # (T, H, W) reconstruction
        self.q_particles = None
        self.q_weights = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def fit(self, optotype):
        """Store the ground-truth image (natural orientation, rows=y, cols=x)."""
        if optotype.dim() == 3:
            optotype = optotype.squeeze(0)
        h, w = optotype.shape
        self.half_n = h // 2
        self.S_true = optotype.to(self.device)

    def simulate_random_walk(self, T):
        """Brownian-motion eye path with diffusion coefficient D_diff."""
        self.n_steps = int(T / self.dt)
        sigma = np.sqrt(self.D_diff * self.dt)
        disp = np.random.normal(0.0, sigma, size=(self.n_steps, 2))
        self.walk = np.vstack([np.zeros((1, 2)), np.cumsum(disp, axis=0)])

    # ------------------------------------------------------------------
    # Core encoding computation
    # ------------------------------------------------------------------

    def _spatial_kernels(self, cx, cy):
        """
        Gaussian receptive-field weights for a batch of fixation points.

        Parameters
        ----------
        cx, cy : (B,) tensors — fixation coordinates

        Returns
        -------
        gx_w : (B, n_cells, W) — horizontal Gaussian  (columns → x)
        gy_w : (B, n_cells, H) — vertical   Gaussian  (rows    → y)
        """
        B = cx.shape[0]
        H = W = self.img_size

        sigma_s = 0.5 * self.ds
        sigma_e = 0.203 * self.ds
        sigma2 = sigma_s ** 2 + sigma_e ** 2
        neg_half_inv_sigma2 = -0.5 / sigma2

        px_y = (torch.arange(H, device=self.device, dtype=torch.float32)
                + 0.5 - self.half_n) * self.dy                # (H,)  rows → y
        px_x = (torch.arange(W, device=self.device, dtype=torch.float32)
                + 0.5 - self.half_n) * self.dx                # (W,)  cols → x

        xg = torch.as_tensor(self.ganglion_x, device=self.device, dtype=torch.float32)
        yg = torch.as_tensor(self.ganglion_y, device=self.device, dtype=torch.float32)

        diff_x = cx[:, None, None] + px_x[None, None, :] - xg[None, :, None]   # (B, n_cells, W)
        diff_y = cy[:, None, None] + px_y[None, None, :] - yg[None, :, None]   # (B, n_cells, H)

        gx_w = torch.exp((-0.5 / sigma2) * diff_x * diff_x)                 # (B, n_cells, W)
        gy_w = torch.exp((-0.5 / sigma2)* diff_y * diff_y)                 # (B, n_cells, H)
        return gx_w, gy_w

    def _glm_rates(self, S, cx, cy, gx_w=None, gy_w=None):
        """
        ON/OFF firing rates for every ganglion cell (batched).

        Parameters
        ----------
        S : (H, W) tensor — image  (rows=y, cols=x)
        cx, cy : (B,) tensors — fixation points
        gx_w, gy_w : optional precomputed spatial kernels

        Returns
        -------
        lam_on, lam_off : (B, n_cells) tensors
        """
        if gx_w is None or gy_w is None:
            gx_w, gy_w = self._spatial_kernels(cx, cy)

        sigma_s = 0.5 * self.ds
        sigma_e = 0.203 * self.ds
        sigma2 = sigma_s ** 2 + sigma_e ** 2
        inv_two_pi_sigma2 = 1.0 / (2.0 * np.pi * sigma2)
        log_ratio = np.log(self.lambda1 / self.lambda0)

        tmp = torch.matmul(gy_w, S)                                          # (B, n_cells, W)  integrate rows (y)
        c_raw = (tmp * gx_w).sum(dim=2) * inv_two_pi_sigma2                  # (B, n_cells)     integrate cols (x)

        g_norm = c_raw.detach().amax(dim=1, keepdim=True).clamp(min=1e-9)
        c = (c_raw / g_norm).clamp(0.0, 1.0)                                 # contrast ∈ [0, 1]

        lam_on  = self.lambda0 * torch.exp(log_ratio * c)
        lam_off = self.lambda0 * torch.exp(log_ratio * (1.0 - c))
        return lam_on, lam_off

    # ------------------------------------------------------------------
    # Encoding: image → spikes
    # ------------------------------------------------------------------

    def encode_spikes(self, grid_range=10.0, grid_resolution=30):
        """Generate hex grid and Poisson spikes for every time step."""
        self.ganglion_x, self.ganglion_y = generate_hex_grid(grid_range, grid_resolution)
        self.n_cells = len(self.ganglion_x)

        T = self.n_steps + 1
        all_cx = torch.as_tensor(self.walk[:, 0], device=self.device, dtype=torch.float32)
        all_cy = torch.as_tensor(self.walk[:, 1], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            lam_on, lam_off = self._glm_rates(self.S_true, all_cx, all_cy)   # (T, n_cells)

        self.spikes_on  = np.zeros((T, self.n_cells), dtype=int)
        self.spikes_off = np.zeros((T, self.n_cells), dtype=int)
        for t in range(T):
            self.spikes_on[t]  = np.random.poisson(lam_on[t].cpu().numpy() * self.dt)
            self.spikes_off[t] = np.random.poisson(lam_off[t].cpu().numpy() * self.dt)

    # ------------------------------------------------------------------
    # Particle filter (eye-position inference)
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
    # Hessian (Gauss-Newton, basic unoptimised loop)
    # ------------------------------------------------------------------

    def compute_hessian(self, S_flat, samples, r_on, r_off):
        """
        Gauss-Newton Hessian of the Poisson NLL w.r.t. flattened S.

        Parameters
        ----------
        S_flat : (N,) — current image flattened  (N = H * W)
        samples : (M, 2) — fixation positions
        r_on, r_off : (n_cells,) — spike counts at current time step

        Returns
        -------
        H : (N, N) —  J^T J
        """
        M = samples.shape[0]
        N = S_flat.numel()
        H, W = self.img_size, self.img_size
        dt = self.dt
        eps = 1e-12

        cx = samples[:, 0]
        cy = samples[:, 1]
        gx_w, gy_w = self._spatial_kernels(cx, cy)                         # (M, n_cells, W/H)

        J = torch.zeros(M, N, device=self.device, dtype=torch.float32)

        for i in range(M):
            cx_i = cx[i:i + 1]
            cy_i = cy[i:i + 1]
            gx_i = gx_w[i:i + 1]
            gy_i = gy_w[i:i + 1]

            def loss_fn(s, _cx=cx_i, _cy=cy_i, _gx=gx_i, _gy=gy_i):
                img = s.reshape(H, W)
                lo, lf = self._glm_rates(img, _cx, _cy, _gx, _gy)
                return ((lo * dt - r_on * torch.log(lo * dt + eps)).sum()
                        + (lf * dt - r_off * torch.log(lf * dt + eps)).sum()) / M

            J[i] = grad(loss_fn)(S_flat).detach()

        H = J.T @ J
        return 0.5 * (H + H.T)

    # ------------------------------------------------------------------
    # Decoding: spikes → reconstructed image
    # ------------------------------------------------------------------

    def decode_sequential(self, n_particles=80, n_samples=30,
                          n_iter=20, lr=1e-2, beta=0.05, gamma=0.1,
                          hessian_every=1, hessian_tau=0.5,
                          anchor_weight=1.0, True_Path=True, verbose=True):
        """
        Online decoding — update S at each time step.

        Uses a particle filter for eye-position inference and
        Hessian-preconditioned Adam with a trust-region anchoring term.
        """
        dev = self.device
        T = self.n_steps + 1
        N = self.img_size * self.img_size
        dt = self.dt
        eps = 1e-12

        # Convert spikes to torch
        spikes_on_t = torch.as_tensor(self.spikes_on, dtype=torch.float32, device=dev)
        spikes_off_t = torch.as_tensor(self.spikes_off, dtype=torch.float32, device=dev)

        # Initialise parameters
        S_param = nn.Parameter(torch.full((self.img_size, self.img_size), 0.5, device=dev))
        Hessian = torch.zeros((N, N), device=dev, dtype=torch.float32)
        optimizer = torch.optim.Adam([S_param], lr=lr)

        # Particle filter
        particles = torch.zeros(n_particles, 2, device=dev)
        weights = torch.ones(n_particles, device=dev) / n_particles

        # History
        self.q_particles = torch.zeros(T, n_particles, 2, device=dev)
        self.q_weights = torch.zeros(T, n_particles, device=dev)
        self.S_hat_history = np.zeros((T, self.img_size, self.img_size))

        for t in range(T):
            cur_aw = 0.0 if t == 0 else anchor_weight

            # ---- particle filter ----
            if t > 0:
                S_prev = S_param.detach()
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
                samples = self._sample_positions(particles, weights, n_samples)          # (n_samples, 2)
            cx = samples[:, 0]
            cy = samples[:, 1]
            gx_w, gy_w = self._spatial_kernels(cx, cy)                              # (n_samples, n_cells, W/H)

            S_anchor = S_param.detach().clone().flatten()
            r_on_t  = spikes_on_t[t]
            r_off_t = spikes_off_t[t]

            # ---- Adam optimisation with Hessian trust region ----
            last_loss = 0.0
            for _ in range(n_iter):
                optimizer.zero_grad()

                lam_on, lam_off = self._glm_rates(S_param, cx, cy, gx_w, gy_w)      # (n_samples, n_cells)

                # Poisson NLL averaged over samples
                Er = ((lam_on * dt - r_on_t * torch.log(lam_on * dt + eps)).sum(dim=1)
                      + (lam_off * dt - r_off_t * torch.log(lam_off * dt + eps)).sum(dim=1)
                      ).mean()

                # Hessian trust region
                S_flat = S_param.reshape(-1)
                diff = S_flat - S_anchor
                Eg = 0.5 * cur_aw * (diff @ (Hessian @ diff))

                # L1 prior (first-order Taylor around anchor)
                log_p_a = -beta * torch.sum(torch.abs(S_anchor))
                grad_lp = -beta * torch.sign(S_anchor)
                Ep = -(-log_p_a + (grad_lp * diff).sum())

                # Range penalty  [0, 1]
                range_pen = gamma * (
                    torch.clamp(S_param - 1.0, min=0.0)
                    + torch.clamp(-S_param, min=0.0)
                ).sum()

                loss = Er + Eg + Ep + range_pen
                loss.backward()
                optimizer.step()
                last_loss = loss.item()

            # ---- update Hessian (EMA) ----
            if t % hessian_every == 0:
                S_new = S_param.detach().clone().flatten()
                H_new = self.compute_hessian(S_new, samples, r_on_t, r_off_t)
                decay = float(np.exp(-self.dt / hessian_tau))
                Hessian = decay * Hessian + H_new

            # ---- record ----
            self.S_hat_history[t] = S_param.detach().cpu().numpy()

            if verbose and t % max(T // 10, 1) == 0:
                print(f"[decode seq] t={t}/{T - 1}  loss={last_loss:.3f}")

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

    enc = HexEncoder(dx=0.05, dy=0.05, dt=0.01, ds=0.4, D_diff=5.0, img_size=32)
    enc.fit(optotype)
    enc.simulate_random_walk(T=0.750)
    enc.encode_spikes(grid_range=10, grid_resolution=25)

    enc.decode_sequential(
        n_particles=100, n_samples=100,
        n_iter=20, lr=1e-2,
        beta=0.001, gamma=10.0,
        anchor_weight=1.0, hessian_tau=10.0,
        hessian_every=1, verbose=True,
    )


    anim = enc.animate(interval=100, save_path="reconstruction_simple2.gif")
    plt.show()
