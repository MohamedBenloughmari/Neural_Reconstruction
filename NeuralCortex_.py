import numpy as np
import torch
import torch.nn as nn
from torch.func import grad
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import random

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
        self.device = device or (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.S_true = None         
        self.n_steps = None
        self.walk = None          
        self.ganglion_on_x = None     
        self.ganglion_on_y = None     
        self.ganglion_off_x = None     
        self.ganglion_off_y = None     

        self.n_cells = None
        self.spikes_on = None      
        self.spikes_off = None     
        self.S_hat_history = None  
        self.q_particles = None
        self.q_weights = None

    def fit(self, optotype):
        if optotype.dim() == 3:
            optotype = optotype.squeeze(0)
        h, w = optotype.shape
        self.half_n = h // 2
        self.S_true = optotype.to(self.device)

    def simulate_random_walk(self, T):
        self.n_steps = int(T / self.dt)
        sigma = np.sqrt(self.D_diff * self.dt)
        disp = np.random.normal(0.0, sigma, size=(self.n_steps, 2))
        self.walk = np.vstack([np.zeros((1, 2)), np.cumsum(disp, axis=0)])


    def _spatial_kernels(self, cx, cy):

        H = W = self.img_size

        sigma_s = 0.5 * self.ds
        sigma_e = 0.203 * self.ds
        sigma2 = sigma_s ** 2 + sigma_e ** 2


        px_y = (torch.arange(H, device=self.device, dtype=torch.float32)
                + 0.5 - self.half_n) * self.dy                
        px_x = (torch.arange(W, device=self.device, dtype=torch.float32)
                + 0.5 - self.half_n) * self.dx                # (W,)  cols → x

        xg_on = torch.as_tensor(self.ganglion_x, device=self.device, dtype=torch.float32)
        yg_on = torch.as_tensor(self.ganglion_y, device=self.device, dtype=torch.float32)
        xg_off = torch.as_tensor(self.ganglion_x-self.ds/2, device=self.device, dtype=torch.float32)
        yg_off = torch.as_tensor(self.ganglion_y+ self.ds*np.sqrt(3)/6 , device=self.device, dtype=torch.float32)


        diff_on_x = cx[:, None, None] + px_x[None, None, :] - xg_on[None, :, None]   
        diff_on_y = cy[:, None, None] + px_y[None, None, :] - yg_on[None, :, None]   
        diff_off_x = cx[:, None, None] + px_x[None, None, :] - xg_off[None, :, None]   
        diff_off_y = cy[:, None, None] + px_y[None, None, :] - yg_off[None, :, None]  


        gx_on_w = torch.exp((-0.5 / sigma2) * diff_on_x * diff_on_x)                 
        gy_on_w = torch.exp((-0.5 / sigma2)* diff_on_y * diff_on_y) 
        gx_off_w = torch.exp((-0.5 / sigma2) * diff_off_x * diff_off_x)                
        gy_off_w = torch.exp((-0.5 / sigma2)* diff_off_y * diff_off_y)                
        return gx_on_w, gy_on_w,gx_off_w, gy_off_w

    def _glm_rates(self, S, cx, cy, gx_on_w=None, gy_on_w=None,gx_off_w=None, gy_off_w=None):
 
        if gx_on_w is None or gy_on_w is None or gx_off_w is None or gy_off_w is None:
            gx_on_w, gy_on_w,gx_off_w, gy_off_w = self._spatial_kernels(cx, cy)

        sigma_s = 0.5 * self.ds
        sigma_e = 0.203 * self.ds
        sigma2 = sigma_s ** 2 + sigma_e ** 2
        log_ratio = np.log(self.lambda1 / self.lambda0)

        tmp_on = torch.matmul(gy_on_w, S)                                          
        c_raw_on = (tmp_on * gx_on_w).sum(dim=2) * (1.0 / (2.0 * np.pi * sigma2))              
        tmp_off = torch.matmul(gy_off_w, S)                                          
        c_raw_off = (tmp_off * gx_off_w).sum(dim=2) * (1.0 / (2.0 * np.pi * sigma2))              

        g_norm_on = c_raw_on.detach().amax(dim=1, keepdim=True).clamp(min=1e-9)
        c_on = (c_raw_on / g_norm_on).clamp(0.0, 1.0)                                 
        g_norm_off = c_raw_off.detach().amax(dim=1, keepdim=True).clamp(min=1e-9)
        c_off = (c_raw_off / g_norm_off).clamp(0.0, 1.0)                                 

        lam_on  = self.lambda0 * torch.exp(log_ratio * c_on)
        lam_off = self.lambda0 * torch.exp(log_ratio * (1.0 - c_off))
        return lam_on, lam_off


    def encode_spikes(self, grid_range=10.0, grid_resolution=30):
        self.ganglion_x, self.ganglion_y = generate_hex_grid(grid_range, grid_resolution)
        self.ganglion_off_x = self.ganglion_x - self.ds / 2
        self.ganglion_off_y = self.ganglion_y + self.ds * np.sqrt(3) / 6
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

    def compute_hessian(self, S_flat, samples, r_on, r_off):
        M = samples.shape[0]
        N = S_flat.numel()
        H, W = self.img_size, self.img_size
        dt = self.dt
        eps = 1e-12

        cx = samples[:, 0]
        cy = samples[:, 1]
        gx_on_w, gy_on_w, gx_off_w, gy_off_w = self._spatial_kernels(cx, cy)  # (M, n_cells, W/H)

        J = torch.zeros(M, N, device=self.device, dtype=torch.float32)

        for i in range(M):
            cx_i = cx[i:i + 1]
            cy_i = cy[i:i + 1]
            gxo_i = gx_on_w[i:i + 1]
            gyo_i = gy_on_w[i:i + 1]
            gxf_i = gx_off_w[i:i + 1]
            gyf_i = gy_off_w[i:i + 1]

            def loss_fn(s, _cx=cx_i, _cy=cy_i, _gxo=gxo_i, _gyo=gyo_i, _gxf=gxf_i, _gyf=gyf_i):
                img = s.reshape(H, W)
                lo, lf = self._glm_rates(img, _cx, _cy, _gxo, _gyo, _gxf, _gyf)
                return ((lo * dt - r_on * torch.log(lo * dt + eps)).sum()
                        + (lf * dt - r_off * torch.log(lf * dt + eps)).sum()) / M

            J[i] = grad(loss_fn)(S_flat).detach()

        H = J.T @ J
        return 0.5 * (H + H.T)


    def decode_sequential(self, n_particles=80, n_samples=30,
                          n_iter=20, lr=1e-2, beta=0.05, gamma=0.1,
                          hessian_every=1, hessian_tau=0.5,
                          anchor_weight=1.0, verbose=True):
 
        dev = self.device
        T = self.n_steps + 1
        N = self.img_size * self.img_size
        dt = self.dt
        eps = 1e-12

        spikes_on_t = torch.as_tensor(self.spikes_on, dtype=torch.float32, device=dev)
        spikes_off_t = torch.as_tensor(self.spikes_off, dtype=torch.float32, device=dev)

        S_param = nn.Parameter(torch.full((self.img_size, self.img_size), 0.5, device=dev))
        Hessian = torch.zeros((N, N), device=dev, dtype=torch.float32)
        optimizer = torch.optim.Adam([S_param], lr=lr)

        particles = torch.zeros(n_particles, 2, device=dev)
        weights = torch.ones(n_particles, device=dev) / n_particles

        self.q_particles = torch.zeros(T, n_particles, 2, device=dev)
        self.q_weights = torch.zeros(T, n_particles, device=dev)
        self.S_hat_history = np.zeros((T, self.img_size, self.img_size))

        for t in range(T):
            cur_aw = 0.0 if t == 0 else anchor_weight

            if t > 0:
                S_prev = S_param.detach()
                r_on_t  = spikes_on_t[t]
                r_off_t = spikes_off_t[t]
                particles, weights = self._propagate_particles(
                    particles, S_prev, r_on_t, r_off_t)
                particles, weights = self._resample(particles, weights)

            self.q_particles[t] = particles
            self.q_weights[t]  = weights

            samples = self._sample_positions(particles, weights, n_samples)   
            cx = samples[:, 0]
            cy = samples[:, 1]
            gx_on_w, gy_on_w, gx_off_w, gy_off_w = self._spatial_kernels(cx, cy)  

            S_anchor = S_param.detach().clone().flatten()
            r_on_t  = spikes_on_t[t]
            r_off_t = spikes_off_t[t]

            last_loss = 0.0
            for _ in range(n_iter):
                optimizer.zero_grad()

                lam_on, lam_off = self._glm_rates(S_param, cx, cy,
                                                     gx_on_w, gy_on_w,
                                                     gx_off_w, gy_off_w)   

                Er = ((lam_on * dt - r_on_t * torch.log(lam_on * dt + eps)).sum(dim=1)
                      + (lam_off * dt - r_off_t * torch.log(lam_off * dt + eps)).sum(dim=1)
                      ).mean()
                
                S_flat = S_param.reshape(-1)
                diff = S_flat - S_anchor
                Eg = 0.5 * cur_aw * (diff @ (Hessian @ diff))

                log_p_a = -beta * torch.sum(torch.abs(S_anchor))
                grad_lp = -beta * torch.sign(S_anchor)
                Ep = -(-log_p_a + (grad_lp * diff).sum())

                range_pen = gamma * (
                    torch.clamp(S_param - 1.0, min=0.0)
                    + torch.clamp(-S_param, min=0.0)
                ).sum()

                loss = Er + Eg + Ep + range_pen
                loss.backward()
                optimizer.step()
                last_loss = loss.item()

            if t % hessian_every == 0:
                S_new = S_param.detach().clone().flatten()
                H_new = self.compute_hessian(S_new, samples, r_on_t, r_off_t)
                decay = float(np.exp(-self.dt / hessian_tau))
                Hessian = decay * Hessian + H_new

            self.S_hat_history[t] = S_param.detach().cpu().numpy()

            if verbose and t % max(T // 10, 1) == 0:
                print(f"[decode seq] t={t}/{T - 1}  loss={last_loss:.3f}")

        return self.S_hat_history


    def animate(self, interval=80, save_path=None):
        xg, yg = self.ganglion_x, self.ganglion_y
        xg_off, yg_off = self.ganglion_off_x, self.ganglion_off_y
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
                              cmap="YlOrRd", vmin=0, vmax=1, s=2)
        ax_on.set_xlim(xg.min(), xg.max())
        ax_on.set_ylim(yg.min(), yg.max())
        ax_on.set_title("ON spikes")

        off_sc = ax_off.scatter(xg_off, yg_off, c=np.zeros(n_cells),
                                cmap="YlGnBu", vmin=0, vmax=1, s=2)
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
        n_iter=50, lr=1e-3,
        beta=0.001, gamma=10.0,
        anchor_weight=1.0, hessian_tau=10.0,
        hessian_every=1, verbose=True,
    )


    anim = enc.animate(interval=100, save_path="reconstruction_simple.gif")
    plt.show()
