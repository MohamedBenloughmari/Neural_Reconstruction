import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation


def create_animation(ganglion_x, ganglion_y, half_n, dx, dy,
                     walk, optotype_display, spikes_on, spikes_off,
                     S_hat_history, n_steps, dt,
                     interval=80, save_path=None):
    xg, yg = ganglion_x, ganglion_y
    half_x = half_n * dx
    half_y = half_n * dy
    pts_x, pts_y = xg, yg
    n_cells = len(pts_x)

    has_decoder = S_hat_history is not None
    n_panels = 4 if has_decoder else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    ax_opt, ax_on, ax_off = axes[0], axes[1], axes[2]
    ax_rec = axes[3] if has_decoder else None
    for ax in axes:
        ax.set_aspect('equal')

    cx0, cy0 = walk[0]
    opt_img = ax_opt.imshow(
        optotype_display,
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
                          cmap='YlOrRd', vmin=0, vmax=1, s=18)
    ax_on.set_xlim(xg.min(), xg.max())
    ax_on.set_ylim(yg.min(), yg.max())
    ax_on.set_title('ON spikes')

    off_sc = ax_off.scatter(pts_x, pts_y, c=np.zeros(n_cells),
                            cmap='YlGnBu', vmin=0, vmax=1, s=18)
    ax_off.set_xlim(xg.min(), xg.max())
    ax_off.set_ylim(yg.min(), yg.max())
    ax_off.set_title('OFF spikes')

    on_global = max(spikes_on.max(), 1)
    off_global = max(spikes_off.max(), 1)

    has_decoder = S_hat_history is not None
    if has_decoder:
        S0 = S_hat_history[0]
        vmax0 = max(abs(S0).max(), 1e-3)
        rec_img = ax_rec.imshow(S0, cmap='gray_r', vmin=0, vmax=vmax0, origin='upper')
        ax_rec.set_title(r'Reconstruction $\hat{optotype}$')
        ax_rec.set_xticks([])
        ax_rec.set_yticks([])

    def _update(frame):
        cx, cy = walk[frame]
        opt_img.set_extent([cx - half_x, cx + half_x, cy - half_y, cy + half_y])
        walk_line.set_data(walk[:frame + 1, 0], walk[:frame + 1, 1])
        fovea.set_data([cx], [cy])
        time_text.set_text(f't = {frame * dt:.3f} s')
        on_sc.set_array(spikes_on[frame].astype(float) / on_global)
        off_sc.set_array(spikes_off[frame].astype(float) / off_global)
        if has_decoder:
            S_hat = S_hat_history[frame]
            rec_img.set_data(S_hat)
            vmin, vmax = float(S_hat.min()), float(S_hat.max())
            if vmax - vmin < 1e-6:
                vmax = vmin + 1e-3
            rec_img.set_clim(vmin, vmax)
        return opt_img, walk_line, fovea, on_sc, off_sc

    anim = mpl_animation.FuncAnimation(
        fig, _update, frames=n_steps + 1, interval=interval, blit=False,
    )
    plt.tight_layout()
    if save_path:
        writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
        anim.save(save_path, writer=writer, fps=1000 // interval, dpi=120)
    return anim
