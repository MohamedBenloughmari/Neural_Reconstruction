"""
Calculate_error_single_var.py — Single-variable equivalent of Calculate_error.py.
Instead of phase_corr error, uses |θ| (absolute decoded rotation) as the error metric.
"""

import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image

from NeuralCortexSimple_single_var import (
    HexEncoder,
    load_image_tensor,
)


_EPS = 1e-10


def save_reconstructed_frames(S_hat_history, output_dir="reconstructed_frames"):
    os.makedirs(output_dir, exist_ok=True)
    T = S_hat_history.shape[0]
    for t in range(T):
        frame = S_hat_history[t]
        f_min, f_max = frame.min(), frame.max()
        if f_max > f_min:
            norm = (frame - f_min) / (f_max - f_min)
        else:
            norm = np.zeros_like(frame)
        norm = 1.0 - norm
        img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        img.save(os.path.join(output_dir, f"frame_{t:04d}.png"))
    print(f"Saved {T} frames to '{output_dir}/'")
    return output_dir


if __name__ == "__main__":
    image_path = "images_c/myopia_0p0D.png"

    optotype = load_image_tensor(image_path, image_size=32, invert=True)
    print(f"Loaded image from: {image_path}")

    seed = 55
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    enc = HexEncoder(dx=0.033, dy=0.033, dt=0.01, ds=0.4, D_diff=0.0, img_size=32)
    enc.fit(optotype)
    enc.simulate_random_walk(T=0.750)
    enc.encode_spikes(grid_range=10, grid_resolution=25)

    enc.decode_sequential(
        n_particles=100, n_samples=100,
        n_iter=50, lr=1e-3,
        beta=0.001,
        anchor_weight=1.0, hessian_tau=10.0,
        hessian_every=1, verbose=True,
    )

    # ── Save reconstructed frames ────────────────────────────────────
    frames_dir = save_reconstructed_frames(enc.S_hat_history)

    # ── Error metric: |θ| over time ───────────────────────────────────
    theta_abs = np.abs(enc.theta_history)
    T = len(theta_abs)
    time = np.arange(T) * enc.dt

    plt.figure(figsize=(10, 4))
    plt.plot(time, theta_abs, "b-", lw=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel(r"$|\theta|$ (rad)")
    plt.title(r"Absolute decoded rotation $|\theta|$ over time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("theta_error_single_var.png", dpi=150)
    print("Saved theta_error_single_var.png")

    # Print summary
    print(f"\nMean |θ| = {theta_abs.mean():.4f} rad  = {np.rad2deg(theta_abs.mean()):.2f} deg")
    print(f"Max  |θ| = {theta_abs.max():.4f} rad  = {np.rad2deg(theta_abs.max()):.2f} deg")
    print(f"Var  |θ| = {theta_abs.var():.4f} rad  = {np.rad2deg(theta_abs.var()):.2f} deg")

    anim = enc.animate(interval=100, save_path="reconstruction_single_var.gif")
    plt.show()
