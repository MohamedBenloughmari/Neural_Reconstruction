"""
compare_phase_corr.py — Single-example sanity check for the phase-correlation error.

Runs one encode/decode example with the real HexEncoder, then computes the
phase-correlation error vs time TWICE:
  1. SSD method  — write reconstructed frames to disk, reload, correlate, delete.
  2. RAM method  — replicate the same transform purely in memory (no disk).
Both curves are plotted on top of each other to confirm they match.
"""

import os
import shutil
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from skimage.registration import phase_cross_correlation

from NeuralCortexSimple import HexEncoder, load_image_tensor


# ---------------------------------------------------------------------------
# SSD method (frames written to disk, reloaded, correlated, then deleted)
# ---------------------------------------------------------------------------

def save_reconstructed_frames(S_hat_history, output_dir):
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


def phase_corr_from_disk(frames_dir, ref_np):
    frame_paths = sorted(p for p in os.listdir(frames_dir) if p.endswith(".png"))
    errors = []
    for fname in frame_paths:
        frame_img = Image.open(os.path.join(frames_dir, fname)).convert("L")
        if frame_img.size != (ref_np.shape[1], ref_np.shape[0]):
            frame_img = frame_img.resize((ref_np.shape[1], ref_np.shape[0]), Image.BILINEAR)
        frame_np = np.array(frame_img, dtype=np.float64)
        _, error, _ = phase_cross_correlation(ref_np, frame_np, normalization=None)
        errors.append(float(error))
    return errors


# ---------------------------------------------------------------------------
# RAM method (same transform, in memory, no disk)
# ---------------------------------------------------------------------------

def phase_corr_in_ram(S_hat_history, ref_np):
    errors = []
    for frame in S_hat_history:
        f_min, f_max = frame.min(), frame.max()
        if f_max > f_min:
            norm = (frame - f_min) / (f_max - f_min)
        else:
            norm = np.zeros_like(frame)
        norm = 1.0 - norm
        frame_np = np.asarray((norm * 255).astype(np.uint8), dtype=np.float64)
        _, error, _ = phase_cross_correlation(ref_np, frame_np, normalization=None)
        errors.append(float(error))
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    image_path = "images_c/myopia_3p0D.png"
    seed = 12
    DX = 1 / 28
    D_diff = 10.0
    ds = 0.4

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    optotype = load_image_tensor(image_path, image_size=32, invert=True)
    ref_np = np.array(Image.open(image_path).convert("L"), dtype=np.float64)

    enc = HexEncoder(dx=DX, dy=DX, dt=0.01, ds=ds, D_diff=D_diff, img_size=32)
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

    S_hat_history = enc.S_hat_history
    times = np.arange(S_hat_history.shape[0]) * enc.dt

    # ---- SSD method ----
    frames_dir = "out/frames_example"
    save_reconstructed_frames(S_hat_history, frames_dir)
    errors_disk = phase_corr_from_disk(frames_dir, ref_np)
    shutil.rmtree(frames_dir)

    # ---- RAM method ----
    errors_ram = phase_corr_in_ram(S_hat_history, ref_np)

    max_abs_diff = np.max(np.abs(np.array(errors_disk) - np.array(errors_ram)))
    print(f"\nMax |SSD - RAM| error difference: {max_abs_diff:.3e}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, errors_disk, "o-", label="SSD (disk round-trip)", ms=3)
    ax.plot(times, errors_ram, "x--", label="RAM (in-memory)", ms=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase correlation error")
    ax.set_title(f"Phase correlation error vs time  |  {os.path.basename(image_path)}"
                 f"  seed={seed}  DX={DX:.4f}  D_diff={D_diff}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("out", exist_ok=True)
    plot_path = "out/compare_phase_corr.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to '{plot_path}'")
    plt.show()
