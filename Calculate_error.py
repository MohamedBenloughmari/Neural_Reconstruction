import sys
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from PIL import Image

from neural_cortex import (
    NeuralEncoder,
    letter_to_tensor,
    load_image_tensor,
)

_EPS = 1e-10
DX=0.1
from skimage.registration import phase_cross_correlation


def score_theta(theta, S, T):
    rotated = rotate(T, theta, reshape=False, mode="constant")
    _, error, _ = phase_cross_correlation(
        S.astype(np.float64), rotated.astype(np.float64), normalization=None
    )
    return np.abs(error) ** 2


def optimize_theta(S, T):
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(
        lambda th: score_theta(th, S, T),
        bounds=(0, 360 - _EPS),
        method="bounded",
    )
    return float(res.x)


def save_reconstructed_frames(S_hat_history, output_dir="reconstructed_frames"):
    """Save every frame in S_hat_history as a grayscale PNG."""
    os.makedirs(output_dir, exist_ok=True)
    T = S_hat_history.shape[0]
    for t in range(T):
        frame = S_hat_history[t]
        # Normalise to [0, 255]
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


def phase_corr_against_reference(frames_dir, ref_path):
    ref_img = Image.open(ref_path).convert("L")
    ref_np = np.array(ref_img, dtype=np.float64)

    # Pre-normalise reference once
    ref_norm = (ref_np - ref_np.mean()) / (ref_np.std() + _EPS)

    frame_paths = sorted([p for p in os.listdir(frames_dir) if p.endswith(".png")])
    results = []

    for fname in frame_paths:
        fpath = os.path.join(frames_dir, fname)
        frame_img = Image.open(fpath).convert("L").resize(
            (ref_img.width, ref_img.height), Image.BILINEAR
        )
        frame_np = np.array(frame_img, dtype=np.float64)

        # Phase cross-correlation (shift + error)
        shift, error, phasediff = phase_cross_correlation(
            ref_np, frame_np, normalization=None
        )

        # Normalized cross-correlation coefficient [-1, 1]
        frame_norm = (frame_np - frame_np.mean()) / (frame_np.std() + _EPS)
        ncc = float(np.mean(ref_norm * frame_norm))

        results.append({
            "frame": fname,
            "shift": shift.tolist(),
            "error": float(error),
            "phasediff": float(phasediff),
            "ncc": ncc,
        })
        print(f"  {fname}  shift={shift}  error={error:.4f}  ncc={ncc:.4f}")

    return results



if __name__ == "__main__":
    # argv[1] : optotype image path  (optional)
    # argv[2] : reference image path for phase-corr  (optional, prompted if missing)
    image_path = "images_c/myopia_0p0D.png"
    ref_image_path = "images_c/myopia_0p0D.png"


    optotype = load_image_tensor(
        image_path, image_size=32, invert=True, deblur=False
    )

    seed = 500
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    mean_offset = float(optotype.mean())
    sim = NeuralEncoder(
        dx=DX, dy=DX, dt=0.02, ds=0.3, D_diff=5.0, img_size=32
    )
    sim.fit(optotype - mean_offset, blur_sigma=0.0)
    sim.simulate_random_walk(T=0.750)
    sim.compute_activations(grid_range=10, grid_resolution=30)
    sim.decode(
        mean_offset=mean_offset,
        n_particles=100,
        n_samples=100,
        beta=0.000,
        gamma=10.0,
        adam_iter=20,
        lr=1e-2,
        anchor_weight=1.0,
        hessian_tau=10.0,
        hessian_every=1,
        verbose=False,
    )

    # ── 1. Save reconstructed frames ─────────────────────────────────────────
    frames_dir = save_reconstructed_frames(sim.S_hat_history)

    # ── 2. Theta optimisation ─────────────────────────────────────────────────
    optotype_np = optotype.cpu().numpy()
    T = sim.S_hat_history.shape[0]


    # ── 3. Phase correlation against a user-supplied reference ────────────────
    if ref_image_path is None:
        ref_image_path = input(
            "Enter path to reference image for phase-correlation (or press Enter to skip): "
        ).strip()

    if ref_image_path:
        print(f"\nPhase-correlation of reconstructed frames vs '{ref_image_path}':")
        corr_results = phase_corr_against_reference(frames_dir, ref_image_path)

        # Optional: plot error over time
        errors = [r["error"] for r in corr_results]
        time = np.arange(len(errors)) * 0.02
        plt.figure(figsize=(10, 4))
        plt.plot(time, errors, "r-", lw=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Phase-corr error")
        plt.title(f"Phase-corr error vs '{os.path.basename(ref_image_path)}'")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("phase_corr_error0.png", dpi=150)
        print("Saved phase_corr_error.png")

    anim = sim.animate(interval=100, save_path="O0_2_s_hex.gif")
    plt.show()