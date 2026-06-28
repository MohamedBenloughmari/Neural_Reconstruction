import sys
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import csv

from neural_cortex import (
    NeuralEncoder,
    load_image_tensor,
)

_EPS = 1e-10
DX = 0.3

from skimage.registration import phase_cross_correlation


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


def phase_corr_against_reference(frames_dir, ref_np, ref_norm):
    frame_paths = sorted([p for p in os.listdir(frames_dir) if p.endswith(".png")])
    results = []
    for fname in frame_paths:
        fpath = os.path.join(frames_dir, fname)
        frame_img = Image.open(fpath).convert("L")
        if frame_img.size != (ref_np.shape[1], ref_np.shape[0]):
            frame_img = frame_img.resize((ref_np.shape[1], ref_np.shape[0]), Image.BILINEAR)
        frame_np = np.array(frame_img, dtype=np.float64)
        _, error, _ = phase_cross_correlation(
            ref_np, frame_np, normalization=None
        )
        results.append({"frame": fname, "error": float(error)})
    return results


if __name__ == "__main__":
    D_list = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0]
    seeds = range(10, 15)
    image_path = "images_c/myopia_0p0D.png"
    ref_image_path = "images_c/myopia_0p0D.png"

    base_dir = f"results/size{DX}"
    os.makedirs(base_dir, exist_ok=True)

    optotype = load_image_tensor(
        image_path, image_size=32, invert=True, deblur=False
    )
    mean_offset = float(optotype.mean())

    ref_img = Image.open(ref_image_path).convert("L")
    ref_np = np.array(ref_img, dtype=np.float64)
    ref_norm = (ref_np - ref_np.mean()) / (ref_np.std() + _EPS)

    csv_path = os.path.join(base_dir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "DX", "D_diff", "time", "error"])

    for seed in tqdm(seeds, desc="Seeds"):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        all_errors = {}

        for D_diff in tqdm(D_list, desc="D_diff", leave=False):
            sim = NeuralEncoder(
                dx=DX, dy=DX, dt=0.02, ds=0.3, D_diff=D_diff, img_size=32
            )
            sim.fit(optotype - mean_offset, blur_sigma=0.0)
            sim.simulate_random_walk(T=0.750)
            sim.compute_activations(grid_range=10, grid_resolution=30)
            sim.decode(
                mean_offset=mean_offset,
                n_particles=100,
                n_samples=100,
                beta=0.001,
                gamma=10.0,
                adam_iter=20,
                lr=1e-2,
                anchor_weight=1.0,
                hessian_tau=10.0,
                hessian_every=1,
                verbose=False,
            )

            frames_dir = os.path.join(base_dir, "frames", f"seed_{seed}", f"D_diff_{D_diff}")
            save_reconstructed_frames(sim.S_hat_history, frames_dir)
            corr_results = phase_corr_against_reference(frames_dir, ref_np, ref_norm)

            errors_list = [r["error"] for r in corr_results]
            times = np.arange(len(errors_list)) * 0.02
            all_errors[D_diff] = (times, errors_list)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for t, e in zip(times, errors_list):
                    writer.writerow([seed, DX, D_diff, t, e])

        fig, ax = plt.subplots(figsize=(12, 6))
        for D_diff, (times, errors_list) in all_errors.items():
            ax.plot(times, errors_list, label=f"D={D_diff}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Phase correlation error")
        ax.set_title(f"Seed {seed}  |  Phase Correlation Error vs Time  |  DX={DX}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="D_diff")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(base_dir, f"aggregated_plot{seed}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(f"\nDone. Results saved to '{base_dir}/'")
