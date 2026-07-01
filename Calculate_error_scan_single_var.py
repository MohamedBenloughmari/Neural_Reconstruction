"""
Calculate_error_scan_single_var.py — Single-variable equivalent of Calculate_error_scan.py.
Sweeps DX × D_diff × seeds, replacing phase_corr error with |θ|.
"""

import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

from NeuralCortexSimple_single_var import (
    HexEncoder,
    load_image_tensor,
)


if __name__ == "__main__":
    D_list = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0]
    DX_list = [0.1, 0.3, 0.5, 0.7, 1.0]
    seeds = range(10, 15)
    image_path = "images_c/myopia_0p0D.png"
    ds = 0.4  # receptive-field size (fixed)

    base_dir = "results/single_var"
    os.makedirs(base_dir, exist_ok=True)

    optotype = load_image_tensor(image_path, image_size=32, invert=True)

    csv_path = os.path.join(base_dir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "DX", "D_diff", "time", "abs_theta"])

    for seed in tqdm(seeds, desc="Seeds"):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        for DX in tqdm(DX_list, desc="DX", leave=False):
            all_errors = {}

            for D_diff in tqdm(D_list, desc="D_diff", leave=False):
                enc = HexEncoder(dx=DX, dy=DX, dt=0.01, ds=ds, D_diff=D_diff, img_size=32)
                enc.fit(optotype)
                enc.simulate_random_walk(T=0.750)
                enc.encode_spikes(grid_range=10, grid_resolution=25)
                enc.decode_sequential(
                    n_particles=100, n_samples=100,
                    n_iter=20, lr=1e-2,
                    beta=0.001,
                    anchor_weight=1.0, hessian_tau=10.0,
                    hessian_every=1, verbose=False,
                )

                theta_abs = np.abs(enc.theta_history)
                times = np.arange(len(theta_abs)) * enc.dt
                all_errors[D_diff] = (times, theta_abs)

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for t, e in zip(times, theta_abs):
                        writer.writerow([seed, DX, D_diff, t, e])

            fig, ax = plt.subplots(figsize=(12, 6))
            for D_diff, (times, errors_list) in all_errors.items():
                ax.plot(times, errors_list, label=f"D={D_diff}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(r"$|\theta|$ (rad)")
            ax.set_title(f"Seed {seed}  |  DX={DX}  |  |θ| vs Time")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="D_diff")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = os.path.join(base_dir, f"plot_seed{seed}_DX{DX}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()

    print(f"\nDone. Results saved to '{base_dir}/'")
