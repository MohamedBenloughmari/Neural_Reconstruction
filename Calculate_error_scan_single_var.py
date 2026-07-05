"""
Calculate_error_scan_single_var.py — Single-variable equivalent of Calculate_error_scan.py.
Sweeps images × seeds × DX × D_diff, replacing phase_corr error with |θ|.
"""

import os
import glob
import re
import numpy as np
import torch
import random
from tqdm import tqdm
import csv

from NeuralCortexSimple_single_var import (
    HexEncoder,
    load_image_tensor,
)


if __name__ == "__main__":
    pix = 1/28
    D_list = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0]
    DX_list = [pix, 1.5*pix, 2*pix, 2.5*pix, 3*pix]
    seeds = range(10, 20)
    image_paths = sorted(glob.glob("images_c/*.png"))
    ds = 0.4

    base_dir = "results/single_var"
    os.makedirs(base_dir, exist_ok=True)

    csv_path = os.path.join(base_dir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "D", "seed", "DX", "D_diff", "time", "abs_theta"])

    for image_path in tqdm(image_paths, desc="Images"):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        m = re.match(r"myopia_(\d+)p(\d+)D", basename)
        p_val = float(m.group(1))
        d_val = float(m.group(2))

        optotype = load_image_tensor(image_path, image_size=32, invert=True)

        for seed in tqdm(seeds, desc="Seeds", leave=False):
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

            for DX in tqdm(DX_list, desc="DX", leave=False):
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

                        with open(csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            for t, e in zip(times, theta_abs):
                                writer.writerow([p_val, d_val, seed, DX, D_diff, t, e])

    print(f"\nDone. Results saved to '{base_dir}/'")
