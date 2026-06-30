import sys
import os
import numpy as np
import torch
import random
from tqdm import tqdm
import csv

from NeuralCortexTheta import (
    HexEncoder,
    load_image_tensor,
)

DX_list = [0.01, 0.02, 0.03 ,0.04, 0.05]

if __name__ == "__main__":
    D_list = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0]
    seeds = range(10, 20)
    image_path = "images_c/myopia_0p0D.png"

    base_dir = "results2/theta_scan"
    os.makedirs(base_dir, exist_ok=True)

    optotype = load_image_tensor(
        image_path, image_size=32, invert=True
    )

    csv_path = os.path.join(base_dir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "dx", "D_diff", "time", "abs_theta_error"])

    for dx in tqdm(DX_list, desc="dx"):
        for seed in tqdm(seeds, desc="Seeds", leave=False):
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

            for D_diff in tqdm(D_list, desc="D_diff", leave=False):
                enc = HexEncoder(
                    dx=dx, dy=dx, dt=0.02, D_diff=D_diff, ds=0.3, img_size=32
                )
                enc.fit(optotype)
                enc.simulate_random_walk(T=0.750)
                enc.encode_spikes(grid_range=10, grid_resolution=30)
                enc.decode_sequential(
                    n_particles=100, n_samples=100,
                    verbose=False,
                )

                times = np.arange(len(enc.theta_history)) * enc.dt
                abs_theta_errors = np.abs(enc.theta_history)

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for t, e in zip(times, abs_theta_errors):
                        writer.writerow([seed, dx, D_diff, t, e])

    print(f"\nDone. Results saved to '{base_dir}/'")
