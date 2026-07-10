"""
Calculate_error.py — Sweeps images × seeds × DX × D_diff, computing the
phase-correlation error vs time for the reconstructed frames.

Reconstructed frames are processed purely in RAM (no SSD writes): the exact
PNG normalise → invert → uint8-quantise transform is replicated in memory
before correlating against the ground-truth reference image.
"""

import os
import glob
import re
import numpy as np
import torch
import random
from tqdm import tqdm
import csv

from PIL import Image
from skimage.registration import phase_cross_correlation

from NeuralCortexSimple import (
    HexEncoder,
    load_image_tensor,
)


def phase_corr_in_ram(S_hat_history, ref_np):
    """Phase-correlation error per frame, computed in memory (no disk I/O)."""
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


if __name__ == "__main__":
    pix = 1/28
    D_list = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0]
    DX_list = [pix, 1.5*pix, 2*pix, 2.5*pix, 3*pix]
    seeds = range(10, 20)
    image_paths = sorted(glob.glob("images_c1/*.png"))
    ds = 0.4

    base_dir = "results/phase_corr"
    os.makedirs(base_dir, exist_ok=True)

    csv_path = os.path.join(base_dir, "data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "D", "seed", "DX", "D_diff", "time", "error"])

    for image_path in tqdm(image_paths, desc="Images"):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        m = re.match(r"myopia_(\d+)p(\d+)D", basename)
        p_val = float(m.group(1))
        d_val = float(m.group(2))

        optotype = load_image_tensor(image_path, image_size=32, invert=True)
        ref_np = np.array(Image.open(image_path).convert("L"), dtype=np.float64)

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
                            beta=0.001, gamma=10.0,
                            anchor_weight=1.0, hessian_tau=10.0,
                            hessian_every=1, verbose=False,
                        )

                        errors = phase_corr_in_ram(enc.S_hat_history, ref_np)
                        times = np.arange(len(errors)) * enc.dt

                        with open(csv_path, "a", newline="") as f:
                            writer = csv.writer(f)
                            for t, e in zip(times, errors):
                                writer.writerow([p_val, d_val, seed, DX, D_diff, t, e])

    print(f"\nDone. Results saved to '{base_dir}/'")
