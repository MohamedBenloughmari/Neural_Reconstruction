"""
final20_win.py — Windows-compatible version of final20.py.

Sweeps images x seeds x DX x D_diff across a multiprocessing pool.
Uses 'spawn' start method (required on Windows) with freeze_support().
Heavy imports (torch, NeuralCortexSimple) are delayed into the worker
function so child processes don't redundantly re-import at module level.

Run politely on a shared machine with:
    start /LOW python final20_win.py
"""

import os
import glob
import re
import csv
import multiprocessing as mp
from tqdm import tqdm


NUM_WORKERS = 10

pix = 1 / 28
D_list = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0]
DX_list = [pix, 1.5 * pix, 2 * pix, 2.5 * pix, 3 * pix]
seeds = range(10, 20)
ds = 0.4
base_dir = os.path.join("results", "single_var")


def _worker(task):
    """
    Top-level picklable worker. Imports heavy deps lazily so that
    Windows spawn doesn't re-import them at module level in every child.
    """
    import numpy as np
    import torch
    import random

    from NeuralCortexSimple_single_var import (
        HexEncoder,
        load_image_tensor,
    )

    image_path, p_val, d_val, seed, DX, D_diff = task

    torch.set_num_threads(1)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    optotype = load_image_tensor(image_path, image_size=32, invert=True)

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

    return [[p_val, d_val, seed, DX, D_diff, t, e] for t, e in zip(times, theta_abs)]


def _build_tasks():
    tasks = []
    image_paths = sorted(glob.glob(os.path.join("images_c", "*.png")))
    for image_path in image_paths:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        m = re.match(r"myopia_(\d+)p(\d+)D", basename)
        p_val = float(m.group(1))
        d_val = float(m.group(2))
        for seed in seeds:
            for DX in DX_list:
                for D_diff in D_list:
                    tasks.append((image_path, p_val, d_val, seed, DX, D_diff))
    return tasks


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, "data.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "D", "seed", "DX", "D_diff", "time", "abs_theta"])

    tasks = _build_tasks()
    print(f"Dispatching {len(tasks)} tasks across {NUM_WORKERS} workers...")

    with mp.Pool(processes=NUM_WORKERS) as pool:
        for rows in tqdm(
            pool.imap_unordered(_worker, tasks, chunksize=1),
            total=len(tasks),
            desc="Tasks",
        ):
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

    print(f"\nDone. Results saved to '{base_dir}'")
