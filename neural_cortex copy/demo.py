import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from neural_cortex import NeuralCortex, load_image_tensor


if __name__ == "__main__":
    image_path = "images_c/myopia_0p0D.png"

    optotype = load_image_tensor(image_path, image_size=32, invert=True, deblur=False)
    print(f"Loaded image from: {image_path}")

    seed = 55
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    mean_offset = float(optotype.mean())
    optotype_centered = optotype - mean_offset

    sim = NeuralCortex(dx=0.2, dy=0.2, dt=0.01, ds=0.3, D_diff=5.0, img_size=32)
    sim.fit(optotype_centered, blur_sigma=0.0)
    sim.simulate_random_walk(T=0.750)
    sim.compute_activations(grid_range=10, grid_resolution=30)
    sim.decode(
        mean_offset=mean_offset,
        n_particles=100, n_samples=100,
        beta=0.001, gamma=10.0,
        adam_iter=20, lr=1e-2,
        anchor_weight=1.0,
        hessian_tau=10.0,
        hessian_every=1,
        verbose=True,
    )

    anim = sim.animate(interval=100, save_path="C_de_Landolt.gif")
    plt.show()
