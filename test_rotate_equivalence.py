"""
Test: scipy.ndimage.rotate vs torch F.affine_grid + F.grid_sample equivalence.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


def rotate_scipy(img_tensor, theta_deg):
    """Current method: scipy.ndimage.rotate (order=1 = bilinear)."""
    arr = img_tensor.numpy()
    rotated = rotate(arr, angle=theta_deg, reshape=False, order=1,
                     mode="constant", cval=0.0)
    return torch.from_numpy(rotated).float()


def rotate_torch(img_tensor, theta_deg):
    """Proposed method: F.affine_grid + F.grid_sample."""
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    _, _, H, W = img_tensor.shape
    theta_rad = np.deg2rad(theta_deg)
    device = img_tensor.device

    cos = np.cos(theta_rad)
    sin = np.sin(theta_rad)
    rot_mat = torch.tensor([[cos, -sin, 0.0],
                            [sin,  cos, 0.0]],
                           device=device, dtype=torch.float32).unsqueeze(0)

    grid = F.affine_grid(rot_mat, img_tensor.size(), align_corners=False)
    out = F.grid_sample(img_tensor, grid, mode='bilinear',
                        padding_mode='zeros', align_corners=False)
    return out.squeeze(0).squeeze(0)


def test_equivalence():
    torch.manual_seed(42)
    np.random.seed(42)

    # Random test image
    H, W = 32, 32
    img = torch.rand(H, W, dtype=torch.float32)

    angles = [0, 15, 30, 45, 60, 90, 120, 180, 270, 355, -30, -45]

    all_close = True
    diffs = []

    for angle in angles:
        scipy_result = rotate_scipy(img.clone(), angle)
        torch_result = rotate_torch(img.clone(), angle)

        diff = (scipy_result - torch_result).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        diffs.append((angle, max_diff, mean_diff))

        if max_diff > 1e-2:
            all_close = False
            print(f"angle={angle:>5} deg: max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  ** MISMATCH **")
        else:
            print(f"angle={angle:>5} deg: max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")

    if all_close:
        print("\nPASS: all angles match within 1e-2 tolerance.")
    else:
        print("\nFAIL: some angles exceed 1e-2 tolerance (boundary interpolation differences).")

    # Visual comparison for a few angles
    plot_angles = [0, 30, 60, 90]
    fig, axes = plt.subplots(2, len(plot_angles), figsize=(12, 6))
    for i, angle in enumerate(plot_angles):
        s = rotate_scipy(img.clone(), angle)
        t = rotate_torch(img.clone(), angle)
        axes[0, i].imshow(s.cpu(), cmap='gray')
        axes[0, i].set_title(f'scipy {angle}°')
        axes[1, i].imshow(t.cpu(), cmap='gray')
        axes[1, i].set_title(f'torch  {angle}°')
    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('rotate_comparison.png', dpi=120)
    plt.close()
    print("Saved visual comparison to rotate_comparison.png")

    return all_close


if __name__ == "__main__":
    test_equivalence()
