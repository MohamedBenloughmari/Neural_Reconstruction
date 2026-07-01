"""
Debug: find the correct affine_grid parameters to match scipy.ndimage.rotate.
Tests each axis convention and sign.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate


def scipy_rotate(arr, theta_deg):
    return rotate(arr, angle=theta_deg, reshape=False, order=1,
                  mode='constant', cval=0.0)


def torch_rotate_affine(img_torch, theta_deg, cos_sign, sin_sign,
                         align_corners):
    """affine_grid variant: 2x3 matrix = [[c, s, 0], [s2, c2, 0]]"""
    if img_torch.dim() == 2:
        img_torch = img_torch.unsqueeze(0).unsqueeze(0)
    _, _, H, W = img_torch.shape
    t = np.deg2rad(theta_deg)
    dev = img_torch.device

    # The two rows of the 2x3 matrix: [a, b, tx], [c, d, ty]
    # In (x, y) ordering:
    # x_src = a*x_out + b*y_out + tx
    # y_src = c*x_out + d*y_out + ty
    a = cos_sign * np.cos(t)
    b = sin_sign * np.sin(t)   # varies
    c = -sin_sign * np.sin(t)  # varies
    d = cos_sign * np.cos(t)

    rot = torch.tensor([[a, b, 0.0],
                         [c, d, 0.0]], device=dev, dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(rot, img_torch.size(), align_corners=align_corners)
    out = F.grid_sample(img_torch, grid, mode='bilinear',
                        padding_mode='zeros', align_corners=align_corners)
    return out.squeeze(0).squeeze(0)


def torch_rotate_gridsample(img_torch, theta_deg, align_corners):
    """Manual grid: rotate around origin (0,0) in normalized coords."""
    if img_torch.dim() == 2:
        img_torch = img_torch.unsqueeze(0).unsqueeze(0)
    _, _, H, W = img_torch.shape
    t_rad = np.deg2rad(theta_deg)
    dev = img_torch.device

    cos = np.cos(t_rad)
    sin = np.sin(t_rad)

    # Build a regular grid
    xx = torch.linspace(-1, 1, W, device=dev)
    yy = torch.linspace(-1, 1, H, device=dev)
    gy, gx = torch.meshgrid(yy, xx, indexing='ij')

    # Rotate the grid coordinates by -theta (inverse transform: we sample
    # from source where it was before the forward rotation)
    # Forward rotation of image contents by +theta means:
    # point at (x_out, y_out) came from source point R_{-theta} * (x_out, y_out)
    x_src = cos * gx + sin * gy    # R_{-theta} in (x,y)
    y_src = -sin * gx + cos * gy

    grid = torch.stack([x_src, y_src], dim=-1).unsqueeze(0)  # (1,H,W,2)
    out = F.grid_sample(img_torch, grid, mode='bilinear',
                        padding_mode='zeros', align_corners=align_corners)
    return out.squeeze(0).squeeze(0)


if __name__ == '__main__':
    np.random.seed(42)
    H, W = 32, 32

    # Create positional test: gradient in x direction
    xx = np.arange(W, dtype=np.float32) / (W - 1)
    xx_tile = np.tile(xx, (H, 1))
    img = torch.from_numpy(xx_tile)

    test_angles = [15, 30, 45, -15, -30, -45]

    for angle in test_angles:
        scipy = torch.from_numpy(scipy_rotate(img.numpy(), angle))

        print(f"\n=== angle = {angle} deg ===")
        for ac in [False, True]:
            for cos_s, sin_s in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                tor = torch_rotate_affine(img, angle, cos_s, sin_s, ac)
                md = (scipy - tor).abs().max().item()
                tag = ''
                if md < 0.1:
                    tag = ' <<<< MATCH'
                print(f"  align={ac} cos={cos_s:+d} sin={sin_s:+d}  max_diff={md:.4f}{tag}")
