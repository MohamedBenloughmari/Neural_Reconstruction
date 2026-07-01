"""
Compare scipy.rotate against torchvision.rotate and manual grid_sample.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate


def scipy_rotate(arr, theta_deg):
    return rotate(arr, angle=theta_deg, reshape=False, order=1,
                  mode='constant', cval=0.0)


def torchvision_rotate(img_torch, theta_deg):
    """Use torchvision's rotate which should match scipy."""
    from torchvision.transforms.functional import rotate as tv_rotate
    from torchvision.transforms.functional import InterpolationMode
    if img_torch.dim() == 2:
        img_torch = img_torch.unsqueeze(0)
    # torchvision rotate: positive angle = CCW
    out = tv_rotate(img_torch, angle=theta_deg, interpolation=InterpolationMode.BILINEAR,
                    expand=False, fill=0.0)
    return out.squeeze(0)


def diff_summary(name_a, a, name_b, b, angle):
    d = (a - b).abs()
    print(f"  {name_a} vs {name_b}: angle={angle:>6} deg  max={d.max().item():.4e}  mean={d.mean().item():.4e}")


if __name__ == '__main__':
    np.random.seed(42)
    H, W = 32, 32

    # Positional test pattern: x-gradient
    xx = np.arange(W, dtype=np.float32) / (W - 1)
    x_img = torch.from_numpy(np.tile(xx, (H, 1)))

    print("=== x-gradient image ===")
    for angle in [0, 5, 10, 15, 30, 45, 90, -30, -45, -90]:
        s = torch.from_numpy(scipy_rotate(x_img.numpy(), angle))
        tv = torchvision_rotate(x_img, angle)
        diff_summary("scipy", s, "torchvision", tv, angle)

    print("\n=== random image ===")
    r_img = torch.rand(H, W).float()
    for angle in [0, 5, 10, 15, 30, 45, 90, -30, -45, -90]:
        s = torch.from_numpy(scipy_rotate(r_img.numpy(), angle))
        tv = torchvision_rotate(r_img, angle)
        diff_summary("scipy", s, "torchvision", tv, angle)
