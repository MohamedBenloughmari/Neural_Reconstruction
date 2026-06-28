import torch
import torch.nn.functional as F


def gaussian_kernel_2d(sigma, kernel_size=None):
    if kernel_size is None:
        kernel_size = int(4 * sigma + 1) | 1
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    x, y = torch.meshgrid(ax, ax, indexing='ij')
    k = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k[None, None, :, :]


def gaussian_blur_torch(img_2d, sigma):
    if sigma <= 0:
        return img_2d
    kernel = gaussian_kernel_2d(sigma)
    kernel = kernel.to(img_2d.device)
    padding = kernel.shape[-1] // 2
    img_4d = img_2d[None, None, :, :]
    return F.conv2d(img_4d, kernel, padding=padding)[0, 0]

