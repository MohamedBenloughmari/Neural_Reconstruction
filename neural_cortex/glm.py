from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GLMContext:
    t_xg: torch.Tensor
    t_yg: torch.Tensor
    t_px_x: torch.Tensor
    t_px_y: torch.Tensor
    t_neg_half_inv_sigma2: torch.Tensor
    t_inv_two_pi_sigma2: torch.Tensor
    t_lambda0: torch.Tensor
    t_log_ratio: torch.Tensor
    t_dt: torch.Tensor
    t_eps: torch.Tensor


def make_glm_context(ganglion_x, ganglion_y, half_n, dx, dy, ds,
                     lambda0, lambda1, dt, device):
    t_xg = torch.as_tensor(ganglion_x, dtype=torch.float32, device=device)
    t_yg = torch.as_tensor(ganglion_y, dtype=torch.float32, device=device)
    H = W = 2 * half_n
    t_px_x = (torch.arange(H, device=device, dtype=torch.float32) + 0.5 - half_n) * dx
    t_px_y = (torch.arange(W, device=device, dtype=torch.float32) + 0.5 - half_n) * dy
    sigma_s = 0.5 * ds
    sigma_e = 0.203 * ds
    sigma2 = sigma_s ** 2 + sigma_e ** 2
    t_sigma2 = torch.tensor(sigma2, dtype=torch.float32, device=device)
    t_inv_sigma2 = 1.0 / t_sigma2
    t_neg_half_inv_sigma2 = -0.5 * t_inv_sigma2
    t_log_ratio = torch.tensor(np.log(lambda1 / lambda0), dtype=torch.float32, device=device)
    t_two_pi_sigma2 = 2.0 * float(np.pi) * t_sigma2
    t_inv_two_pi_sigma2 = 1.0 / t_two_pi_sigma2
    t_lambda0 = torch.tensor(lambda0, dtype=torch.float32, device=device)
    t_dt = torch.tensor(dt, dtype=torch.float32, device=device)
    t_eps = torch.tensor(1e-12, dtype=torch.float32, device=device)

    return GLMContext(
        t_xg=t_xg, t_yg=t_yg,
        t_px_x=t_px_x, t_px_y=t_px_y,
        t_neg_half_inv_sigma2=t_neg_half_inv_sigma2,
        t_inv_two_pi_sigma2=t_inv_two_pi_sigma2,
        t_lambda0=t_lambda0, t_log_ratio=t_log_ratio,
        t_dt=t_dt, t_eps=t_eps,
    )


def glm_rates_np(S, cx, cy, ganglion_x, ganglion_y, half_n, dx, dy, ds,
                 lambda0, lambda1):
    H, W = S.shape
    sigma_s = 0.5 * ds
    sigma_e = 0.203 * ds
    sigma2 = sigma_s ** 2 + sigma_e ** 2
    px_x = (np.arange(H) + 0.5 - half_n) * dx
    px_y = (np.arange(W) + 0.5 - half_n) * dy
    xg, yg = ganglion_x, ganglion_y
    diff_x = (cx + px_x[None, :]) - xg[:, None]
    diff_y = (cy + px_y[None, :]) - yg[:, None]
    gx_w = np.exp(-0.5 * diff_x ** 2 / sigma2)
    gy_w = np.exp(-0.5 * diff_y ** 2 / sigma2)
    tmp = gx_w @ S
    c_raw = np.sum(tmp * gy_w, axis=1) / (2.0 * np.pi * sigma2)
    g_norm = max(c_raw.max(), 1e-9)
    c = np.clip(c_raw / g_norm, 0.0, 1.0)
    lam_on = lambda0 * np.exp(np.log(lambda1 / lambda0) * c)
    lam_off = lambda0 * np.exp(np.log(lambda1 / lambda0) * (1.0 - c))
    return lam_on, lam_off


def precompute_spatial_kernels(cx, cy, ctx):
    neg_half_div_s2 = ctx.t_neg_half_inv_sigma2
    diff_x = (cx[:, None, None] + ctx.t_px_x[None, None, :]) - ctx.t_xg[None, :, None]
    diff_y = (cy[:, None, None] + ctx.t_px_y[None, None, :]) - ctx.t_yg[None, :, None]
    gx_w = torch.exp(neg_half_div_s2 * diff_x * diff_x)
    gy_w = torch.exp(neg_half_div_s2 * diff_y * diff_y)
    return gx_w, gy_w


def glm_rates_torch_batch(S, cx, cy, ctx, gx_w=None, gy_w=None):
    if gx_w is None or gy_w is None:
        gx_w, gy_w = precompute_spatial_kernels(cx, cy, ctx)
    tmp = torch.matmul(gx_w, S)
    c_raw = (tmp * gy_w).sum(dim=2) * ctx.t_inv_two_pi_sigma2
    g_norm = c_raw.detach().amax(dim=1, keepdim=True).clamp(min=1e-9)
    c = (c_raw / g_norm).clamp(0.0, 1.0)
    lam_on = ctx.t_lambda0 * torch.exp(ctx.t_log_ratio * c)
    lam_off = ctx.t_lambda0 * torch.exp(ctx.t_log_ratio * (1.0 - c))
    return lam_on, lam_off
