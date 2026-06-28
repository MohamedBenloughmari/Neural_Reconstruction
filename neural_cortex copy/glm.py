from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GLMContext:
    xg: torch.Tensor
    yg: torch.Tensor
    px_x: torch.Tensor
    px_y: torch.Tensor
    lambda0: torch.Tensor
    lambda1:torch.Tensor
    log_ratio: torch.Tensor
    dt: torch.Tensor
    eps: torch.Tensor
    dx : torch.Tensor
    dy :torch.Tensor
    ds :torch.Tensor


def make_glm_context(ganglion_x, ganglion_y,optotype,half_n, dx, dy, ds,
                     lambda0, lambda1, dt, device):
    xg = torch.as_tensor(ganglion_x, dtype=torch.float32, device=device)
    yg = torch.as_tensor(ganglion_y, dtype=torch.float32, device=device)
    H,W = optotype.shape
    px_x = (torch.arange(H, device=device, dtype=torch.float32) + 0.5 - half_n) * dx
    px_y = (torch.arange(W, device=device, dtype=torch.float32) + 0.5 - half_n) * dy
    sigma_s = 0.5 * ds
    sigma_e = 0.203 * ds
    sigma2 = sigma_s ** 2 + sigma_e ** 2
    sigma2 = torch.tensor(sigma2, dtype=torch.float32, device=device)
    #t_inv_sigma2 = 1.0 / t_sigma2
    #t_neg_half_inv_sigma2 = -0.5 * t_inv_sigma2
    log_ratio = torch.tensor(np.log(lambda1 / lambda0), dtype=torch.float32, device=device)
    #t_two_pi_sigma2 = 2.0 * float(np.pi) * t_sigma2
    #t_inv_two_pi_sigma2 = 1.0 / t_two_pi_sigma2
    lambda0 = torch.tensor(lambda0, dtype=torch.float32, device=device)
    dt = torch.tensor(dt, dtype=torch.float32, device=device)
    eps = torch.tensor(1e-12, dtype=torch.float32, device=device)
    dx=torch.tensor(dx,dtype=torch.float32,device=device)
    dy=torch.tensor(dy,dtype=torch.float32,device=device)
    ds=torch.tensor(ds,dtype=torch.float32,device=device)
    
    return GLMContext(
        xg=xg, yg=yg,
        px_x=px_x, px_y=px_y,
        lambda0=lambda0,lambda1=lambda1,
        log_ratio=log_ratio,dt=dt,eps=eps,
        dx=dx,dy=dy,ds=ds
        )


def glm_rates(S, cx, cy, glm_ctx):
    H, W = S.shape
    sigma_s = 0.5 * ds
    sigma_e = 0.203 * ds
    sigma2 = sigma_s ** 2 + sigma_e ** 2
    px_x = (np.arange(H) + 0.5 - half_n) * dx
    px_y = (np.arange(W) + 0.5 - half_n) * dy
    xg, yg = ganglion_x, ganglion_y
    diff_x = -cx + px_x[:,None] - xg[None,:]
    diff_y = -cy + px_y[:,None] - yg[None,:]
    gx_w = np.exp(-0.5 * diff_x ** 2 / sigma2)
    gy_w = np.exp(-0.5 * diff_y ** 2 / sigma2)
    tmp = gx_w @ S
    c_raw = np.sum(tmp * gy_w, axis=1) / (2.0 * np.pi * sigma2)
    g_norm = max(c_raw.max(), 1e-9)
    c = np.clip(c_raw / g_norm, 0.0, 1.0)
    lam_on = lambda0 * np.exp(np.log(lambda1 / lambda0) * c)
    lam_off = lambda0 * np.exp(np.log(lambda1 / lambda0) * (1.0 - c))
    return lam_on, lam_off





































