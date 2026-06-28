import numpy as np
import torch
from torch.func import grad

from .gaussian import img_to_encoder_torch
from .glm import precompute_spatial_kernels, glm_rates_torch_batch


def Er_per_sample(glm_ctx, S, cx, cy, r_on_t, r_off_t, gx_w, gy_w):
    lam_on, lam_off = glm_rates_torch_batch(S, cx, cy, glm_ctx, gx_w, gy_w)
    dt = glm_ctx.t_dt
    return ((lam_on * dt - r_on_t * torch.log(lam_on * dt + glm_ctx.t_eps)).sum(dim=1)
            + (lam_off * dt - r_off_t * torch.log(lam_off * dt + glm_ctx.t_eps)).sum(dim=1))


def Er(glm_ctx, S, samples_t, r_on_t, r_off_t, device, gx_w=None, gy_w=None):
    cx = torch.as_tensor(samples_t[:, 0], dtype=torch.float32, device=device)
    cy = torch.as_tensor(samples_t[:, 1], dtype=torch.float32, device=device)
    per_sample = Er_per_sample(glm_ctx, S, cx, cy, r_on_t, r_off_t, gx_w, gy_w)
    return per_sample.sum() / samples_t.shape[0]


def Ep(A_param, A_anchor, beta):
    log_p_anchor = -beta * torch.sum(torch.abs(A_anchor))
    grad_log_p = -beta * torch.sign(A_anchor)
    neg_Ep_lin = log_p_anchor - (grad_log_p * (A_param - A_anchor)).sum()
    return -neg_Ep_lin


def adam_update_A(glm_ctx, A_param, Hessian, optimizer,
                  samples_t, r_on_t, r_off_t,
                  A_anchor, beta, gamma, n_iter,
                  img_size, mean_offset, device, anchor_weight=1.0):
    cx = torch.as_tensor(samples_t[:, 0], dtype=torch.float32, device=device)
    cy = torch.as_tensor(samples_t[:, 1], dtype=torch.float32, device=device)
    gx_w, gy_w = precompute_spatial_kernels(cx, cy, glm_ctx)
    last_loss = 0.0
    for _ in range(n_iter):
        optimizer.zero_grad()
        S = img_to_encoder_torch(A_param.reshape(img_size, img_size) + mean_offset)
        Er_val = Er(glm_ctx, S, samples_t, r_on_t, r_off_t, device, gx_w, gy_w)

        diff = A_param - A_anchor
        Eg = 0.5 * anchor_weight * (diff @ (Hessian @ diff))
        Ep_val = Ep(A_param, A_anchor, beta)

        range_pen = gamma * (
            torch.clamp(S - 1.0, min=0.0)
            + torch.clamp(-S, min=0.0)
        ).sum()

        loss = Er_val + Eg + Ep_val + range_pen
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    return last_loss


def compute_gauss_newton_hessian(glm_ctx, A, samples_t, r_on_t, r_off_t,
                                  gx_w, gy_w, img_size, mean_offset, device):
    n_samples = samples_t.shape[0]
    N_sp = A.numel()

    def Er_single(A_vec, cx1, cy1, r_on1, r_off1, gx1, gy1):
        S = img_to_encoder_torch(A_vec.reshape(img_size, img_size) + mean_offset)
        lam_on, lam_off = glm_rates_torch_batch(S, cx1, cy1, glm_ctx, gx1, gy1)
        dt = glm_ctx.t_dt
        loss = ((lam_on * dt - r_on1 * torch.log(lam_on * dt + glm_ctx.t_eps)).sum()
                + (lam_off * dt - r_off1 * torch.log(lam_off * dt + glm_ctx.t_eps)).sum())
        return loss / n_samples

    J = torch.zeros(n_samples, N_sp, device=device, dtype=torch.float32)
    for i in range(n_samples):
        cx_i = samples_t[i:i+1, 0:1]
        cy_i = samples_t[i:i+1, 1:2]
        gx_i = gx_w[i:i+1]
        gy_i = gy_w[i:i+1]
        r_on_i = r_on_t.unsqueeze(0)
        r_off_i = r_off_t.unsqueeze(0)

        def bound_fn(a):
            return Er_single(a, cx_i, cy_i, r_on_i, r_off_i, gx_i, gy_i)

        J[i] = grad(bound_fn)(A).detach()

    H_gn = J.T @ J
    return 0.5 * (H_gn + H_gn.T)


def update_hessian(glm_ctx, H_prev, A_anchor, samples_t,
                   r_on_t, r_off_t, tau, dt,
                   img_size, mean_offset, device):
    cx = torch.as_tensor(samples_t[:, 0], dtype=torch.float32, device=device)
    cy = torch.as_tensor(samples_t[:, 1], dtype=torch.float32, device=device)
    gx_w, gy_w = precompute_spatial_kernels(cx, cy, glm_ctx)

    H_new = compute_gauss_newton_hessian(
        glm_ctx, A_anchor, samples_t, r_on_t, r_off_t,
        gx_w, gy_w, img_size, mean_offset, device,
    )
    decay = float(np.exp(-dt / tau))
    return decay * H_prev + H_new
