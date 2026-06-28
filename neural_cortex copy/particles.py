import torch

def propagate_particles(particles, spikes_on_t, spikes_off_t, get_rates,
                        diff_sigma, dt, eps, device):
    n_p = particles.shape[0]
    noise = torch.randn(n_p, 2, device=device) * diff_sigma
    new_p = particles + noise
    with torch.no_grad():
        lam_on, lam_off = get_rates(new_p[:, 0], new_p[:, 1])
        log_w = (spikes_on_t * torch.log(lam_on * dt + eps) - lam_on * dt
                 + spikes_off_t * torch.log(lam_off * dt + eps) - lam_off * dt
                 ).sum(dim=1)
        log_w = log_w - log_w.max()
        w = torch.exp(log_w)
        w = w / w.sum()
    return new_p, w


def resample_particles(particles, weights, device):
    n_p = weights.shape[0]
    csum = torch.cumsum(weights, dim=0)
    u = torch.rand(n_p, device=device)
    idx = torch.searchsorted(csum, u)
    idx = idx.clamp(0, n_p - 1)
    uniform_w = torch.ones(n_p, device=device) / n_p
    return particles[idx], uniform_w


def sample_positions(particles, weights, n_samples, device):
    n_p = weights.shape[0]
    csum = torch.cumsum(weights, dim=0)
    u = torch.rand(n_samples, device=device)
    idx = torch.searchsorted(csum, u)
    idx = idx.clamp(0, n_p - 1)
    return particles[idx]
