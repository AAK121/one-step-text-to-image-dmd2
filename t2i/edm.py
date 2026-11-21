"""EDM training loss, Heun sampler with classifier-free guidance, and helpers."""
import copy
import random

import numpy as np
import torch

SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0


def karras_sigmas(n, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO, device="cpu"):
    """n sigmas from sigma_max down to sigma_min (Karras et al. 2022, Eq. 5)."""
    i = torch.arange(n, device=device, dtype=torch.float64)
    s = (sigma_max ** (1 / rho) + i / max(n - 1, 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return s.float()


def edm_loss(net, x0, ids=None, text=None, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    """Denoising score matching with EDM weighting lambda(sigma)."""
    B = x0.shape[0]
    sigma = (torch.randn(B, device=x0.device) * P_std + P_mean).exp()
    w = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    xt = x0 + sigma.view(-1, 1, 1, 1) * torch.randn_like(x0)
    D = net(xt, sigma, ids=ids, text=text)
    return (w.view(-1, 1, 1, 1) * (D - x0) ** 2).mean()


def guided_denoise(net, x, sigma, text_c, text_u, scale):
    """CFG on the x0-prediction: D_u + s (D_c - D_u). Batched as one forward pass."""
    if scale == 1.0:
        return net(x, sigma, text=text_c)
    ctx = torch.cat([text_c[0], text_u[0]]), torch.cat([text_c[1], text_u[1]]), torch.cat([text_c[2], text_u[2]])
    d = net(torch.cat([x, x]), torch.cat([sigma, sigma]), text=ctx)
    dc, du = d.chunk(2)
    return du + scale * (dc - du)


@torch.no_grad()
def heun_sample(net, ids, null_ids, steps=18, cfg=2.0, z=None, S_churn=0.0):
    """Deterministic 2nd-order Heun sampler (EDM Algorithm 1) with CFG. Returns images in [-1, 1]."""
    device = ids.device
    B = ids.shape[0]
    res = net.img_res
    if z is None:
        z = torch.randn(B, getattr(net, "in_ch", 3), res, res, device=device)
    text_c = net.encode_text(ids)
    text_u = net.encode_text(null_ids.expand(B, -1))
    sig = torch.cat([karras_sigmas(steps, device=device), torch.zeros(1, device=device)])
    x = z * sig[0]
    for i in range(steps):
        s, s_next = sig[i], sig[i + 1]
        sv = s.expand(B)
        d = (x - guided_denoise(net, x, sv, text_c, text_u, cfg)) / s
        x_next = x + (s_next - s) * d
        if s_next > 0:
            d2 = (x_next - guided_denoise(net, x_next, s_next.expand(B), text_c, text_u, cfg)) / s_next
            x_next = x + (s_next - s) * 0.5 * (d + d2)
        x = x_next
    return x.clamp(-1, 1)


@torch.no_grad()
def one_step(gen, ids, z=None, text=None):
    """DMD one-step generator: G(z, c) = D(sigma_max * z, sigma_max, c)."""
    B = ids.shape[0] if ids is not None else text[0].shape[0]
    device = next(gen.parameters()).device
    if z is None:
        z = torch.randn(B, getattr(gen, "in_ch", 3), gen.img_res, gen.img_res, device=device)
    s = torch.full((B,), SIGMA_MAX, device=device)
    return gen(z * SIGMA_MAX, s, ids=ids, text=text).clamp(-1, 1)


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model, decay=None):
        d = self.decay if decay is None else decay
        for e, p in zip(self.shadow.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - d)
        for e, b in zip(self.shadow.buffers(), model.buffers()):
            e.copy_(b)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
