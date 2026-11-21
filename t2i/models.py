"""From-scratch networks: a small transformer text encoder and a text-conditioned
UNet denoiser with EDM preconditioning (Karras et al. 2022).

The same `T2IDenoiser` class is used for
  * the teacher diffusion model (real score, mu_base in the DMD paper),
  * the fake-score model (mu_fake), and
  * the one-step generator G(z, c) = D(sigma_max * z, sigma_max, c).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PAD


# --------------------------------------------------------------------------- text
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, max_len=40, dim=256, depth=4, heads=4, dropout=0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, ids):
        """ids: (B, L) long -> tokens (B, L, D), pooled (B, D), key-padding mask (B, L; True = pad)."""
        mask = ids == PAD
        h = self.tok(ids) + self.pos[:, : ids.shape[1]]
        h = self.blocks(h, src_key_padding_mask=mask)
        h = self.norm(h)
        keep = (~mask).unsqueeze(-1).to(h.dtype)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return h, pooled, mask


# --------------------------------------------------------------------------- unet
def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def norm(ch):
    return nn.GroupNorm(32 if ch % 32 == 0 else 8, ch)


def zero(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout=0.0):
        super().__init__()
        self.n1 = norm(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)  # adaptive group-norm scale/shift
        self.n2 = norm(cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = zero(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return (self.skip(x) + h) / math.sqrt(2)


class AttnBlock(nn.Module):
    """Self-attention over pixels followed by cross-attention to text tokens."""

    def __init__(self, ch, ctx_dim, head_dim=64):
        super().__init__()
        self.heads = max(1, ch // head_dim)
        self.n1 = norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.o1 = zero(nn.Conv2d(ch, ch, 1))
        self.n2 = norm(ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.kv = nn.Linear(ctx_dim, 2 * ch)
        self.o2 = zero(nn.Conv2d(ch, ch, 1))

    def _split(self, t):  # (B, C, N) -> (B, H, N, C/H)
        B, C, N = t.shape
        return t.reshape(B, self.heads, C // self.heads, N).transpose(-1, -2)

    def forward(self, x, ctx, ctx_mask):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, 3 * C, H * W).chunk(3, dim=1)
        q, k, v = (self._split(t).contiguous() for t in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o1(h.transpose(-1, -2).reshape(B, C, H, W))

        q = self._split(self.q(self.n2(x)).reshape(B, C, H * W)).contiguous()
        k, v = self.kv(ctx).transpose(1, 2).chunk(2, dim=1)  # (B, C, L) each
        k, v = (self._split(t).contiguous() for t in (k, v))
        attn_mask = (~ctx_mask)[:, None, None, :]  # True = attend
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return x + self.o2(h.transpose(-1, -2).reshape(B, C, H, W))


class UNet(nn.Module):
    def __init__(self, in_ch=3, ch=128, mults=(1, 2, 2, 3), num_res=2, attn_res=(16, 8),
                 ctx_dim=256, img_res=64, dropout=0.1):
        super().__init__()
        emb_dim = ch * 4
        self.ch = ch
        self.t_mlp = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.txt_mlp = nn.Sequential(nn.Linear(ctx_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)

        self.down = nn.ModuleList()
        chans = [ch]
        cur, res = ch, img_res
        for i, m in enumerate(mults):
            for _ in range(num_res):
                blk = nn.ModuleList([ResBlock(cur, ch * m, emb_dim, dropout)])
                cur = ch * m
                if res in attn_res:
                    blk.append(AttnBlock(cur, ctx_dim))
                self.down.append(blk)
                chans.append(cur)
            if i != len(mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(cur, cur, 3, stride=2, padding=1)]))
                chans.append(cur)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(cur, cur, emb_dim, dropout), AttnBlock(cur, ctx_dim),
                                  ResBlock(cur, cur, emb_dim, dropout)])

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mults))):
            for j in range(num_res + 1):
                blk = nn.ModuleList([ResBlock(cur + chans.pop(), ch * m, emb_dim, dropout)])
                cur = ch * m
                if res in attn_res:
                    blk.append(AttnBlock(cur, ctx_dim))
                if i != 0 and j == num_res:
                    blk.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    blk.append(nn.Conv2d(cur, cur, 3, padding=1))
                    res *= 2
                self.up.append(blk)
        self.out = nn.Sequential(norm(cur), nn.SiLU(), zero(nn.Conv2d(cur, in_ch, 3, padding=1)))

    def _run(self, blk, h, emb, ctx, mask):
        for layer in blk:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            elif isinstance(layer, AttnBlock):
                h = layer(h, ctx, mask)
            else:
                h = layer(h)
        return h

    def forward(self, x, c_noise, ctx, pooled, mask):
        emb = self.t_mlp(timestep_embedding(c_noise * 1000.0, self.ch)) + self.txt_mlp(pooled)
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            h = self._run(blk, h, emb, ctx, mask)
            hs.append(h)
        h = self._run(self.mid, h, emb, ctx, mask)
        for blk in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            h = self._run(blk, h, emb, ctx, mask)
        return self.out(h)


# --------------------------------------------------------------------------- EDM wrapper
class T2IDenoiser(nn.Module):
    """D(x; sigma, text) with EDM preconditioning. Output is the predicted clean image x0."""

    def __init__(self, vocab_size, max_len=40, text_dim=256, text_depth=4, ch=128,
                 mults=(1, 2, 2, 3), num_res=2, attn_res=(16, 8), img_res=64, dropout=0.1,
                 sigma_data=0.5, in_ch=3):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, max_len=max_len, text_dim=text_dim,
                           text_depth=text_depth, ch=ch, mults=list(mults), num_res=num_res,
                           attn_res=list(attn_res), img_res=img_res, dropout=dropout,
                           sigma_data=sigma_data, in_ch=in_ch)
        self.sigma_data = sigma_data
        self.img_res = img_res
        self.in_ch = in_ch
        self.text = TextEncoder(vocab_size, max_len, text_dim, text_depth)
        self.unet = UNet(in_ch, ch, tuple(mults), num_res, tuple(attn_res), text_dim, img_res, dropout)

    def encode_text(self, ids):
        return self.text(ids)

    def predict_eps(self, x, t, ids=None, text=None):
        """Vanilla DDPM epsilon head. `t` is an integer timestep in [0, T), not an EDM sigma.

        The UNet embeds `c_noise * 1000`, so passing `t / 1000` feeds it the discrete timestep.
        """
        if text is None:
            text = self.encode_text(ids)
        ctx, pooled, mask = text
        return self.unet(x, t.float().reshape(-1) / 1000.0, ctx, pooled, mask)

    def forward(self, x, sigma, ids=None, text=None):
        """x: (B,3,H,W) noisy image, sigma: (B,) noise std, ids: token ids or precomputed text tuple."""
        if text is None:
            text = self.encode_text(ids)
        ctx, pooled, mask = text
        sigma = sigma.float().reshape(-1, 1, 1, 1)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1 / (sd ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log().flatten() / 4
        F_x = self.unet((c_in * x).to(x.dtype), c_noise, ctx, pooled, mask)
        return c_skip * x + c_out * F_x.float()


def build_model(cfg: dict) -> T2IDenoiser:
    return T2IDenoiser(**cfg)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
