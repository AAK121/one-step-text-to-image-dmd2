"""Small image autoencoder, trained from scratch on our portraits.

The diffusion teacher runs on the 32x32 latent. The decoder is part of the
submitted generator, so it stays small enough that decoder plus one-step
denoiser can fit under 50M.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_ch, out_ch, k=3, stride=1):
    return nn.Conv2d(in_ch, out_ch, k, stride, padding=k // 2)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        g = 32 if in_ch % 32 == 0 else 8
        self.norm = nn.GroupNorm(g, in_ch)
        self.conv1 = conv(in_ch, out_ch)
        self.conv2 = conv(out_ch, out_ch)
        self.skip = conv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm(x)))
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class Encoder(nn.Module):
    def __init__(self, z_ch=16, n_down=3):
        super().__init__()
        self.stem = conv(3, 64)
        stages = [
            nn.ModuleList([Block(64, 64), conv(64, 128, stride=2)]),
            nn.ModuleList([Block(128, 128), conv(128, 192, stride=2)]),
            nn.ModuleList([Block(192, 192), conv(192, 192, stride=2)]),
        ]
        self.down = nn.ModuleList(stages[:n_down])
        # One downsample ends at 128 channels. Two and three already end at 192.
        last_ch = (128, 192, 192)[n_down - 1]
        self.proj = conv(last_ch, 192, 1) if last_ch != 192 else nn.Identity()
        self.mid = Block(192, 192)
        self.head = conv(192, 2 * z_ch, 1)

    def forward(self, x):
        h = self.stem(x)
        for block, down in self.down:
            h = down(block(h))
        h = self.mid(self.proj(h))
        mean, logvar = self.head(h).chunk(2, dim=1)
        logvar = logvar.clamp(-30, 20)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, z_ch=16, n_down=3):
        super().__init__()
        self.stem = conv(z_ch, 192, 1)
        self.mid = Block(192, 192)
        # 32 -> 256 is three ups, 64 -> 256 is two, 128 -> 256 is one.
        ups = [
            nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), Block(192, 192)]),
            nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), Block(192, 128)]),
            nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), Block(128, 64)]),
        ]
        if n_down == 1:
            ups = [nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), Block(192, 64)])]
        elif n_down == 2:
            ups = ups[1:]
        self.up = nn.ModuleList(ups)
        self.head = nn.Sequential(nn.GroupNorm(32, 64), nn.SiLU(), conv(64, 3))

    def forward(self, z):
        h = self.mid(self.stem(z))
        for up, block in self.up:
            h = block(up(h))
        return self.head(h).tanh()


class Autoencoder(nn.Module):
    def __init__(self, z_ch=16, latent_res=32, n_down=None):
        super().__init__()
        # latent_res 32/64/128 means 3/2/1 downsamples of a 256 image.
        # A smaller image passes n_down explicitly (64 -> 32 is one downsample).
        if n_down is None:
            if latent_res not in (32, 64, 128):
                raise ValueError(f"latent_res must be 32, 64, or 128, got {latent_res}")
            n_down = {32: 3, 64: 2, 128: 1}[latent_res]
        elif n_down not in (1, 2, 3):
            raise ValueError(f"n_down must be 1, 2, or 3, got {n_down}")
        self.z_ch = z_ch
        self.latent_res = latent_res
        self.n_down = n_down
        self.encoder = Encoder(z_ch, n_down)
        self.decoder = Decoder(z_ch, n_down)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return self.decoder(z), mean, logvar

    @torch.no_grad()
    def encode_mean(self, x):
        mean, _ = self.encoder(x)
        return mean

    @torch.no_grad()
    def decode(self, z):
        return self.decoder(z)


def ae_loss(recon, x, mean, logvar, kl_weight=1e-6):
    rec = F.l1_loss(recon, x)
    # Match horizontal and vertical differences so the decoder keeps edges.
    rec = rec + 0.5 * (
        F.l1_loss(recon[..., :, 1:] - recon[..., :, :-1], x[..., :, 1:] - x[..., :, :-1])
        + F.l1_loss(recon[..., 1:, :] - recon[..., :-1, :], x[..., 1:, :] - x[..., :-1, :])
    )
    kl = 0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar).mean()
    return rec + kl_weight * kl, rec.detach(), kl.detach()
