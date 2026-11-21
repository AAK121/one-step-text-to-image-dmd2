"""Stage C: Distribution Matching Distillation (Yin et al., CVPR 2024, Algorithm 1).

Networks (all share the teacher's architecture and are initialised from its EMA weights):
  real  mu_base : frozen teacher, used with classifier-free guidance        (real score)
  fake  mu_fake : continually trained with the denoising loss on G's output  (fake score)
  G             : one-step generator G(z, c) = D(sigma_max z, sigma_max, c)

Generator objective:  D_KL  (distribution matching, Eq. 7)  +  lambda_reg * L_reg  (Eq. 9)
"""
import argparse
import copy
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from make_pairs import load_teacher
from t2i.edm import SIGMA_MAX, EMA, edm_loss, guided_denoise, karras_sigmas, seed_all
from t2i.models import count_params
from train_teacher import load_preview_prompts


def disable_dropout(m):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    return m


def gen_forward(G, z, ids):
    s = torch.full((z.shape[0],), SIGMA_MAX, device=z.device)
    return G(z * SIGMA_MAX, s, ids=ids)


class RegressionLoss(nn.Module):
    """Distance for L_reg. The paper uses LPIPS; LPIPS needs an ImageNet-pretrained VGG, so the
    default here is the fully from-scratch pseudo-Huber metric (Song & Dhariwal 2023)."""

    def __init__(self, kind="phuber"):
        super().__init__()
        self.kind = kind
        if kind == "lpips":
            import lpips  # optional dependency
            self.lp = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

    def forward(self, a, b):
        if self.kind == "l2":
            return F.mse_loss(a, b)
        if self.kind == "phuber":
            c = 0.00054 * math.sqrt(a[0].numel())
            return (((a - b) ** 2).flatten(1).sum(1) + c ** 2).sqrt().sub(c).mean() / math.sqrt(a[0].numel())
        up = lambda x: F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return self.lp(up(a.float()), up(b.float())).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher/last.pt")
    ap.add_argument("--pairs", default="runs/pairs.pt")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/dmd")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32, help="batch for the distribution-matching stream")
    ap.add_argument("--reg-batch", type=int, default=32, help="batch for the paired regression stream")
    ap.add_argument("--lr-g", type=float, default=2e-5)
    ap.add_argument("--lr-fake", type=float, default=2e-5)
    ap.add_argument("--cfg", type=float, default=2.0, help="guidance scale of the real score")
    ap.add_argument("--lambda-reg", type=float, default=0.25)
    ap.add_argument("--reg-loss", choices=["phuber", "l2", "lpips"], default="phuber")
    ap.add_argument("--fake-steps", type=int, default=1, help="fake-score updates per G update")
    ap.add_argument("--t-min", type=float, default=0.02)
    ap.add_argument("--t-max", type=float, default=0.98)
    ap.add_argument("--num-train-t", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--student", default="", help="init G from this smaller checkpoint instead of the teacher")
    ap.add_argument("--vae", default="", help="decode latent previews with this autoencoder")
    ap.add_argument("--no-dm", action="store_true", help="regression only; skip the distribution-matching gradient")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # The math SDPA kernel materializes a 64x64 score matrix and OOMs at 256.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    real, tok, tck = load_teacher(args.teacher, dev)
    if args.student:
        from t2i.models import T2IDenoiser
        sck = torch.load(args.student, map_location="cpu", weights_only=False)
        G = T2IDenoiser(**sck["config"])
        G.load_state_dict(sck["ema"])
        G = disable_dropout(G).to(dev).train().requires_grad_(True)
    else:
        G = disable_dropout(copy.deepcopy(real)).train().requires_grad_(True)
    fake = None if args.no_dm else disable_dropout(copy.deepcopy(real)).train().requires_grad_(True)
    n_g = count_params(G)
    print(f"generator params: {n_g / 1e6:.2f}M  fake {0 if fake is None else count_params(fake) / 1e6:.2f}M")
    assert n_g < 50e6
    vae, mu, std = None, None, None
    if args.vae:
        import numpy as np
        from t2i.vae import Autoencoder
        vck = torch.load(args.vae, map_location=dev, weights_only=False)
        vae = Autoencoder(vck["z_ch"], vck.get("latent_res", 32), n_down=vck.get("n_down")).to(dev).eval().requires_grad_(False)
        vae.load_state_dict(vck["model"])
        stats = np.load(os.path.join(args.data, "latent_stats.npz"))
        mu = torch.from_numpy(stats["mean"]).float().to(dev).view(1, -1, 1, 1)
        std = torch.from_numpy(stats["std"]).float().to(dev).view(1, -1, 1, 1)
        assert n_g + count_params(vae.decoder) < 50e6, "generator plus decoder must stay under 50M"

    opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr_g, betas=(0.0, 0.999), weight_decay=0.0)
    opt_f = None if fake is None else torch.optim.AdamW(fake.parameters(), lr=args.lr_fake, betas=(0.0, 0.999), weight_decay=0.0)
    ema = EMA(G, args.ema)
    reg_loss_fn = RegressionLoss(args.reg_loss).to(dev)

    pairs = torch.load(args.pairs, map_location="cpu")
    with open(os.path.join(args.data, "captions.json"), encoding="utf-8") as f:
        captions = [c for cs in json.load(f) for c in cs]
    null_ids = tok.encode_batch([""]).to(dev)
    sigmas = karras_sigmas(args.num_train_t, device=dev)  # descending, index 0 = sigma_max
    lo, hi = int(args.t_min * args.num_train_t), int(args.t_max * args.num_train_t)

    step = 0
    ck_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev)
        G.load_state_dict(ck["G"]); ema.shadow.load_state_dict(ck["ema"])
        opt_g.load_state_dict(ck["opt_g"]); step = ck["step"]
        if fake is not None:
            fake.load_state_dict(ck["fake"]); opt_f.load_state_dict(ck["opt_f"])
        print("resumed from", step)

    prompts = load_preview_prompts(args.data)
    prev_ids = tok.encode_batch(prompts).to(dev)
    res, ch = G.img_res, G.in_ch
    prev_z = torch.randn(len(prompts), ch, res, res, generator=torch.Generator().manual_seed(0)).to(dev)

    def to_image(lat):
        if vae is None:
            return lat.clamp(-1, 1)
        return vae.decode(lat.float() / 0.5 * std + mu).float().clamp(-1, 1)
    text_u_cache = None

    def save(path):
        torch.save({"config": G.config, "tokenizer": tok.to_dict(), "G": G.state_dict(),
                    "fake": None if fake is None else fake.state_dict(),
                    "ema": ema.shadow.state_dict(), "opt_g": opt_g.state_dict(),
                    "opt_f": None if opt_f is None else opt_f.state_dict(),
                    "step": step, "args": vars(args)}, path)

    t0 = time.time()
    logs = {}
    while step < args.iters:
        B = args.batch
        ids = tok.encode_batch(random.choices(captions, k=B)).to(dev)
        z = torch.randn(B, ch, res, res, device=dev)

        # ---------------- generator update ----------------
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = gen_forward(G, z, ids)

            # distribution matching gradient (Eq. 7) with the paper's weighting w_t
            if args.no_dm:
                loss_dm = x.sum() * 0
            else:
                with torch.no_grad():
                    t_idx = torch.randint(lo, hi + 1, (B,), device=dev)
                    sigma = sigmas[t_idx]
                    xt = x.detach() + sigma.view(-1, 1, 1, 1) * torch.randn_like(x)
                    text_c = real.encode_text(ids)
                    if text_u_cache is None or text_u_cache[0].shape[0] != B:
                        text_u_cache = real.encode_text(null_ids.expand(B, -1))
                    p_real = guided_denoise(real, xt, sigma, text_c, text_u_cache, args.cfg)
                    p_fake = fake(xt, sigma, ids=ids)
                    w = 1.0 / (x.detach() - p_real).abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-5)
                    grad = torch.nan_to_num((p_fake - p_real) * w)
                loss_dm = 0.5 * F.mse_loss(x, (x - grad).detach())

            # paired regression loss (Eq. 9)
            ri = torch.randint(0, pairs["z"].shape[0], (args.reg_batch,))
            zr, idr, yr = (pairs["z"][ri].to(dev).float(), pairs["ids"][ri].to(dev),
                           pairs["y"][ri].to(dev).float())
            yhat = gen_forward(G, zr, idr)
            loss_reg = reg_loss_fn(yhat, yr)
            loss_g = loss_dm + args.lambda_reg * loss_reg
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        gn_g = torch.nn.utils.clip_grad_norm_(G.parameters(), 10.0)
        opt_g.step()
        ema.update(G)

        # ---------------- fake score update (Eq. 6) ----------------
        loss_f = x.sum() * 0
        if fake is not None:
            for k in range(args.fake_steps):
                if k > 0:  # fresh samples for extra fake updates
                    ids = tok.encode_batch(random.choices(captions, k=B)).to(dev)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        x = gen_forward(G, torch.randn(B, ch, res, res, device=dev), ids)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss_f = edm_loss(fake, x.detach().float(), ids=ids)
                opt_f.zero_grad(set_to_none=True)
                loss_f.backward()
                torch.nn.utils.clip_grad_norm_(fake.parameters(), 1.0)
                opt_f.step()

        step += 1
        for k_, v_ in dict(dm=loss_dm.item(), reg=loss_reg.item(), fake=loss_f.item(), gn=gn_g.item()).items():
            logs[k_] = logs.get(k_, 0.0) + v_
        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step} " + " ".join(f"{k} {v / args.log_every:.4f}" for k, v in logs.items())
                  + f" {args.log_every / dt:.2f} it/s", flush=True)
            logs, t0 = {}, time.time()
        if step % args.sample_every == 0 or step == 1:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                img = to_image(gen_forward(ema.shadow, prev_z, prev_ids))
                yref = to_image(gen_forward(ema.shadow, pairs["z"][:8].to(dev).float(), pairs["ids"][:8].to(dev)))
                yteach = to_image(pairs["y"][:8].to(dev).float())
            grid = torch.cat([img, yref, yteach])
            save_image(grid * 0.5 + 0.5, os.path.join(args.out, f"sample_{step:06d}.png"), nrow=8)
        if step % args.save_every == 0 or step == args.iters:
            save(ck_path)
    save(ck_path)
    print("done")


if __name__ == "__main__":
    main()
