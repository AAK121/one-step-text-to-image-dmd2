"""Vanilla DDPM baseline. Same T2IDenoiser and the same data as the EDM teacher.

Linear beta schedule, 1000 diffusion timesteps, epsilon prediction, 10% caption dropout,
AdamW. Sampling is DDIM. This objective is separate from the EDM teacher: DMD does not
distill this checkpoint.
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from t2i.data import CaptionImageDataset
from t2i.edm import EMA, seed_all
from t2i.models import T2IDenoiser, count_params
from train_teacher import PREVIEW_PROMPTS


def linear_alpha_bar(T, beta_start, beta_end, device):
    beta = torch.linspace(beta_start, beta_end, T, device=device)
    return torch.cumprod(1.0 - beta, dim=0)


@torch.no_grad()
def ddim_sample(net, ids, null_ids, alpha_bar, steps=50, cfg=2.0, z=None):
    """Deterministic DDIM (eta = 0) on the epsilon prediction. Returns images in [-1, 1]."""
    device = ids.device
    B = ids.shape[0]
    T = alpha_bar.shape[0]
    if z is None:
        z = torch.randn(B, 3, net.img_res, net.img_res, device=device)
    # Evenly spaced timesteps, descending, always including the last training step.
    raw = torch.linspace(T - 1, 0, steps, device=device).round().long()
    ts = torch.unique_consecutive(raw)
    x = z.float()
    text_c = net.encode_text(ids)
    text_u = net.encode_text(null_ids.expand(B, -1))
    for i, t in enumerate(ts):
        ab = alpha_bar[t]
        ab_prev = alpha_bar[ts[i + 1]] if i + 1 < len(ts) else x.new_tensor(1.0)
        t_batch = t.expand(B)
        eps_c = net.predict_eps(x, t_batch, text=text_c)
        if cfg == 1.0:
            eps = eps_c
        else:
            eps_u = net.predict_eps(x, t_batch, text=text_u)
            eps = eps_u + cfg * (eps_c - eps_u)
        eps = eps.float()
        x0 = ((x - (1.0 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return x.clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/ddpm")
    ap.add_argument("--iters", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--mults", default="1,2,2,2")
    ap.add_argument("--attn-res", default="64,32")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--drop-caption", type=float, default=0.1)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--beta-start", type=float, default=1e-4)
    ap.add_argument("--beta-end", type=float, default=0.02)
    ap.add_argument("--preview-steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--sample-every", type=int, default=5000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # The math SDPA kernel materializes a 64x64 score matrix and OOMs at this resolution.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    ds = CaptionImageDataset(args.data, res=args.res, drop_caption=args.drop_caption)
    dl = DataLoader(ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.workers > 0)
    net = T2IDenoiser(vocab_size=len(ds.tok), max_len=ds.tok.max_len, ch=args.ch,
                      mults=tuple(int(m) for m in args.mults.split(",")),
                      attn_res=tuple(int(r) for r in args.attn_res.split(",")),
                      img_res=args.res, dropout=args.dropout).to(dev)
    n = count_params(net)
    print(f"device {torch.cuda.get_device_name(0)}  res {args.res}  attn {args.attn_res}  T {args.T}")
    print(f"ddpm params: {n / 1e6:.2f}M (text encoder {count_params(net.text) / 1e6:.2f}M)")
    assert n < 50e6, "model must have < 50M parameters"

    alpha_bar = linear_alpha_bar(args.T, args.beta_start, args.beta_end, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    ema = EMA(net, args.ema)
    step = 0
    ckpt_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print("resumed from step", step)

    preview_ids = ds.tok.encode_batch(PREVIEW_PROMPTS).to(dev)
    null_ids = ds.tok.encode_batch([""]).to(dev)
    preview_z = torch.randn(len(PREVIEW_PROMPTS), 3, net.img_res, net.img_res,
                            generator=torch.Generator().manual_seed(0)).to(dev)

    def save(path):
        torch.save({"config": net.config, "tokenizer": ds.tok.to_dict(), "model": net.state_dict(),
                    "ema": ema.shadow.state_dict(), "opt": opt.state_dict(), "step": step,
                    "args": vars(args)}, path)

    net.train()
    t0, run_loss = time.time(), 0.0
    while step < args.iters:
        for x, ids in dl:
            x, ids = x.to(dev, non_blocking=True), ids.to(dev, non_blocking=True)
            lr = args.lr * min(1.0, (step + 1) / args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            B = x.shape[0]
            t = torch.randint(0, args.T, (B,), device=dev)
            eps = torch.randn_like(x)
            ab = alpha_bar[t].view(-1, 1, 1, 1)
            xt = ab.sqrt() * x + (1.0 - ab).sqrt() * eps
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net.predict_eps(xt, t, ids=ids)
                loss = F.mse_loss(pred.float(), eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ema.update(net, min(args.ema, (1 + step) / (10 + step)))
            step += 1
            run_loss += loss.item()

            if step % args.log_every == 0:
                dt = time.time() - t0
                print(f"step {step} loss {run_loss / args.log_every:.4f} gnorm {gn:.2f} lr {lr:.2e} "
                      f"{args.log_every / dt:.2f} it/s  mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                      flush=True)
                t0, run_loss = time.time(), 0.0
            if step % args.sample_every == 0:
                ema.shadow.eval()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    img = ddim_sample(ema.shadow, preview_ids, null_ids, alpha_bar,
                                      steps=args.preview_steps, cfg=args.cfg, z=preview_z)
                save_image(img * 0.5 + 0.5, os.path.join(args.out, f"sample_{step:07d}.png"), nrow=8)
                net.train()
            if step % args.save_every == 0 or step == args.iters:
                save(ckpt_path)
            if step >= args.iters:
                break
    save(ckpt_path)
    print("done")


if __name__ == "__main__":
    main()
