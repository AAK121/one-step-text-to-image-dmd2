"""Stage A: train the text-conditioned EDM diffusion teacher from scratch.

This plays the role of the "pretrained base model" mu_base in DMD (Yin et al., CVPR 2024).
Captions are dropped 10% of the time so classifier-free guidance can be used.
"""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from t2i.data import CaptionImageDataset
from t2i.edm import EMA, edm_loss, heun_sample, seed_all
from t2i.models import T2IDenoiser, count_params

PREVIEW_PROMPTS = [
    "1girl, long hair, pink hair, green eyes, smile, portrait",
    "1girl, short hair, blue hair, blue eyes, blush, portrait",
    "1girl, long hair, white hair, red eyes, expressionless, portrait",
    "1girl, short hair, black hair, brown eyes, smile, school uniform",
    "1boy, short hair, brown hair, blue eyes, smile, portrait",
    "1girl, long hair, blonde hair, green eyes, open mouth, portrait",
    "1girl, twintails, pink hair, pink eyes, blush, smile",
    "1girl, long hair, purple hair, yellow eyes, looking at viewer, portrait",
]


def load_preview_prompts(data_dir=None):
    """Anime prompts stay the default. A dataset can drop preview_prompts.json beside its captions."""
    if data_dir:
        path = os.path.join(data_dir, "preview_prompts.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return PREVIEW_PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/teacher")
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
    print(f"device {torch.cuda.get_device_name(0)}  res {args.res}  attn {args.attn_res}")
    print(f"teacher params: {n / 1e6:.2f}M (text encoder {count_params(net.text) / 1e6:.2f}M)")
    # The teacher is a training oracle. The 50M cap stays on the submitted generator
    # in train_dmd.py and export.py, which is the only network test.py loads.
    if n >= 50e6:
        print(f"teacher exceeds 50M ({n / 1e6:.2f}M); it is not a submission candidate")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    ema = EMA(net, args.ema)
    step = 0
    ckpt_path = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); step = ck["step"]
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = edm_loss(net, x, ids=ids)
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
                      f"{args.log_every / dt:.2f} it/s", flush=True)
                t0, run_loss = time.time(), 0.0
            if step % args.sample_every == 0:
                ema.shadow.eval()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    img = heun_sample(ema.shadow, preview_ids, null_ids, steps=18, cfg=2.0, z=preview_z)
                save_image(img * 0.5 + 0.5, os.path.join(args.out, f"sample_{step:07d}.png"), nrow=8)
            if step % args.save_every == 0 or step == args.iters:
                save(ckpt_path)
            if step >= args.iters:
                break
    save(ckpt_path)
    print("done")


if __name__ == "__main__":
    main()
