"""EDM teacher on the portrait autoencoder's 32x32 latent.

The decoder turns a sample back into a 256 image for the preview grid. This
teacher is an oracle. The submitted one-step model still has to stay under 50M
once the decoder is counted with it.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from t2i.edm import EMA, edm_loss, heun_sample, seed_all
from t2i.models import T2IDenoiser, count_params
from t2i.tokenizer import WordTokenizer
from t2i.vae import Autoencoder
from train_teacher import load_preview_prompts


class LatentSet(Dataset):
    def __init__(self, root, drop_caption=0.1):
        self.latents = np.load(os.path.join(root, "latents.npy"), mmap_mode="r")
        with open(os.path.join(root, "captions.json"), encoding="utf-8") as f:
            self.captions = json.load(f)
        self.tok = WordTokenizer.load(os.path.join(root, "tokenizer.json"))
        self.drop_caption = drop_caption

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, i):
        z = torch.from_numpy(np.array(self.latents[i])).float()
        cap = "" if random.random() < self.drop_caption else self.captions[i][0]
        return z, torch.tensor(self.tok.encode(cap), dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed_portrait")
    ap.add_argument("--vae", default="runs/vae/last.pt")
    ap.add_argument("--out", default="runs/teacher_latent")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--mults", default="1,2,2,2")
    ap.add_argument("--attn-res", default="16,8")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--drop-caption", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    ds = LatentSet(args.data, args.drop_caption)
    dl = DataLoader(ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.workers > 0)
    z_ch = ds.latents.shape[1]
    net = T2IDenoiser(vocab_size=len(ds.tok), max_len=ds.tok.max_len, ch=args.ch,
                      mults=tuple(int(m) for m in args.mults.split(",")),
                      attn_res=tuple(int(r) for r in args.attn_res.split(",")),
                      img_res=ds.latents.shape[-1], in_ch=z_ch, dropout=args.dropout).to(dev)
    vae_ck = torch.load(args.vae, map_location=dev, weights_only=False)
    vae = Autoencoder(vae_ck["z_ch"], vae_ck.get("latent_res", 32), n_down=vae_ck.get("n_down")).to(dev)
    vae.load_state_dict(vae_ck["model"])
    vae.eval().requires_grad_(False)
    stats = np.load(os.path.join(args.data, "latent_stats.npz"))
    mu = torch.from_numpy(stats["mean"]).float().to(dev).view(1, -1, 1, 1)
    std = torch.from_numpy(stats["std"]).float().to(dev).view(1, -1, 1, 1)

    n = count_params(net)
    print(f"device {torch.cuda.get_device_name(0)}  latent {z_ch}x{net.img_res}  attn {args.attn_res}")
    print(f"latent teacher {n / 1e6:.2f}M  decoder {count_params(vae.decoder) / 1e6:.2f}M")
    if n >= 50e6:
        print(f"teacher exceeds 50M ({n / 1e6:.2f}M); it is not a submission candidate")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    ema = EMA(net, args.ema)
    step = 0

    prompts = load_preview_prompts(args.data)
    preview_ids = ds.tok.encode_batch(prompts).to(dev)
    null_ids = ds.tok.encode_batch([""]).to(dev)
    preview_z = torch.randn(len(prompts), z_ch, net.img_res, net.img_res,
                            generator=torch.Generator().manual_seed(0)).to(dev)

    def save(path):
        torch.save({"config": net.config, "tokenizer": ds.tok.to_dict(), "model": net.state_dict(),
                    "ema": ema.shadow.state_dict(), "opt": opt.state_dict(), "step": step,
                    "vae": args.vae, "args": vars(args)}, path)

    def preview():
        ema.shadow.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lat = heun_sample(ema.shadow, preview_ids, null_ids, steps=18, cfg=2.0, z=preview_z)
        raw = lat.float() / 0.5 * std + mu
        with torch.autocast("cuda", dtype=torch.bfloat16):
            img = vae.decode(raw).float()
        save_image(img * 0.5 + 0.5, os.path.join(args.out, f"sample_{step:07d}.png"), nrow=8)

    net.train()
    t0, run_loss = time.time(), 0.0
    while step < args.iters:
        for z, ids in dl:
            z, ids = z.to(dev, non_blocking=True), ids.to(dev, non_blocking=True)
            lr = args.lr * min(1.0, (step + 1) / args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = edm_loss(net, z, ids=ids)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ema.update(net, min(args.ema, (1 + step) / (10 + step)))
            step += 1
            run_loss += loss.item()
            if step % args.log_every == 0:
                dt = time.time() - t0
                print(f"step {step} loss {run_loss / args.log_every:.4f} gnorm {gn:.2f} "
                      f"{args.log_every / dt:.2f} it/s", flush=True)
                t0, run_loss = time.time(), 0.0
            if step % args.sample_every == 0:
                preview()
            if step % args.save_every == 0 or step == args.iters:
                save(os.path.join(args.out, "last.pt"))
            if step >= args.iters:
                break
    save(os.path.join(args.out, "last.pt"))
    print("done")


if __name__ == "__main__":
    main()
