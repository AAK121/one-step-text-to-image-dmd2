"""Train the portrait autoencoder from scratch, then write scaled latents.

Latents are the encoder mean, shifted and scaled to std 0.5 so the EDM teacher
can use its usual sigma_data.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from t2i.models import count_params
from t2i.vae import Autoencoder, ae_loss


class ImageSet(Dataset):
    def __init__(self, path):
        self.images = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.images[i])).permute(2, 0, 1).float() / 127.5 - 1.0
        if torch.rand(()) < 0.5:
            x = x.flip(-1)
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed_portrait")
    ap.add_argument("--out", default="runs/vae")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--z-ch", type=int, default=16)
    ap.add_argument("--latent-res", type=int, default=32, choices=[32, 64, 128])
    ap.add_argument("--kl", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    images_path = os.path.join(args.data, f"images_{args.res}.npy")
    ds = ImageSet(images_path)
    dl = DataLoader(ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.workers > 0)
    n_down = int(round(math.log2(args.res / args.latent_res)))
    if args.latent_res * (2 ** n_down) != args.res or n_down not in (1, 2, 3):
        raise SystemExit(f"cannot map {args.res} images to latent {args.latent_res} in 1-3 downsamples")
    net = Autoencoder(args.z_ch, args.latent_res, n_down=n_down).to(dev)
    n_dec = count_params(net.decoder)
    print(f"device {torch.cuda.get_device_name(0)}  images {len(ds)}")
    print(f"autoencoder {count_params(net) / 1e6:.2f}M  decoder {n_dec / 1e6:.2f}M  "
          f"latent {args.latent_res}x{args.latent_res}x{args.z_ch}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    preview = torch.from_numpy(np.array(ds.images[:8])).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)

    step, epochs_done = 0, 0
    t0, run_rec, run_kl = time.time(), 0.0, 0.0
    target = args.epochs * len(dl)
    while step < target:
        for x in dl:
            x = x.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                recon, mean, logvar = net(x)
                loss, rec, kl = ae_loss(recon.float(), x.float(), mean.float(), logvar.float(), args.kl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            run_rec += rec.item()
            run_kl += kl.item()
            if step % args.log_every == 0:
                dt = time.time() - t0
                print(f"step {step}/{target} rec {run_rec / args.log_every:.4f} "
                      f"kl {run_kl / args.log_every:.4f} {args.log_every / dt:.2f} it/s", flush=True)
                t0, run_rec, run_kl = time.time(), 0.0, 0.0
            if step % args.sample_every == 0 or step == target:
                net.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = net.decode(net.encode_mean(preview)).float()
                grid = torch.cat([preview, pred], dim=0) * 0.5 + 0.5
                save_image(grid, os.path.join(args.out, f"recon_{step:07d}.png"), nrow=8)
                net.train()
            if step % args.save_every == 0 or step == target:
                torch.save({"model": net.state_dict(), "z_ch": args.z_ch,
                            "latent_res": args.latent_res, "n_down": n_down, "step": step,
                            "args": vars(args)}, os.path.join(args.out, "last.pt"))
            if step >= target:
                break
        epochs_done += 1
    for name in ("captions.json", "tokenizer.json", "preview_prompts.json", "eval_captions.json"):
        src = os.path.join(args.data, name)
        dst = os.path.join(args.out, name)
        if not os.path.exists(src):
            continue
        if os.path.abspath(src) != os.path.abspath(dst):
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
    encode(net, images_path, args.out, dev, args.batch)
    print("done")


@torch.no_grad()
def encode(net, images_path, out_dir, dev, batch):
    net.eval()
    images = np.load(images_path, mmap_mode="r")
    n = len(images)
    # Two passes so a 128 latent (about 15 GB at fp16) is never held in RAM.
    sum_c = torch.zeros(net.z_ch, dtype=torch.float64)
    sumsq = torch.zeros(net.z_ch, dtype=torch.float64)
    count = 0
    shape = None
    for i in range(0, n, batch):
        x = torch.from_numpy(np.array(images[i:i + batch])).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = net.encode_mean(x).float()
        sum_c += m.sum(dim=(0, 2, 3)).double().cpu()
        sumsq += m.square().sum(dim=(0, 2, 3)).double().cpu()
        count += m.shape[0] * m.shape[2] * m.shape[3]
        shape = m.shape[1:]
        if i % (batch * 20) == 0:
            print(f"encode stats {i}/{n}", flush=True)
    mu = sum_c / count
    std = (sumsq / count - mu.square()).clamp(min=1e-8).sqrt()
    out = np.lib.format.open_memmap(os.path.join(out_dir, "latents.npy"), mode="w+",
                                    dtype=np.float16, shape=(n, *shape))
    mu_t = mu.float().to(dev).view(1, -1, 1, 1)
    std_t = std.float().to(dev).view(1, -1, 1, 1)
    for i in range(0, n, batch):
        x = torch.from_numpy(np.array(images[i:i + batch])).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = net.encode_mean(x).float()
        out[i:i + m.shape[0]] = ((m - mu_t) / std_t * 0.5).half().cpu().numpy()
        if i % (batch * 20) == 0:
            print(f"encode write {i}/{n}", flush=True)
    out.flush()
    np.savez(os.path.join(out_dir, "latent_stats.npz"), mean=mu.float().numpy(), std=std.float().numpy())
    print(f"latents {tuple(out.shape)}  per-channel std before scale "
          f"{std.min().item():.3f}-{std.max().item():.3f}")


if __name__ == "__main__":
    main()
