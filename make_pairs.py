"""Stage B: build the paired regression dataset D = {(z, caption, y)} for DMD (Sec. 3.3).

y is obtained by running the deterministic Heun ODE sampler of the (guided) teacher from
noise z. Following the paper, this only needs a modest number of pairs.
"""
import argparse
import json
import os
import random

import torch
from tqdm import tqdm

from t2i.edm import heun_sample, seed_all
from t2i.models import T2IDenoiser
from t2i.tokenizer import WordTokenizer


def load_teacher(path, dev):
    ck = torch.load(path, map_location="cpu")
    net = T2IDenoiser(**ck["config"])
    net.load_state_dict(ck["ema"])
    return net.to(dev).eval().requires_grad_(False), WordTokenizer.from_dict(ck["tokenizer"]), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher/last.pt")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/pairs.pt")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=40, help="Heun steps (2*steps-1 NFE)")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--vae", default="", help="decode latent samples for the preview grid")
    args = ap.parse_args()
    seed_all(args.seed)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # The math SDPA kernel materializes a 64x64 score matrix and OOMs at 256.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    net, tok, _ = load_teacher(args.teacher, dev)
    with open(os.path.join(args.data, "captions.json"), encoding="utf-8") as f:
        caps = [c for cs in json.load(f) for c in cs]
    null_ids = tok.encode_batch([""]).to(dev)

    Z, IDS, Y, TXT = [], [], [], []
    for _ in tqdm(range(0, args.n, args.batch)):
        texts = random.choices(caps, k=args.batch)
        ids = tok.encode_batch(texts).to(dev)
        z = torch.randn(args.batch, net.in_ch, net.img_res, net.img_res, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = heun_sample(net, ids, null_ids, steps=args.steps, cfg=args.cfg, z=z)
        Z.append(z.half().cpu()); IDS.append(ids.cpu()); Y.append(y.half().cpu()); TXT += texts
    out = {"z": torch.cat(Z)[: args.n], "ids": torch.cat(IDS)[: args.n], "y": torch.cat(Y)[: args.n],
           "texts": TXT[: args.n], "cfg": args.cfg, "steps": args.steps}
    torch.save(out, args.out)
    from torchvision.utils import save_image
    preview = out["y"][:64].float()
    if args.vae:
        import numpy as np
        from t2i.vae import Autoencoder
        ck = torch.load(args.vae, map_location=dev, weights_only=False)
        vae = Autoencoder(ck["z_ch"], ck.get("latent_res", 32), n_down=ck.get("n_down")).to(dev).eval()
        vae.load_state_dict(ck["model"])
        stats = np.load(os.path.join(args.data, "latent_stats.npz"))
        mu = torch.from_numpy(stats["mean"]).float().to(dev).view(1, -1, 1, 1)
        std = torch.from_numpy(stats["std"]).float().to(dev).view(1, -1, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            preview = vae.decode(preview.to(dev) / 0.5 * std + mu).float().cpu()
    save_image(preview * 0.5 + 0.5, args.out.replace(".pt", "_preview.png"), nrow=8)
    print("saved", args.out, tuple(out["y"].shape))


if __name__ == "__main__":
    main()
