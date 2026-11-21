"""Score a cartoon latent checkpoint on the frozen 500 captions.

Handles teacher checkpoints (EMA + Heun) and DMD checkpoints (EMA + one-step).
Always decodes through the autoencoder saved with the run.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from transformers import CLIPModel, CLIPProcessor

from t2i.edm import SIGMA_MAX, heun_sample, seed_all
from t2i.models import T2IDenoiser, count_params
from t2i.tokenizer import WordTokenizer
from t2i.vae import Autoencoder


def load_net(ck):
    net = T2IDenoiser(**ck["config"])
    # Prefer EMA for both teacher and DMD exports.
    state = ck.get("ema") or ck.get("G") or ck["model"]
    net.load_state_dict(state)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["auto", "onestep", "teacher"], default="auto")
    ap.add_argument("--images", default="", help="uint8 npy of real images at decode resolution")
    ap.add_argument("--eval-captions", default="")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--teacher-steps", type=int, default=18)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = load_net(ck).to(dev).eval()
    tok = WordTokenizer.from_dict(ck["tokenizer"])
    run_args = ck.get("args", {})
    vae_path = run_args.get("vae") or ck.get("vae")
    data_root = run_args.get("data") or os.path.dirname(vae_path)
    cfg = args.cfg if args.cfg is not None else float(run_args.get("cfg") or 2.0)

    mode = args.mode
    if mode == "auto":
        mode = "teacher" if ("model" in ck and "G" not in ck) else "onestep"

    vck = torch.load(vae_path, map_location=dev, weights_only=False)
    vae = Autoencoder(vck["z_ch"], vck.get("latent_res", 32), n_down=vck.get("n_down")).to(dev).eval()
    vae.load_state_dict(vck["model"])
    stats = np.load(os.path.join(data_root, "latent_stats.npz"))
    mu = torch.from_numpy(stats["mean"]).float().to(dev).view(1, -1, 1, 1)
    std = torch.from_numpy(stats["std"]).float().to(dev).view(1, -1, 1, 1)

    # Real images live next to the processed tensors, not necessarily next to latents.
    images_path = args.images
    if not images_path:
        for cand in (
            os.path.join(data_root, "images_128.npy"),
            os.path.join(data_root, "images_64.npy"),
            "data/processed_cartoon128/images_128.npy" if net.img_res == 64 else "data/processed_cartoon/images_64.npy",
            "data/processed_cartoon128/images_128.npy",
            "data/processed_cartoon/images_64.npy",
        ):
            if os.path.exists(cand):
                images_path = cand
                break
    eval_path = args.eval_captions or (
        "data/processed_cartoon128/eval_captions.json"
        if "128" in images_path or net.img_res == 64
        else "data/processed_cartoon/eval_captions.json"
    )
    # Both resolutions share the same 500 captions; prefer the 128 freeze.
    if not os.path.exists(eval_path):
        eval_path = "data/processed_cartoon128/eval_captions.json"

    prompts = json.load(open(eval_path, encoding="utf-8"))[: args.n]
    images = np.load(images_path, mmap_mode="r")
    idx = random.sample(range(len(images)), len(prompts))
    null_ids = tok.encode_batch([""]).to(dev)

    print(
        f"ckpt {args.ckpt}  mode {mode}  latent {net.in_ch}x{net.img_res}  "
        f"params {count_params(net)/1e6:.2f}M  cfg {cfg}  images {images_path}",
        flush=True,
    )

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    isc = InceptionScore(normalize=True).to(dev)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(dev).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_scores = []

    def up(x):
        return F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def decode(lat):
        return vae.decode(lat.float() / 0.5 * std + mu).float().clamp(-1, 1)

    @torch.no_grad()
    def generate(ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "onestep":
                z = torch.randn(ids.shape[0], net.in_ch, net.img_res, net.img_res, device=dev)
                s = torch.full((ids.shape[0],), SIGMA_MAX, device=dev)
                lat = net(z * SIGMA_MAX, s, ids=ids)
            else:
                lat = heun_sample(net, ids, null_ids, steps=args.teacher_steps, cfg=cfg)
        return decode(lat)

    for i in range(0, len(idx), args.batch):
        real = torch.from_numpy(np.stack([images[j] for j in idx[i:i + args.batch]])).permute(0, 3, 1, 2)
        fid.update(up(real.to(dev).float() / 255.0), real=True)
        p = prompts[i:i + args.batch]
        ids = tok.encode_batch(p).to(dev)
        x = (generate(ids) * 0.5 + 0.5).clamp(0, 1)
        fid.update(up(x), real=False)
        isc.update(up(x))
        inp = clip_proc(
            text=p,
            images=[im for im in (x * 255).round().byte().cpu()],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(dev)
        out = clip_model(**inp)
        ie = F.normalize(out.image_embeds.float(), dim=-1)
        te = F.normalize(out.text_embeds.float(), dim=-1)
        clip_scores.extend((100 * (ie * te).sum(-1).clamp(min=0)).tolist())
        print(f"scored {min(i + args.batch, len(idx))}/{len(idx)}", flush=True)

    is_mean, is_std = isc.compute()
    res = {
        "ckpt": args.ckpt,
        "weights": "ema",
        "mode": mode,
        "n": len(prompts),
        "fid": float(fid.compute().item()),
        "is_mean": float(is_mean.item()),
        "is_std": float(is_std.item()),
        "clip_score": float(np.mean(clip_scores)),
        "cfg": cfg,
        "images": images_path,
        "eval_captions": eval_path,
        "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    if mode == "teacher":
        res.update(teacher_steps=args.teacher_steps, nfe=2 * args.teacher_steps - 1)

    if args.latency:
        lat = []
        for k in range(30):
            ids = tok.encode_batch([prompts[k % len(prompts)]]).to(dev)
            torch.cuda.synchronize()
            t0 = time.time()
            _ = generate(ids)
            torch.cuda.synchronize()
            lat.append(time.time() - t0)
        lat = lat[5:]
        res["latency_ms_bs1_mean"] = 1000 * float(np.mean(lat))
        res["latency_ms_bs1_max"] = 1000 * float(np.max(lat))

    print(json.dumps(res, indent=2), flush=True)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "eval_500.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
