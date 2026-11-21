"""Evaluation: FID, Inception Score, CLIP score and latency (batch size 1).

Pretrained Inception / CLIP networks are used *only as metrics* (standard practice),
never inside the generator. Requires: pip install -r requirements-train.txt

  python evaluate.py --ckpt checkpoints/generator.pt --mode onestep --n 5000
  python evaluate.py --ckpt checkpoints/teacher.pt   --mode teacher --teacher-steps 18 --n 5000
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from t2i.edm import heun_sample, one_step, seed_all
from test import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/generator.pt")
    ap.add_argument("--mode", choices=["onestep", "teacher"], default="onestep")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--teacher-steps", type=int, default=18)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out", default=None, help="optional json file for the results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_all(args.seed)
    dev = "cuda"

    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from transformers import CLIPModel, CLIPProcessor

    net, tok = load(args.ckpt, dev)
    images = np.load(os.path.join(args.data, f"images_{net.img_res}.npy"), mmap_mode="r")
    eval_path = os.path.join(args.data, "eval_captions.json")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            prompts = json.load(f)[: args.n]
        idx = random.sample(range(len(images)), len(prompts))
    else:
        with open(os.path.join(args.data, "captions.json"), encoding="utf-8") as f:
            captions = json.load(f)
        idx = random.sample(range(len(images)), min(args.n, len(images)))
        prompts = [random.choice(captions[i]) for i in idx]
    null_ids = tok.encode_batch([""]).to(dev)

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    isc = InceptionScore(normalize=True).to(dev)
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(dev).eval()
    clip_proc = CLIPProcessor.from_pretrained(args.clip_model)
    clip_scores = []

    @torch.no_grad()
    def clip_update(x01, texts):
        """CLIP score = 100 * max(cos(E_img, E_txt), 0) (Hessel et al. 2021), averaged over samples."""
        inp = clip_proc(text=texts, images=[im for im in (x01 * 255).round().byte().cpu()],
                        return_tensors="pt", padding=True, truncation=True, max_length=77).to(dev)
        out = clip_model(**inp)
        ie = F.normalize(out.image_embeds.float(), dim=-1)
        te = F.normalize(out.text_embeds.float(), dim=-1)
        clip_scores.extend((100 * (ie * te).sum(-1).clamp(min=0)).tolist())
    up = lambda x: F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

    for i in range(0, len(idx), args.batch):
        real = torch.from_numpy(np.stack([images[j] for j in idx[i:i + args.batch]])).permute(0, 3, 1, 2)
        fid.update(up(real.to(dev).float() / 255.0), real=True)
        p = prompts[i:i + args.batch]
        ids = tok.encode_batch(p).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = one_step(net, ids) if args.mode == "onestep" else \
                heun_sample(net, ids, null_ids, steps=args.teacher_steps, cfg=args.cfg)
        x = (x.float() * 0.5 + 0.5).clamp(0, 1)
        fid.update(up(x), real=False)
        isc.update(up(x))
        clip_update(x, p)

    # latency: batch size 1, includes text encoding, after warm-up
    lat = []
    for k in range(60):
        ids = tok.encode_batch([prompts[k % len(prompts)]]).to(dev)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = one_step(net, ids) if args.mode == "onestep" else \
                heun_sample(net, ids, null_ids, steps=args.teacher_steps, cfg=args.cfg)
        torch.cuda.synchronize(); lat.append(time.time() - t0)
    lat = lat[10:]

    is_mean, is_std = isc.compute()
    res = {"ckpt": args.ckpt, "mode": args.mode, "n": len(idx), "fid": fid.compute().item(),
           "is_mean": is_mean.item(), "is_std": is_std.item(), "clip_score": float(np.mean(clip_scores)),
           "latency_ms_bs1_mean": 1000 * float(np.mean(lat)), "latency_ms_bs1_max": 1000 * float(np.max(lat))}
    if args.mode == "teacher":
        res.update(teacher_steps=args.teacher_steps, nfe=2 * args.teacher_steps - 1, cfg=args.cfg)
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
