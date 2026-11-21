"""Sample the frozen preview prompts at several guidance scales.

Uses the EMA weights and the same noise as the training previews, so cfg 2.0
lines up with sample_0030000.png. Character names are already absent from these
prompts and from the training captions.
"""
import argparse
import os

import torch
from torchvision.utils import save_image

from t2i.edm import heun_sample
from t2i.models import T2IDenoiser
from t2i.tokenizer import WordTokenizer
from train_ddpm import ddim_sample, linear_alpha_bar
from train_teacher import PREVIEW_PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/cfg_sweep")
    ap.add_argument("--cfgs", default="1,2,4")
    ap.add_argument("--kind", choices=["auto", "ddpm", "teacher"], default="auto")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    kind = args.kind
    if kind == "auto":
        kind = "ddpm" if "T" in ck.get("args", {}) else "teacher"
    steps = args.steps if args.steps is not None else (50 if kind == "ddpm" else 18)
    dev = "cuda"
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    net = T2IDenoiser(**ck["config"])
    net.load_state_dict(ck["ema"])
    net = net.to(dev).eval()
    tok = WordTokenizer.from_dict(ck["tokenizer"])
    ids = tok.encode_batch(PREVIEW_PROMPTS).to(dev)
    null = tok.encode_batch([""]).to(dev)
    z = torch.randn(len(PREVIEW_PROMPTS), 3, net.img_res, net.img_res,
                    generator=torch.Generator().manual_seed(0)).to(dev)
    alpha = None
    if kind == "ddpm":
        a = ck["args"]
        alpha = linear_alpha_bar(a["T"], a["beta_start"], a["beta_end"], dev)

    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(os.path.dirname(args.ckpt) or "ckpt"))[0]
    print(f"{name} kind {kind} step {ck.get('step')} steps {steps}")
    for cfg in [float(x) for x in args.cfgs.split(",")]:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "ddpm":
                img = ddim_sample(net, ids, null, alpha, steps=steps, cfg=cfg, z=z)
            else:
                img = heun_sample(net, ids, null, steps=steps, cfg=cfg, z=z)
        path = os.path.join(args.out, f"{name}_cfg{cfg:g}_s{steps}.png")
        save_image(img.float() * 0.5 + 0.5, path, nrow=len(PREVIEW_PROMPTS))
        print("wrote", path)


if __name__ == "__main__":
    main()
