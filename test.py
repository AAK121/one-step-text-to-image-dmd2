"""Inference script used for grading.

    python test.py --test-prompts /abs/path/prompts.csv --output-dir /abs/path/out

CSV format: header `id,prompt` (or bare rows). One image `image_<id>.png` is written per row.
Loads checkpoints/generator.pt by default: one network evaluation (+ decoder if latent).
"""
import argparse
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from t2i.edm import SIGMA_MAX, heun_sample  # noqa: E402
from t2i.models import T2IDenoiser, count_params  # noqa: E402
from t2i.tokenizer import WordTokenizer  # noqa: E402
from t2i.vae import Decoder  # noqa: E402


def read_prompts(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        has_header = header is not None and header[0].strip().lower() == "id"
        if header is not None and not has_header:
            reader = [header] + list(reader)
        for r in reader:
            if not r or not r[0].strip():
                continue
            rows.append((r[0].strip(), ",".join(r[1:]).strip().strip('"')))
    return rows


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = T2IDenoiser(**ck["config"])
    net.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    net = net.to(device).eval().requires_grad_(False)
    tok = WordTokenizer.from_dict(ck["tokenizer"])
    decoder = mu = std = None
    if ck.get("vae"):
        v = ck["vae"]
        decoder = Decoder(v["z_ch"], v.get("n_down") or ({32: 3, 64: 2, 128: 1}[v.get("latent_res", 32)]))
        decoder.load_state_dict({k: v_.float() for k, v_ in v["decoder"].items()})
        decoder = decoder.to(device).eval().requires_grad_(False)
        mu = torch.from_numpy(v["mean"]).float().to(device).view(1, -1, 1, 1)
        std = torch.from_numpy(v["std"]).float().to(device).view(1, -1, 1, 1)
    return net, tok, decoder, mu, std, ck


def to_pil(x, size):
    x = (x.float().clamp(-1, 1) * 0.5 + 0.5)
    if size is not None and size != x.shape[-1]:
        x = F.interpolate(x[None], size=(size, size), mode="bicubic", align_corners=False)[0].clamp(0, 1)
    return Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"))


@torch.no_grad()
def generate(net, ids, z, decoder, mu, std, mode, null_ids, teacher_steps, cfg, device, amp_dtype, use_amp):
    with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
        if mode == "onestep":
            s = torch.full((z.shape[0],), SIGMA_MAX, device=z.device)
            lat = net(z * SIGMA_MAX, s, ids=ids)
        else:
            lat = heun_sample(net, ids, null_ids, steps=teacher_steps, cfg=cfg, z=z)
    if decoder is None:
        return lat.float().clamp(-1, 1)
    return decoder(lat.float() / 0.5 * std + mu).float().clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-prompts", "--test_prompts", dest="test_prompts", required=True)
    ap.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    ap.add_argument("--mode", choices=["onestep", "teacher"], default="onestep")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--size", type=int, default=None,
                    help="optional resize of the saved PNG; default keeps the native decode size")
    ap.add_argument("--teacher-steps", type=int, default=18)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or os.path.join(HERE, "checkpoints", "generator.pt")
    net, tok, decoder, mu, std, ck = load(ckpt, device)
    n_den = count_params(net)
    n_dec = count_params(decoder) if decoder is not None else 0
    print(f"loaded {ckpt} (denoiser {n_den / 1e6:.2f}M"
          + (f" + decoder {n_dec / 1e6:.2f}M" if n_dec else "")
          + f") on {device}")
    assert n_den + n_dec < 50e6, n_den + n_dec

    os.makedirs(args.output_dir, exist_ok=True)
    rows = read_prompts(args.test_prompts)
    print(f"{len(rows)} prompts")

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    null_ids = tok.encode_batch([""]).to(device)
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    in_ch = getattr(net, "in_ch", 3)

    # warm-up
    z0 = torch.randn(1, in_ch, net.img_res, net.img_res, device=device)
    generate(net, tok.encode_batch(["warm up"]).to(device), z0, decoder, mu, std,
             args.mode, null_ids, args.teacher_steps, args.cfg, device, amp_dtype, use_amp)

    per_image = []
    total_t0 = time.time()
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        ids = tok.encode_batch([p for _, p in chunk]).to(device)
        z = torch.randn(len(chunk), in_ch, net.img_res, net.img_res, generator=g).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        imgs = generate(net, ids, z, decoder, mu, std, args.mode, null_ids,
                        args.teacher_steps, args.cfg, device, amp_dtype, use_amp)
        if device == "cuda":
            torch.cuda.synchronize()
        per_image.append((time.time() - t0) / len(chunk))
        for (pid, _), im in zip(chunk, imgs):
            to_pil(im, args.size).save(os.path.join(args.output_dir, f"image_{pid}.png"))

    print(f"generated {len(rows)} images in {time.time() - total_t0:.2f}s (incl. saving); "
          f"model time per image: mean {1000 * sum(per_image) / max(len(per_image), 1):.2f} ms, "
          f"max {1000 * max(per_image, default=0):.2f} ms")
    if device == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
