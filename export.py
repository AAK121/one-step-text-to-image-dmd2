"""Export a slim inference checkpoint into checkpoints/generator.pt.

Bundles the EMA one-step denoiser, tokenizer, and (for latent models) the
autoencoder decoder plus latent mean/std used at decode time.

  python export.py \
      --src runs/dmd_cartoon128_dmd2/last.pt \
      --vae runs/vae_cartoon128/last.pt \
      --data runs/vae_cartoon128 \
      --dst checkpoints/generator.pt
"""
import argparse
import os

import numpy as np
import torch

from t2i.models import T2IDenoiser, count_params
from t2i.vae import Autoencoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--kind", choices=["onestep", "teacher"], default="onestep")
    ap.add_argument("--dst", default="checkpoints/generator.pt")
    ap.add_argument("--vae", default="", help="autoencoder checkpoint; required for latent models")
    ap.add_argument("--data", default="", help="directory with latent_stats.npz when --vae is set")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ck["ema"]
    net = T2IDenoiser(**ck["config"])
    net.load_state_dict(sd)
    n_den = count_params(net)

    out = {
        "kind": args.kind,
        "config": ck["config"],
        "tokenizer": ck["tokenizer"],
        "state_dict": sd if args.fp32 else {
            k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()
        },
        "num_params": n_den,
        "train_step": ck.get("step"),
    }

    n_dec = 0
    vae_path = args.vae or (ck.get("args") or {}).get("vae", "")
    data_root = args.data or (ck.get("args") or {}).get("data", "")
    if ck["config"].get("in_ch", 3) != 3:
        if not vae_path or not os.path.exists(vae_path):
            raise SystemExit(f"latent model needs --vae (got {vae_path!r})")
        if not data_root:
            data_root = os.path.dirname(vae_path)
        stats_path = os.path.join(data_root, "latent_stats.npz")
        if not os.path.exists(stats_path):
            raise SystemExit(f"missing {stats_path}")
        vck = torch.load(vae_path, map_location="cpu", weights_only=False)
        ae = Autoencoder(vck["z_ch"], vck.get("latent_res", 32), n_down=vck.get("n_down"))
        ae.load_state_dict(vck["model"])
        n_dec = count_params(ae.decoder)
        assert n_den + n_dec < 50e6, (n_den, n_dec)
        dec_sd = ae.decoder.state_dict()
        if not args.fp32:
            dec_sd = {k: (v.half() if v.is_floating_point() else v) for k, v in dec_sd.items()}
        stats = np.load(stats_path)
        out.update({
            "vae": {
                "z_ch": vck["z_ch"],
                "latent_res": vck.get("latent_res", 32),
                "n_down": vck.get("n_down"),
                "decoder": dec_sd,
                "mean": stats["mean"].astype("float32"),
                "std": stats["std"].astype("float32"),
            },
            "num_params": n_den + n_dec,
            "num_params_denoiser": n_den,
            "num_params_decoder": n_dec,
        })
    else:
        assert n_den < 50e6, n_den

    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    torch.save(out, args.dst)
    print(
        f"exported {args.kind} denoiser {n_den / 1e6:.2f}M"
        + (f" + decoder {n_dec / 1e6:.2f}M = {out['num_params'] / 1e6:.2f}M" if n_dec else "")
        + f" -> {args.dst} ({os.path.getsize(args.dst) / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
