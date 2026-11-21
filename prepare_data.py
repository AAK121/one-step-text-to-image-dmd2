"""Build the training tensors for the text-to-image model.

Two input formats are supported:

  1) --source cub      : the open-source CUB-200-2011 + captions parquet files
                         (huggingface: weiywang/CUB_200_2011_CAP). Used as a stand-in
                         until our own curated dataset (Part 1) is ready.
  2) --source csv      : generic format produced by the Part-1 curation pipeline:
                         a CSV with columns `image_path,caption` (one row per caption;
                         several rows may share the same image). Relative paths are
                         resolved against --image-root.

Output (in --out):
  images_<res>.npy   uint8 array (N, res, res, 3), center-cropped + resized
  captions.json      list (len N) of lists of caption strings
  tokenizer.json     word-level vocabulary built from the captions
"""
import argparse
import csv
import glob
import io
import json
import os
import random
import re
from collections import Counter, OrderedDict

import numpy as np
from PIL import Image
from tqdm import tqdm

from t2i.tokenizer import WordTokenizer, basic_tokenize


def center_crop_resize(img: Image.Image, res: int) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((res, res), Image.BICUBIC, reducing_gap=3.0)
    return np.asarray(img, dtype=np.uint8)


# ----------------------------------------------------------------------------- CUB
STARTERS = ["this", "these", "the", "an", "a", "small", "large", "tiny", "it"]


def split_cub_blob(blob: str, freq: Counter):
    """CUB captions in this dump are concatenated without separators, e.g.
    '... medium beakthis bird has ...'. Split on '.', then split fused tokens like
    'beakthis' -> 'beak' | 'this' whenever both halves are frequent words."""
    out = []
    for sent in blob.split("."):
        words = sent.strip().split()
        cur = []
        for w in words:
            split_done = False
            if len(w) > 3:
                for st in STARTERS:
                    pre = w[: -len(st)]
                    # fused token: prefix is a common word and the fused form is much rarer
                    if w.endswith(st) and len(pre) >= 3 and freq[pre] >= 20 and freq[w] * (1 if st in ("this", "these") else 10) < freq[pre]:
                        cur.append(w[: -len(st)])
                        out.append(" ".join(cur))
                        cur = [st]
                        split_done = True
                        break
            if not split_done:
                cur.append(w)
        if cur:
            out.append(" ".join(cur))
    out = [re.sub(r"\s+", " ", c).strip(" ,") for c in out]
    return [c for c in out if len(c.split()) >= 4]


def load_cub(raw_dir, res):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    assert files, f"no parquet files in {raw_dir}"
    texts = []
    for f in files:
        texts += [r["text"] for r in pq.read_table(f, columns=["text"]).to_pylist()]
    freq = Counter(w for t in texts for w in t.replace(".", " ").split())
    images, caps = [], []
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in tqdm(pf.iter_batches(batch_size=256, columns=["image", "text"]), desc=os.path.basename(f)):
            for r in batch.to_pylist():
                c = split_cub_blob(r["text"], freq)
                if not c:
                    continue
                images.append(center_crop_resize(Image.open(io.BytesIO(r["image"]["bytes"])), res))
                caps.append(c)
    return images, caps


# ----------------------------------------------------------------------------- generic CSV
def load_csv(csv_path, image_root, res):
    groups = OrderedDict()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = row["image_path"]
            if not os.path.isabs(p):
                p = os.path.join(image_root or os.path.dirname(csv_path), p)
            groups.setdefault(p, []).append(row["caption"].strip())
    images, caps = [], []
    for p, c in tqdm(groups.items(), desc="images"):
        try:
            images.append(center_crop_resize(Image.open(p), res))
            caps.append(c)
        except Exception as e:  # skip unreadable files
            print("skip", p, e)
    return images, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cub", "csv"], default="cub")
    ap.add_argument("--raw", default="data/raw/cub_cap", help="parquet dir (cub)")
    ap.add_argument("--csv", default=None, help="captions csv (csv source)")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--max-vocab", type=int, default=5000)
    args = ap.parse_args()

    if args.source == "cub":
        images, caps = load_cub(args.raw, args.res)
    else:
        images, caps = load_csv(args.csv, args.image_root, args.res)

    os.makedirs(args.out, exist_ok=True)
    arr = np.stack(images)
    np.save(os.path.join(args.out, f"images_{args.res}.npy"), arr)
    with open(os.path.join(args.out, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(caps, f)
    all_caps = [c for cs in caps for c in cs]
    tok = WordTokenizer.build(all_caps, min_freq=2, max_len=args.max_len, max_vocab=args.max_vocab)
    tok.save(os.path.join(args.out, "tokenizer.json"))
    lens = [len(basic_tokenize(c)) for c in all_caps]
    print(f"images: {arr.shape}  captions: {len(all_caps)}  vocab: {len(tok)}  "
          f"caption len mean/95%: {np.mean(lens):.1f}/{np.percentile(lens, 95):.0f}")
    print("examples:", caps[0][:4])

    eval_path = os.path.join(args.out, "eval_captions.json")
    if not os.path.exists(eval_path):
        rng = random.Random(0)
        pick = rng.sample(range(len(all_caps)), min(500, len(all_caps)))
        chosen = [all_caps[i] for i in pick]
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(chosen, f)
        print(f"froze {len(chosen)} eval captions at {eval_path}")
    else:
        print(f"kept existing eval captions at {eval_path}")


if __name__ == "__main__":
    main()
