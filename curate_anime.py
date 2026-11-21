"""Filter puruchinera/anime-faces-256 into an image_path,caption CSV.

Keeps single-face rows, strips character-name tags, and drops tags seen fewer than
20 times. Appearance tags (hair, eyes, expression, clothing, portrait, and the other
non-name tags) stay on the caption.

Optional tighter set for the teacher: --portrait-only keeps the portrait tag and
drops hand, lying, and profile rows; --min-character-count N keeps only identities
that occur at least N times and drops unnamed images. Names are still removed from
the caption.
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter

import pyarrow.parquet as pq
from tqdm import tqdm

# Frequent character names that are not written with a "(series)" suffix.
# Any other tag containing "(" and ")" is also treated as a character name.
CHARACTER_NAMES = {
    "frieren", "oyama mahiro", "asuka langley", "souryuu asuka langley", "makima",
    "arima kana", "nagisa kaworu", "makinami mari", "hoshino ai", "oyama mihari",
    "katsuragi misato", "yamada anna", "hoshino ruby", "ikari shinji", "hozuki kaede",
    "kochiya sanae", "ayanami rei", "denji", "hozuki momiji", "murosaki miyo", "reze",
    "kitagawa marin", "asaka karin", "kirigaya suguha", "makinami mari illustrious",
    "anis nikke", "higuchi madoka", "mejiro ardan", "itsumi erika", "megurine luka",
    "ookami mio", "fern", "hakos baelz", "jougasaki mika",
}
MULTI = {
    "2girls", "3girls", "4girls", "5girls", "6girls", "multiple girls",
    "2boys", "3boys", "4boys", "multiple boys", "multiple people",
}
POSE_DROP = {
    "hand up", "hand on own face", "hands on own face", "hand on face",
    "hands up", "lying", "profile",
}
# Same person written two ways. Parenthetical series suffixes are stripped separately.
ALIASES = {
    "souryuu asuka langley": "asuka langley",
    "makinami mari illustrious": "makinami mari",
    "anis nikke": "anis",
}


def is_character(tag):
    return tag in CHARACTER_NAMES or ("(" in tag and ")" in tag)


def is_multi(tags):
    s = set(tags)
    if s & MULTI:
        return True
    girl = bool(s & {"1girl", "girl"})
    boy = bool(s & {"1boy", "boy"})
    return girl and boy


def canon_name(tag):
    base = tag.split(" (")[0].strip()
    return ALIASES.get(base, base)


def split_tags(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def iter_tag_rows(paths):
    idx = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1024, columns=["tags"]):
            for text in batch.column("tags").to_pylist():
                yield idx, split_tags(text or "")
                idx += 1


def image_bytes(cell):
    if isinstance(cell, dict):
        return cell.get("bytes")
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="data/raw/anime-faces-256")
    ap.add_argument("--image-dir", default="data/raw/anime_images")
    ap.add_argument("--csv", default="data/raw/anime_captions.csv")
    ap.add_argument("--stats", default="data/raw/filter_stats.json")
    ap.add_argument("--min-tag-count", type=int, default=20)
    ap.add_argument("--portrait-only", action="store_true",
                    help="keep rows tagged portrait and drop hand, lying, and profile rows")
    ap.add_argument("--min-character-count", type=int, default=0,
                    help="keep only identities with at least this many images; 0 keeps every row")
    ap.add_argument("--csv-only", action="store_true",
                    help="write the CSV and point it at images already in --image-dir")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    assert paths, f"no parquet files in {args.parquet_dir}"

    kept_tags = {}
    kept_names = {}
    n_in = 0
    n_multi = 0
    n_pose = 0
    n_not_portrait = 0
    n_named = 0
    n_rare_character = 0
    name_hits = Counter()
    for idx, tags in iter_tag_rows(paths):
        n_in += 1
        if is_multi(tags):
            n_multi += 1
            continue
        cleaned = []
        names = []
        for t in tags:
            if is_character(t):
                name_hits[t] += 1
                names.append(canon_name(t))
            else:
                cleaned.append(t)
        if names:
            n_named += 1
        kept_tags[idx] = cleaned
        kept_names[idx] = names
    print(f"rows in {n_in}  multi-person dropped {n_multi}  single-face {len(kept_tags)}")

    # Count an identity on every single-face image, then drop. A character with
    # 2,000 images and 180 portraits still counts as a large sample.
    if args.min_character_count > 0:
        identity_count = Counter(n for names in kept_names.values() for n in set(names))
        frequent = {n for n, c in identity_count.items() if c >= args.min_character_count}
        for idx, names in list(kept_names.items()):
            if names and any(n in frequent for n in names):
                continue
            n_rare_character += 1
            del kept_tags[idx]
            del kept_names[idx]
        print(f"identities kept {len(frequent)} (count>={args.min_character_count})  "
              f"images dropped for rare or unnamed identity {n_rare_character}")
        print("kept identities:")
        for name, c in identity_count.most_common():
            if name not in frequent:
                break
            print(f"  {c:6}  {name}")

    if args.portrait_only:
        for idx, tags in list(kept_tags.items()):
            tagset = set(tags)
            if "portrait" not in tagset:
                n_not_portrait += 1
                del kept_tags[idx]
                del kept_names[idx]
            elif tagset & POSE_DROP:
                n_pose += 1
                del kept_tags[idx]
                del kept_names[idx]
        print(f"not portrait {n_not_portrait}  pose dropped {n_pose}  remaining {len(kept_tags)}")

    freq = Counter(t for tags in kept_tags.values() for t in tags)
    rare = {t for t, c in freq.items() if c < args.min_tag_count}
    final = {}
    empty = 0
    for idx, tags in kept_tags.items():
        tags = [t for t in tags if t not in rare]
        if not tags:
            empty += 1
            continue
        final[idx] = tags
    print(f"tags kept {sum(c >= args.min_tag_count for c in freq.values())}  "
          f"tags dropped for count<{args.min_tag_count} {len(rare)}")
    print(f"character-name tags removed {len(name_hits)}  images that had a name {n_named}")
    print(f"empty captions after filtering {empty}  images kept {len(final)}")
    print("most common removed names:")
    for tag, c in name_hits.most_common(15):
        print(f"  {c:6}  {tag}")

    # Safety: a kept frequent tag should not still be a listed character name.
    survivors = [t for t in freq if freq[t] >= args.min_tag_count and is_character(t)]
    if survivors:
        raise SystemExit(f"character names survived the filter: {survivors[:20]}")

    stats = {
        "images_before": n_in,
        "multi_person_dropped": n_multi,
        "empty_caption_dropped": empty,
        "images_after": len(final),
        "tags_before_frequency_filter": len(freq),
        "tags_kept": sum(c >= args.min_tag_count for c in freq.values()),
        "tags_dropped_rare": len(rare),
        "character_name_tags": len(name_hits),
        "images_with_a_character_name": n_named,
        "not_portrait_dropped": n_not_portrait,
        "pose_dropped": n_pose,
        "rare_or_unnamed_character_dropped": n_rare_character,
        "min_character_count": args.min_character_count,
        "portrait_only": args.portrait_only,
        "tags_per_caption_mean": (sum(len(t) for t in final.values()) / len(final)) if final else 0,
        "captions_per_image": 1,
        "min_tag_count": args.min_tag_count,
    }
    os.makedirs(os.path.dirname(args.stats) or ".", exist_ok=True)
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats))
    if args.stats_only or args.csv_only:
        if args.csv_only:
            os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
            missing = 0
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image_path", "caption"])
                writer.writeheader()
                for idx, tags in final.items():
                    name = f"{idx:05d}.jpg"
                    if not os.path.exists(os.path.join(args.image_dir, name)):
                        missing += 1
                        continue
                    writer.writerow({"image_path": name, "caption": ", ".join(tags)})
            print(f"wrote {len(final) - missing} rows to {args.csv}  missing images {missing}")
            if missing:
                raise SystemExit(f"{missing} filtered images are not in {args.image_dir}")
        return

    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    written = 0
    idx = 0
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "caption"])
        writer.writeheader()
        for path in paths:
            pf = pq.ParquetFile(path)
            for batch in tqdm(pf.iter_batches(batch_size=128, columns=["image"]), desc=os.path.basename(path)):
                for cell in batch.column("image").to_pylist():
                    tags = final.get(idx)
                    idx += 1
                    if tags is None:
                        continue
                    raw = image_bytes(cell)
                    if not raw:
                        print("skip empty image", idx - 1)
                        continue
                    name = f"{idx - 1:05d}.jpg"
                    out = os.path.join(args.image_dir, name)
                    with open(out, "wb") as imgf:
                        imgf.write(raw)
                    writer.writerow({"image_path": name, "caption": ", ".join(tags)})
                    written += 1
    print(f"wrote {written} images to {args.csv}")
    if written != len(final):
        raise SystemExit(f"wrote {written} images, expected {len(final)}")


if __name__ == "__main__":
    main()
