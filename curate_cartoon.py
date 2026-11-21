"""Captions for the Google Cartoon Set 10k release.

Each image already has an attribute CSV. This keeps the attributes that are
visible at 64x64 and writes one caption per image:

  hair color, eye color, skin tone, glasses, beard

Hair and eye names were read off the rendered pixels. The 11 face colors are
bucketed into four skin tones because neighboring shades are not separable
after the 64x64 resize. Hairstyle and eyebrow indices are left out: they are
artwork ids, not words, and most of them occur only a handful of times.
"""
import csv
import json
import os

ROOT = "data/raw/cartoonset10k/cartoonset10k"
OUT_CSV = "data/raw/cartoon_captions.csv"
PREVIEW_PATH = "data/raw/cartoon_preview_prompts.json"

HAIR = {
    0: "platinum",
    1: "gold",
    2: "orange",
    3: "red",
    4: "honey",
    5: "brown",
    6: "darkbrown",
    7: "black",
    8: "gray",
    9: "white",
}
EYE = {0: "brown", 1: "blue", 2: "green", 3: "gray", 4: "black"}
# Measured center-face RGB, dark to pale. Buckets: 0-1, 2-3, 4-6, 7-10.
SKIN = {
    0: "dark", 1: "dark",
    2: "brown", 3: "brown",
    4: "tan", 5: "tan", 6: "tan",
    7: "fair", 8: "fair", 9: "fair", 10: "fair",
}
NO_GLASSES = 11
NO_BEARD = 14


def read_attrs(path):
    attrs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(",")]
            if len(parts) >= 2 and parts[0]:
                attrs[parts[0]] = int(parts[1])
    return attrs


def caption(attrs):
    glasses = "no glasses" if attrs["glasses"] == NO_GLASSES else "glasses"
    beard = "no beard" if attrs["facial_hair"] == NO_BEARD else "beard"
    return (
        f"{HAIR[attrs['hair_color']]} hair, "
        f"{EYE[attrs['eye_color']]} eyes, "
        f"{SKIN[attrs['face_color']]} skin, "
        f"{glasses}, {beard}"
    )


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    stems = sorted(fn[:-4] for fn in os.listdir(ROOT) if fn.endswith(".png"))
    rows = []
    for stem in stems:
        png = os.path.join(ROOT, stem + ".png")
        attrs = read_attrs(os.path.join(ROOT, stem + ".csv"))
        rows.append((os.path.abspath(png), caption(attrs), attrs["hair_color"]))
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "caption"])
        for path, text, _ in rows:
            w.writerow([path, text])
    # One prompt per hair color, first image of that color, so the preview grid
    # shows whether color follows the caption.
    seen = {}
    for _, text, hair in rows:
        seen.setdefault(hair, text)
    prompts = [seen[i] for i in range(10) if i in seen][:8]
    with open(PREVIEW_PATH, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2)
    print(f"images {len(rows)}")
    print("preview:")
    for p in prompts:
        print(" ", p)


if __name__ == "__main__":
    main()
