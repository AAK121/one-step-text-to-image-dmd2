"""Dataset of (image, random caption) pairs kept fully in memory."""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import WordTokenizer


class CaptionImageDataset(Dataset):
    def __init__(self, root="data/processed", res=256, hflip=True, drop_caption=0.1):
        self.images = np.load(os.path.join(root, f"images_{res}.npy"), mmap_mode="r")
        with open(os.path.join(root, "captions.json"), encoding="utf-8") as f:
            self.captions = json.load(f)
        self.tok = WordTokenizer.load(os.path.join(root, "tokenizer.json"))
        self.hflip = hflip
        self.drop_caption = drop_caption  # for classifier-free guidance training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.images[i])).permute(2, 0, 1).float() / 127.5 - 1.0
        if self.hflip and random.random() < 0.5:
            x = x.flip(-1)
        cap = "" if random.random() < self.drop_caption else random.choice(self.captions[i])
        return x, torch.tensor(self.tok.encode(cap), dtype=torch.long)

    def all_captions(self):
        return [c for cs in self.captions for c in cs]
