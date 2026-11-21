"""Word-level tokenizer built from the training captions (no pretrained vocab)."""
import json
import re
from collections import Counter

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")


def basic_tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


class WordTokenizer:
    def __init__(self, vocab, max_len=40):
        self.itos = list(vocab)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.max_len = max_len

    @classmethod
    def build(cls, captions, min_freq=2, max_len=40, max_vocab=8000):
        cnt = Counter()
        for c in captions:
            cnt.update(basic_tokenize(c))
        words = [w for w, f in cnt.most_common(max_vocab) if f >= min_freq]
        return cls(SPECIALS + words, max_len=max_len)

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str):
        """Returns a fixed-length list of ids: <bos> w1 .. wn <eos> <pad>..."""
        ids = [self.stoi.get(w, UNK) for w in basic_tokenize(text)][: self.max_len - 2]
        ids = [BOS] + ids + [EOS]
        return ids + [PAD] * (self.max_len - len(ids))

    def encode_batch(self, texts):
        import torch
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.itos, "max_len": self.max_len}, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["vocab"], d["max_len"])

    def to_dict(self):
        return {"vocab": self.itos, "max_len": self.max_len}

    @classmethod
    def from_dict(cls, d):
        return cls(d["vocab"], d["max_len"])
