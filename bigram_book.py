"""
Tokenizer + bigram model on one book.

Usage:
    python bigram_book.py book.txt
    python bigram_book.py book.txt --prompt "the old man" --length 40
"""
import argparse
import math
import random
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")


# ---------- 1. Tokenizer ----------
class Tokenizer:
    """Word-level tokenizer: lowercases, splits words and punctuation."""

    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.stoi, self.itos = {}, []

    @staticmethod
    def split(text):
        return TOKEN_RE.findall(text.lower())

    def fit(self, text):
        counts = Counter(self.split(text))
        self.itos = ["<unk>"] + [t for t, c in counts.most_common() if c >= self.min_freq]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        return self

    def encode(self, text):
        return [self.stoi.get(t, 0) for t in self.split(text)]

    def decode(self, ids):
        out = " ".join(self.itos[i] for i in ids)
        return re.sub(r"\s+([.,!?;:])", r"\1", out)


# ---------- 2. Bigram model ----------
class BigramModel:
    def __init__(self, vocab_size, smoothing=0.1):
        self.V = vocab_size
        self.k = smoothing
        self.counts = defaultdict(Counter)

    def fit(self, ids):
        for a, b in zip(ids, ids[1:]):
            self.counts[a][b] += 1
        return self

    def prob(self, prev, nxt):
        c = self.counts[prev]
        return (c[nxt] + self.k) / (sum(c.values()) + self.k * self.V)

    def top_next(self, prev, n=5):
        return [(t, c) for t, c in self.counts[prev].most_common(n)]

    def sample_next(self, prev):
        c = self.counts[prev]
        if not c:
            return random.randrange(1, self.V)
        return random.choices(list(c), weights=list(c.values()))[0]

    def generate(self, start, length=30):
        ids = [start]
        for _ in range(length):
            ids.append(self.sample_next(ids[-1]))
        return ids

    def perplexity(self, ids):
        nll = -sum(math.log(self.prob(a, b)) for a, b in zip(ids, ids[1:]))
        return math.exp(nll / max(1, len(ids) - 1))


# ---------- 3. Run it ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("--prompt", default="the")
    ap.add_argument("--length", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    text = open(args.book, encoding="utf-8", errors="ignore").read()
    tok = Tokenizer().fit(text)
    ids = tok.encode(text)

    split = int(len(ids) * 0.9)  # hold out the last 10% to measure perplexity
    model = BigramModel(len(tok.itos)).fit(ids[:split])

    print(f"tokens: {len(ids):,} | vocab: {len(tok.itos):,}")
    print(f"held-out perplexity: {model.perplexity(ids[split:]):.1f}")

    prompt_ids = tok.encode(args.prompt)
    last = prompt_ids[-1]
    print(f"\nafter '{tok.itos[last]}', most likely next tokens:")
    for t, c in model.top_next(last):
        print(f"  {tok.itos[t]!r:12} {c}")

    out = prompt_ids[:-1] + model.generate(last, args.length)
    print("\ngenerated:\n" + tok.decode(out))


if __name__ == "__main__":
    main()
