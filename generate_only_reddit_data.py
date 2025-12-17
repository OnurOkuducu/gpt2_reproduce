#!/usr/bin/env python3
"""
Reddit Instruction Dataset Generator (Balanced)
----------------------------------------------

Generates subreddit-guessing instruction data ONLY.

Each example:
<|endoftext|><ins>{INSTR}<ins><ctx>{POST}<ctx> r/{SUB}   or IDK

Features:
- Balanced sampling across subreddits
- Streaming (low memory)
- FineWeb-style .bin shards
- Token-budget driven

Example:
python generate_reddit_data.py \
  --out_dir reddit_only \
  --max_tokens 500000000 \
  --idk_prob 0.1

Requirements:
pip install datasets==3.6.0 tiktoken tqdm numpy
"""

import os
import re
import argparse
import random
import numpy as np
from tqdm import tqdm
import tiktoken
from datasets import load_dataset

# ===============================
# Config
# ===============================
REDDIT_SUBS = [
    "programming", "science", "bestof",  "Games",
    "Fitness", "DIY", "personalfinance", "philosophy", "history",
]

SUBREDDIT_INSTRUCTIONS = [
    "Guess the subreddit of the following post. Say IDK if you are not sure.",
    "Which subreddit does this post belong to? Say IDK if you are not sure.",
]

# ===============================
# Tokenizer
# ===============================
base_enc = tiktoken.get_encoding("gpt2")
enc = tiktoken.Encoding(
    name="gpt2-with-ins-ctx",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={
        **base_enc._special_tokens,
        "<ins>": 50257,
        "<ctx>": 50258,
    },
)
EOT = enc._special_tokens["<|endoftext|>"]

# ===============================
# Helpers
# ===============================
def clean_text(t, max_len=800):
    if not t:
        return ""
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"&gt;|&lt;|&amp;", "", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip()
    if t.lower() in ("[deleted]", "[removed]"):
        return ""
    return t[:max_len]

def make_sample(text, subreddit, idk_prob):
    instr = random.choice(SUBREDDIT_INSTRUCTIONS)
    if random.random() < idk_prob:
        return f"<ins>{instr}<ins><ctx>{text}<ctx> IDK"
    return f"<ins>{instr}<ins><ctx>{text}<ctx> r/{subreddit}"

def tokenize(sample):
    return np.array([EOT] + enc.encode(sample, allowed_special={"<ins>", "<ctx>"}), dtype=np.uint16)

# ===============================
# Shard Writer
# ===============================
class ShardWriter:
    def __init__(self, out_dir, shard_size):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.buf = np.empty((shard_size,), dtype=np.uint16)
        self.count = 0
        self.idx = 0
        self.pbar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {self.idx}")

    def write(self):
        if self.count == 0:
            return
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20251217
        header[1] = 1
        header[2] = self.count

        split = "val" if self.idx == 0 else "train"
        fname = os.path.join(self.out_dir, f"reddit_{split}_{self.idx:06d}.bin")

        with open(fname, "wb") as f:
            f.write(header.tobytes())
            f.write(self.buf[:self.count].tobytes())

        print(f"[+] wrote {self.count:,} tokens -> {fname}")
        self.idx += 1
        self.count = 0
        self.pbar.close()
        self.pbar = tqdm(total=self.shard_size, unit="tokens", desc=f"Shard {self.idx}")

    def add(self, toks):
        n = len(toks)
        if self.count + n <= self.shard_size:
            self.buf[self.count:self.count+n] = toks
            self.count += n
            self.pbar.update(n)
        else:
            rem = self.shard_size - self.count
            if rem > 0:
                self.buf[self.count:self.count+rem] = toks[:rem]
                self.pbar.update(rem)
            self.write()
            leftover = n - rem
            if leftover > 0:
                self.buf[:leftover] = toks[rem:]
                self.count = leftover
                self.pbar.update(leftover)

# ===============================
# Balanced Reddit Stream
# ===============================
def reddit_stream(subs):
    streams = {
        s: iter(load_dataset(
            "HuggingFaceGECLM/REDDIT_comments",
            split=s,
            streaming=True,
        ))
        for s in subs
    }
    active = subs[:]

    while active:
        sub = random.choice(active)
        try:
            ex = next(streams[sub])
        except StopIteration:
            active.remove(sub)
            continue

        text = clean_text(ex.get("body", ""))
        if len(text) < 10:
            continue

        yield text, sub

# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--shard_size", type=int, default=10_000_000)
    parser.add_argument("--max_tokens", type=int, required=True)
    parser.add_argument("--idk_prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    writer = ShardWriter(args.out_dir, args.shard_size)

    total = 0
    for text, sub in reddit_stream(REDDIT_SUBS):
        toks = tokenize(make_sample(text, sub, args.idk_prob))
        writer.add(toks)
        total += len(toks)

        if total >= args.max_tokens:
            break

        if total % 5_000_000 < len(toks):
            print(f"[progress] {total:,} tokens")

    writer.write()
    print(f"[✓] Finished Reddit dataset: {total:,} tokens")

if __name__ == "__main__":
    main()
