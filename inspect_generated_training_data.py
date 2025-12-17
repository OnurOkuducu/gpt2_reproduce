#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tiktoken
import random
from typing import Optional, Tuple

def build_tokenizer(with_ins_ctx: bool = True):
    base = tiktoken.get_encoding("gpt2")
    if not with_ins_ctx:
        return base

    # Extend GPT-2 with your special tokens at 50257/50258
    custom_specials = {"<ins>": 50257, "<ctx>": 50258}
    enc = tiktoken.Encoding(
        name="gpt2-with-ins-ctx",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **custom_specials},
    )
    return enc

def read_bin(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        toks = np.frombuffer(f.read(), dtype=np.uint16)
    return header, toks

def decode_range(enc, toks: np.ndarray, start: int, length: int) -> str:
    end = min(len(toks), start + length)
    return enc.decode(toks[start:end].astype(np.int64).tolist())

def find_eot_indices(enc, toks: np.ndarray):
    # gpt2 EOT is 50256
    eot = enc.eot_token
    return np.where(toks == eot)[0]

def print_doc_samples(enc, toks: np.ndarray, max_docs: int = 5, max_chars: int = 1200):
    eot_positions = find_eot_indices(enc, toks)
    if len(eot_positions) == 0:
        print("[warn] No <|endoftext|> tokens found; printing raw start snippet instead.\n")
        text = enc.decode(toks[:5000].astype(np.int64).tolist())
        print(text[:max_chars])
        return

    # docs are segments after each EOT
    # doc i = (eot_positions[i] + 1) ... (eot_positions[i+1]) inclusive/exclusive
    doc_starts = (eot_positions + 1).tolist()
    doc_starts = [s for s in doc_starts if s < len(toks)]

    # Pair each start with next eot (or end of array)
    doc_ranges = []
    for i, s in enumerate(doc_starts):
        # end is next eot (inclusive boundary before it)
        nxt = eot_positions[i + 1] if (i + 1) < len(eot_positions) else len(toks)
        doc_ranges.append((s, int(nxt)))

    print(f"[+] Found ~{len(doc_ranges):,} documents (EOT-delimited). Showing up to {max_docs} docs:\n")

    # show first few docs
    for di, (s, e) in enumerate(doc_ranges[:max_docs]):
        doc_toks = toks[s:e]
        txt = enc.decode(doc_toks.astype(np.int64).tolist())
        print("=" * 90)
        print(f"[doc {di}] tokens={len(doc_toks):,} span=[{s}:{e}]")
        print(txt[:max_chars].rstrip())
        if len(txt) > max_chars:
            print("... [truncated]")
    print("=" * 90)

def main():
    ap = argparse.ArgumentParser(description="Inspect a GPT-2-style .bin shard (FineWeb-style header + uint16 tokens)")
    ap.add_argument("path", help="Path to .bin shard")
    ap.add_argument("--with_ins_ctx", action="store_true",
                    help="Use tokenizer extended with <ins>=50257 and <ctx>=50258 (recommended for your data)")
    ap.add_argument("--max_docs", type=int, default=5, help="How many EOT-delimited docs to print")
    ap.add_argument("--max_chars", type=int, default=1200, help="Max characters per printed doc/snippet")
    ap.add_argument("--snippets", type=int, default=3, help="How many random token-window snippets to print")
    ap.add_argument("--window", type=int, default=512, help="Token window size for random snippets")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no_docs", action="store_true", help="Skip EOT-delimited doc printing (just do snippets)")
    args = ap.parse_args()

    enc = build_tokenizer(with_ins_ctx=args.with_ins_ctx)

    header, toks = read_bin(args.path)
    magic, ver, n = int(header[0]), int(header[1]), int(header[2])

    print(f"[file] {args.path}")
    print(f"[hdr ] magic={magic} version={ver} header_token_count={n} actual_tokens={len(toks)}")

    if n != len(toks):
        print("[warn] header[2] does not match token array length (file may be truncated or header differs).")

    # Print some doc samples
    if not args.no_docs:
        print_doc_samples(enc, toks, max_docs=args.max_docs, max_chars=args.max_chars)

    # Random token-window snippets
    random.seed(args.seed)
    if len(toks) > args.window:
        print("\n[+] Random token-window snippets:\n")
        for i in range(args.snippets):
            start = random.randint(0, max(0, len(toks) - args.window))
            txt = decode_range(enc, toks, start, args.window)
            print("-" * 90)
            print(f"[snippet {i}] start={start} window={args.window}")
            print(txt[:args.max_chars].rstrip())
            if len(txt) > args.max_chars:
                print("... [truncated]")
        print("-" * 90)

    # Start/end quick look
    print("\n[+] Start of file (first 256 tokens):\n")
    print(decode_range(enc, toks, 0, 256)[:args.max_chars])

    print("\n[+] End of file (last 256 tokens):\n")
    start = max(0, len(toks) - 256)
    print(decode_range(enc, toks, start, 256)[:args.max_chars])

if __name__ == "__main__":
    main()
