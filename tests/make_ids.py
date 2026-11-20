#!/usr/bin/env python3
"""Tokenize a wikitext-2 test slice with the HF tokenizer for --ppl.

Writes comma-separated token ids (default: first 2048 tokens of the joined
test split) so f32 and int8 runs score the exact same sequence.
"""

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="tests/ppl_ids.txt")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "".join(ds["text"])[: args.max_tokens * 4 + 4096]
    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    ids = tok(text, add_special_tokens=False, return_attention_mask=False).input_ids
    ids = ids[: args.max_tokens]
    with open(args.out, "w") as f:
        f.write(",".join(map(str, ids)))
    print(f"wrote {len(ids)} token ids to {args.out}")


if __name__ == "__main__":
    main()
