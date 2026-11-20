#!/usr/bin/env python3
"""Golden-test reference: last-position logits from HuggingFace transformers.

Requires: pip install torch transformers
Writes <out-dir>/tokens.txt (comma-separated ids of the prompt) and
<out-dir>/logits.txt (one float per line).
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    model = AutoModelForCausalLM.from_pretrained(args.hf_dir, torch_dtype=torch.float32)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1]  # last position

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokens.txt").write_text(",".join(str(i) for i in ids[0].tolist()))
    with open(out / "logits.txt", "w") as f:
        for v in logits.tolist():
            f.write(f"{v:.6f}\n")
    print(f"prompt tokens: {ids[0].tolist()}")
    print(f"argmax: {int(logits.argmax())} ({tok.decode([int(logits.argmax())])!r})")


if __name__ == "__main__":
    main()
