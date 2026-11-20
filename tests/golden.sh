#!/usr/bin/env bash
# Golden test: nanollm must reproduce HF reference last-position logits.
# Prereqs:
#   pip install numpy torch transformers
#   python3 export.py --hf-dir models/Qwen2.5-0.5B-Instruct --out models/qwen05.nnlm
set -euo pipefail
cd "$(dirname "$0")/.."

HF_DIR=${HF_DIR:-models/Qwen2.5-0.5B-Instruct}
NNLM=${NNLM:-models/qwen05.nnlm}
PROMPT=${GOLDEN_PROMPT:-"The capital of France is"}

python3 tests/ref_logits.py --hf-dir "$HF_DIR" --prompt "$PROMPT" --out-dir tests/golden
./build/nanollm -m "$NNLM" --forward "$(cat tests/golden/tokens.txt)" > tests/golden/out_logits.txt
python3 tests/compare.py tests/golden/logits.txt tests/golden/out_logits.txt
