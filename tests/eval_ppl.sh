#!/usr/bin/env bash
# Reproducible F32-vs-INT8 comparison: perplexity + throughput + file size.
# Prereqs: built binary, models/qwen05.nnlm, models/qwen05.i8.nnlm,
#          python3 with datasets+transformers (only for make_ids.py).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f tests/ppl_ids.txt ]; then
    python3 tests/make_ids.py --max-tokens 2048
fi

echo "=== perplexity (wikitext-2 test, first 2048 tokens) ==="
echo -n "f32 : "; ./build/nanollm -m models/qwen05.nnlm    --ppl tests/ppl_ids.txt 2>/dev/null
echo -n "int8: "; ./build/nanollm -m models/qwen05.i8.nnlm --ppl tests/ppl_ids.txt 2>/dev/null

echo "=== throughput (--bench: 128-token prefill + 32 decode steps) ==="
echo -n "f32 : "; ./build/nanollm -m models/qwen05.nnlm    --bench 2>/dev/null
echo -n "int8: "; ./build/nanollm -m models/qwen05.i8.nnlm --bench 2>/dev/null

echo "=== model files ==="
ls -la models/qwen05.nnlm models/qwen05.i8.nnlm | awk '{print $9, $5}'
