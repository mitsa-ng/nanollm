# nanollm

A minimal, dependency-free LLM inference runtime written from scratch in
C++17. It runs Qwen2 / Qwen2.5 dense checkpoints end to end — tokenizer,
weight loading, transformer forward pass, sampling — in a single ~1k-line
translation unit.

The point of this project is to demonstrate **understanding of every layer of
an on-device LLM inference stack**, not to wrap one: the attention path, the
KV cache layout, how a model graph maps onto matrix multiplications, and what
the tokenizer actually does.

## What is implemented

| Component | Notes |
|---|---|
| **Tokenizer** | GPT-2 byte-level BPE (Qwen vocab). Pre-tokenizer approximates Qwen's tiktoken regex; exact for ASCII, best-effort for non-ASCII symbol blocks. |
| **Weight format** | Custom `.nnlm`: fixed little-endian header + per-tensor directory, loaded via `mmap`. `export.py` converts HF safetensors (BF16→F32) using only numpy — safetensors is parsed manually, no torch. |
| **Attention** | GQA (grouped-query, e.g. 14 query heads : 2 KV heads), NeoX-style RoPE (`rotate_half` convention, matches HuggingFace exactly), causal masking during prefill. |
| **KV cache** | Explicit preallocated `[layer][pos][kv_head][head_dim]` cache. Prefill fills it; decode appends. The interactive chat keeps the cache across turns. |
| **Prefill vs decode** | One `forward()` path for both; prefill batches all prompt tokens, decode steps one token at a time against the cache. |
| **MLP / norms** | SwiGLU, RMSNorm (pre-norm), tied embeddings (0.5B) or separate `lm_head`. |
| **Sampling** | Greedy, temperature, top-p. |
| **Matmul** | Row-parallel across hardware threads; deterministic regardless of thread count. |

## Build

```bash
make            # clang++/g++ -O3 -std=c++17, no dependencies
```

## Get a model

```bash
pip install numpy huggingface_hub
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct

python3 export.py --hf-dir models/Qwen2.5-0.5B-Instruct --out models/qwen05.nnlm
```

`export.py` writes `qwen05.nnlm` (weights) plus `qwen05.tok.tsv` /
`qwen05.merges` (tokenizer). `--max-seq N` bounds KV-cache capacity.

## Run

```bash
# one-shot chat
./build/nanollm -m models/qwen05.nnlm -p "Explain KV caching in two sentences."

# interactive multi-turn chat (KV cache persists across turns)
./build/nanollm -m models/qwen05.nnlm

# raw completion, no chat template
./build/nanollm -m models/qwen05.nnlm --raw -p "Once upon a time" --temp 0.8

# prefill throughput / decode latency
./build/nanollm -m models/qwen05.nnlm --bench
```

## Correctness: golden test against HuggingFace

`tests/golden.sh` runs the same prompt through HF transformers (F32) and
through nanollm (`--forward` dumps the full last-position logits), then
compares:

```bash
pip install torch transformers
make test
```

Pass criteria: identical argmax token and `max|Δ logit| < 0.05`.

## Benchmarks

Qwen2.5-0.5B-Instruct, Apple M2 (4 performance cores), `tests/eval_ppl.sh`
reproduces everything:

| | F32 | INT8 (per-channel, weight-only) |
|---|---|---|
| model file | 1.98 GB | 0.90 GB (−55%) |
| prefill | 303 ms/tok | 27.9 ms/tok (**10.9×**) |
| decode | 9.5 tok/s | 23.0 tok/s (**2.4×**) |
| perplexity (wikitext-2 test, 2048 tok) | 10.34 | 10.30 (Δ 0.4%) |

Correctness gates: golden logits test vs HuggingFace F32 passes with
`max|Δ| = 6.9e-5` and identical argmax (F32 path, unchanged after the INT8
work).

Performance notes, in the order they were fixed:

1. Naive single-threaded decode: 2.3 tok/s. Decode matmuls are `M=1`, so
   row-parallel threading had nothing to parallelize.
2. Splitting the matmul loop over output columns (`N`) for `M ≤ 2`: 4.9 tok/s.
3. Lowering the thread-spawn threshold so mid-size projections also
   parallelize: 9.4 tok/s.
4. INT8 weights + NEON dot kernel (16 int8 → 4×float32x4 widening, FMA):
   23.0 tok/s. On the scalar path INT8 was *not* faster than F32 — the
   element-wise int8→float conversion defeats autovectorization; the
   explicit widening kernel is what recovers the bandwidth win. Prefill
   gains more (10.9×) because its arithmetic intensity amortizes weight
   traffic.

## Design notes

- **Why a custom weight format?** mmap + a flat directory keeps the runtime
  dependency-free and makes the memory layout explicit: one file, absolute
  offsets, 16-byte aligned tensors, zero copies at load time.
- **RoPE convention matters.** HF applies RoPE with the *rotate-half*
  (NeoX) convention — first/second half of the head dimension pair up. Mixing
  this up with the interleaved (GPT-J) convention produces plausible-looking
  but wrong text; the golden test exists to catch exactly this class of bug.
- **GQA**: query head `h` reads KV head `h / (n_heads / n_kv_heads)`. With
  14:2 grouping the KV cache holds 7× less K/V than a full multi-head cache.
- **Deterministic threading**: each output row of a matmul is computed by
  exactly one thread, so results never depend on scheduling.
- **Known limitation**: the pre-tokenizer approximates Unicode property
  classes (`\p{L}`, `\p{N}`) with block-range tests. ASCII/English is exact;
  CJK text tokenizes well but not bit-identically to tiktoken.

## Roadmap

- [x] **INT8 quantization** (P2): per-channel symmetric weight quantization,
      NEON widening kernel, perplexity + latency deltas (table above).
- [ ] BF16 weight storage (halves mmap footprint vs F32).
- [ ] Integer-only NEON path (`sdot`) with dynamic activation quantization.
- [ ] Batched prefill attention with tiled matmul instead of per-token loops.
- [ ] Sliding-window / longer contexts with a paged KV cache.

## License

MIT
