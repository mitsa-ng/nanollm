#!/usr/bin/env python3
"""Export a HuggingFace Qwen2/Qwen2.5 checkpoint to nanollm's .nnlm format.

Dependency: numpy only. Safetensors are parsed manually (the format is an
8-byte header length, a JSON directory, then raw tensor bytes) so we do not
need torch. BF16 weights are widened to little-endian f32.

Outputs (given --out models/qwen05.nnlm):
  models/qwen05.nnlm      weights + directory (see src/nanollm.cpp for layout)
  models/qwen05.tok.tsv   BPE vocab as: id<TAB>hex(utf8 byte-uni token)
  models/qwen05.merges    BPE merges as: hex(a) hex(b), first line = count
"""

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np


def load_safetensors(path: Path) -> dict:
    tensors = {}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        (header_len,) = struct.unpack("<Q", mm[:8])
        header = json.loads(mm[8 : 8 + header_len])
        base = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dt = info["dtype"]
            start, end = info["data_offsets"]
            raw = mm[base + start : base + end]
            if dt == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                arr = (u16 << 16).view(np.float32)
            elif dt == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif dt == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            else:
                sys.exit(f"unsupported dtype {dt} for {name}")
            arr = arr.reshape(info["shape"])  # frombuffer is flat; restore real shape
            tensors[name] = np.ascontiguousarray(arr, dtype="<f4")
        mm.close()
    return tensors


def bf16_to_f32_dir(path: Path) -> dict:
    return load_safetensors(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help="HF snapshot dir (config.json + *.safetensors)")
    ap.add_argument("--out", required=True, help="output .nnlm path (prefix for sidecars)")
    ap.add_argument("--max-seq", type=int, default=None, help="KV cache capacity (default min(max_pos, 4096))")
    args = ap.parse_args()

    hf = Path(args.hf_dir)
    cfg = json.loads((hf / "config.json").read_text())

    n_layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim", hidden // n_heads)
    ffn = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rms_eps = float(cfg.get("rms_norm_eps", 1e-6))
    max_seq = args.max_seq or min(cfg.get("max_position_embeddings", 4096), 4096)

    shards = sorted(hf.glob("*.safetensors"))
    if not shards:
        sys.exit(f"no safetensors found in {hf}")
    tensors: dict = {}
    for shard in shards:
        tensors.update(load_safetensors(shard))

    # collect (short_name, array) in stable order
    out: list = []

    def add(hf_name, short):
        if hf_name in tensors:
            out.append((short, tensors[hf_name]))

    add("model.embed_tokens.weight", "embed")
    for i in range(n_layers):
        p = f"model.layers.{i}."
        add(p + "input_layernorm.weight", f"l{i}.ln1.w")
        add(p + "self_attn.q_proj.weight", f"l{i}.wq.w")
        add(p + "self_attn.q_proj.bias", f"l{i}.wq.b")
        add(p + "self_attn.k_proj.weight", f"l{i}.wk.w")
        add(p + "self_attn.k_proj.bias", f"l{i}.wk.b")
        add(p + "self_attn.v_proj.weight", f"l{i}.wv.w")
        add(p + "self_attn.v_proj.bias", f"l{i}.wv.b")
        add(p + "self_attn.o_proj.weight", f"l{i}.wo.w")
        add(p + "post_attention_layernorm.weight", f"l{i}.ln2.w")
        add(p + "mlp.gate_proj.weight", f"l{i}.wg.w")
        add(p + "mlp.up_proj.weight", f"l{i}.wu.w")
        add(p + "mlp.down_proj.weight", f"l{i}.wd.w")
    add("model.norm.weight", "norm.w")
    add("lm_head.weight", "lm_head")
    tie = 0 if any(s == "lm_head" for s, _ in out) else 1

    # ---- write .nnlm ----
    header = bytearray()
    header += b"nanollm1"
    header += struct.pack("<I", 1)  # version
    for v in (n_layers, hidden, n_heads, n_kv, head_dim, ffn, vocab):
        header += struct.pack("<i", v)
    header += struct.pack("<f", rope_theta)
    header += struct.pack("<f", rms_eps)
    header += struct.pack("<i", max_seq)
    header += struct.pack("<i", tie)
    header += struct.pack("<I", 0)  # dtype = f32
    header += struct.pack("<Q", 0)  # data_offset placeholder (abs offsets used)
    header += struct.pack("<I", len(out))

    dir_size = 0
    for name, arr in out:
        dir_size += 2 + len(name.encode()) + 4 + 4 + 8 * arr.ndim + 8 + 8
    data_offset = len(header) + dir_size

    blob = bytearray()
    offsets = []
    pos = data_offset
    for name, arr in out:
        aligned = (pos + 15) & ~15
        blob += b"\x00" * (aligned - pos)
        offsets.append(aligned)
        blob += arr.tobytes(order="C")
        pos = aligned + arr.nbytes

    for (name, arr), off in zip(out, offsets):
        nb = name.encode()
        header += struct.pack("<H", len(nb)) + nb
        header += struct.pack("<I", 0)  # f32
        header += struct.pack("<I", arr.ndim)
        for d in arr.shape:
            header += struct.pack("<Q", int(d))
        header += struct.pack("<Q", arr.nbytes)
        header += struct.pack("<Q", off)
    DOFF = 8 + 4 + 7 * 4 + 4 * 5  # magic, version, 7 cfg ints, rope, eps, max_seq, tie, dtype
    assert DOFF == 60
    header[DOFF:DOFF + 8] = struct.pack("<Q", data_offset)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(header) + bytes(blob))
    total = sum(a.nbytes for _, a in out)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB, {len(out)} tensors, {total / 1e6:.0f}M params)")
    print(f"config: L={n_layers} H={hidden} heads={n_heads}/{n_kv} hd={head_dim} ffn={ffn} "
          f"vocab={vocab} rope={rope_theta} tie={tie} max_seq={max_seq}")

    # ---- tokenizer sidecars ----
    vocab_json = json.loads((hf / "vocab.json").read_text())
    with open(out_path.with_suffix(".tok.tsv"), "w") as f:
        f.write(f"{len(vocab_json)}\n")
        for tok, tid in vocab_json.items():
            f.write(f"{tid}\t{tok.encode('utf-8').hex()}\n")

    merges = [l for l in (hf / "merges.txt").read_text().splitlines()
              if l and not l.startswith("#version")]
    with open(out_path.with_suffix(".merges"), "w") as f:
        f.write(f"{len(merges)}\n")
        for line in merges:
            a, b = line.split(" ")
            f.write(f"{a.encode('utf-8').hex()} {b.encode('utf-8').hex()}\n")
    print(f"tokenizer: {len(vocab_json)} vocab entries, {len(merges)} merges")


if __name__ == "__main__":
    main()
