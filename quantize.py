#!/usr/bin/env python3
"""Quantize a nanollm .nnlm model to INT8 weights (per-output-channel,
symmetric, round-to-nearest) and write a new .nnlm.

Every 2D matmul weight (q/k/v/o projections and the SwiGLU gate/up/down)
becomes int8 with a companion f32 scale tensor "<name>.s" (one scale per
output row). Embeddings, norms, biases, and the (tied) lm_head stay f32:
they arelookup tables or tiny, and quantizing embeddings hurts perplexity
disproportionally.

Runtime cost model: decode is memory-bandwidth-bound, so halving weight
bytes is the win; the dequantization folds into the matmul epilogue.

Usage:
  python3 quantize.py --in models/qwen05.nnlm --out models/qwen05.i8.nnlm
"""

import argparse
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

QUANT_PATTERN = re.compile(r"^l\d+\.(wq|wk|wv|wo|wg|wu|wd)\.w$")


def read_nnlm(path: Path):
    data = path.read_bytes()  # ~2 GB peak; fine on a laptop
    if data[:8] != b"nanollm1":
        sys.exit("not a .nnlm file")
    off = 8
    (ver,) = struct.unpack_from("<I", data, off); off += 4
    if ver != 1:
        sys.exit("unsupported version")
    cfg = struct.unpack_from("<7i", data, off); off += 28
    rope, eps = struct.unpack_from("<2f", data, off); off += 8
    max_seq, tie = struct.unpack_from("<2i", data, off); off += 8
    (dtype0,) = struct.unpack_from("<I", data, off); off += 4
    off += 8  # data_offset (unused; absolute offsets below)
    (n_tensors,) = struct.unpack_from("<I", data, off); off += 4
    tensors = []  # (name, dtype, dims, abs_off, nbytes)
    for _ in range(n_tensors):
        (nlen,) = struct.unpack_from("<H", data, off); off += 2
        name = data[off : off + nlen].decode(); off += nlen
        dtype, ndims = struct.unpack_from("<2I", data, off); off += 8
        dims = struct.unpack_from(f"<{ndims}Q", data, off); off += 8 * ndims
        (nbytes,) = struct.unpack_from("<Q", data, off); off += 8
        (abs_off,) = struct.unpack_from("<Q", data, off); off += 8
        tensors.append((name, dtype, dims, abs_off, nbytes))
    cfg = dict(zip(("n_layers", "hidden", "n_heads", "n_kv", "head_dim", "ffn", "vocab"), cfg))
    cfg.update(rope_theta=rope, rms_eps=eps, max_seq=max_seq, tie=tie)
    return cfg, tensors, data


def write_nnlm(path: Path, cfg, out_tensors):
    header = bytearray()
    header += b"nanollm1"
    header += struct.pack("<I", 1)
    for k in ("n_layers", "hidden", "n_heads", "n_kv", "head_dim", "ffn", "vocab"):
        header += struct.pack("<i", cfg[k])
    header += struct.pack("<f", cfg["rope_theta"])
    header += struct.pack("<f", cfg["rms_eps"])
    header += struct.pack("<i", cfg["max_seq"])
    header += struct.pack("<i", cfg["tie"])
    header += struct.pack("<I", 0)  # dtype: mixed, per-tensor below
    header += struct.pack("<Q", 0)  # data_offset placeholder
    header += struct.pack("<I", len(out_tensors))

    dir_size = sum(2 + len(n.encode()) + 4 + 4 + 8 * len(d) + 8 + 8 for n, _, d, _ in out_tensors)
    data_offset = len(header) + dir_size

    blob = bytearray()
    entries = []
    pos = data_offset
    for name, arr_dtype, dims, payload in out_tensors:
        aligned = (pos + 15) & ~15
        blob += b"\x00" * (aligned - pos)
        entries.append((name, arr_dtype, dims, aligned, len(payload)))
        blob += payload
        pos = aligned + len(payload)

    for name, arr_dtype, dims, off, nbytes in entries:
        nb = name.encode()
        header += struct.pack("<H", len(nb)) + nb
        header += struct.pack("<I", arr_dtype)
        header += struct.pack("<I", len(dims))
        for d in dims:
            header += struct.pack("<Q", int(d))
        header += struct.pack("<Q", nbytes)
        header += struct.pack("<Q", off)
    DOFF = 8 + 4 + 7 * 4 + 4 * 5
    header[DOFF : DOFF + 8] = struct.pack("<Q", data_offset)

    path.write_bytes(bytes(header) + bytes(blob))
    return pos  # total size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg, tensors, data = read_nnlm(Path(args.inp))
    out_tensors = []
    n_quant = 0
    for name, dtype, dims, off, nbytes in tensors:
        if dtype != 0:
            sys.exit(f"input tensor {name} is already quantized")
        w = np.frombuffer(data, dtype="<f4", count=int(np.prod(dims)), offset=off)
        w = w.reshape(dims)
        if QUANT_PATTERN.match(name) and w.ndim == 2:
            scale = np.max(np.abs(w), axis=1) / 127.0
            scale = np.where(scale == 0, 1e-12, scale).astype("<f4")
            q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
            out_tensors.append((name, 1, q.shape, q.tobytes()))
            out_tensors.append((name + ".s", 0, (q.shape[0],), scale.tobytes()))
            n_quant += 1
        else:
            out_tensors.append((name, 0, dims, w.tobytes()))

    out_path = Path(args.out)
    total = write_nnlm(out_path, cfg, out_tensors)
    print(f"quantized {n_quant} weight tensors (INT8 per-channel symmetric)")
    print(f"wrote {out_path} ({total / 1e9:.2f} GB, {len(out_tensors)} tensors)")

    # tokenizer sidecars share the basename
    src = Path(args.inp)
    for ext in (".tok.tsv", ".merges"):
        if (src.parent / (src.stem + ext)).exists():
            shutil.copy(src.parent / (src.stem + ext), out_path.parent / (out_path.stem + ext))
            print(f"copied {out_path.stem}{ext}")


if __name__ == "__main__":
    main()
