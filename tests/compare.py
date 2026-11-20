#!/usr/bin/env python3
"""Compare nanollm --forward logits against the HF reference."""

import sys

import numpy as np

ref = np.loadtxt(sys.argv[1])
got = np.loadtxt(sys.argv[2])
if ref.shape != got.shape:
    sys.exit(f"shape mismatch: ref {ref.shape} vs got {got.shape}")
diff = np.abs(ref - got)
print(f"n={len(ref)}  max|Δ|={diff.max():.6f}  mean|Δ|={diff.mean():.8f}")
print(f"argmax ref={ref.argmax()}  got={got.argmax()}  {'SAME' if ref.argmax() == got.argmax() else 'DIFF'}")
tol = 0.05
ok = ref.argmax() == got.argmax() and diff.max() < tol
print("PASS" if ok else f"FAIL (tolerance {tol})")
sys.exit(0 if ok else 1)
