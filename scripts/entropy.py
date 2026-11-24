"""Measure the lossless entropy stage: zstd over already-quantized payloads.

Quantized activations are low-entropy (near-Gaussian ints), so a fast
lossless pass can shrink stored bytes further with zero fidelity cost. This
measures actual ratios per codec on a real shard — a Pareto refinement for
the paper's bytes/token axis, decode-side cost included.

SCOPE: shard 0 of each store, not the whole store, so ratios and MB/s are
shard-level estimates. The appendix table states that scope (it reads the
shard size from the store manifest); do not quote these as whole-store
numbers.

Usage: PYTHONPATH=src python3 scripts/entropy.py --root runs_m1
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.store import Store  # noqa: E402
from acc.utils import write_json  # noqa: E402


def measure(store_dir, level=3):
    store = Store(store_dir)
    meta = store.shards[0]
    with np.load(Path(store_dir) / meta["file"]) as z:
        payload = {k: z[k] for k in z.files}
    n = meta["n_tokens"]
    raw = sum(a.nbytes for a in payload.values())
    cctx = zstandard.ZstdCompressor(level=level)
    dctx = zstandard.ZstdDecompressor()
    comp = 0
    t_c = t_d = 0.0
    for a in payload.values():
        buf = a.tobytes()
        t0 = time.perf_counter()
        c = cctx.compress(buf)
        t_c += time.perf_counter() - t0
        comp += len(c)
        t0 = time.perf_counter()
        dctx.decompress(c)
        t_d += time.perf_counter() - t0
    return {
        "raw_B_per_tok": raw / n,
        "zstd_B_per_tok": comp / n,
        "ratio": raw / comp,
        "compress_MB_per_s": raw / 1e6 / max(t_c, 1e-9),
        "decompress_MB_per_s": raw / 1e6 / max(t_d, 1e-9),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--codecs", nargs="+", default=[
        "fp16", "int8", "rot-int8", "int6", "rot-int6", "int4", "rot-int4",
        "int4-o16"])
    args = ap.parse_args()

    results = {}
    print(f"{'codec':<10} {'raw B/tok':>10} {'zstd B/tok':>11} {'ratio':>7} "
          f"{'dec MB/s':>9}")
    for c in args.codecs:
        d = Path(args.root) / "cache" / f"{c}-train"
        if not (d / "manifest.json").exists():
            continue
        r = measure(d, args.level)
        results[c] = r
        print(f"{c:<10} {r['raw_B_per_tok']:>10.1f} {r['zstd_B_per_tok']:>11.1f} "
              f"{r['ratio']:>6.2f}x {r['decompress_MB_per_s']:>9.0f}")
    write_json(Path(args.root) / "eval" / "entropy.json", results)


if __name__ == "__main__":
    main()
