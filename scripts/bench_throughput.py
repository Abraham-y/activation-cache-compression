"""Loader throughput benchmark with on-disk provenance, so the paper's
throughput table has an artifact behind it. prefetch=0 so the
disk/decode/pipeline split has clean single-thread semantics.

Usage: PYTHONPATH=src python3 scripts/bench_throughput.py --root runs_m1
Writes <root>/eval/throughput_bench.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.store import BatchLoader, Store  # noqa: E402
from acc.utils import write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--batches", type=int, default=256)
    ap.add_argument("--codecs", nargs="+", default=[
        "fp32", "fp16", "int8", "rot-int8", "int6", "rot-int6", "int4",
        "rot-int4", "int4-o16", "pca256"])
    args = ap.parse_args()

    results = {"note": "prefetch=0 (serial) for clean split timings; "
                       "warm page cache; single M5 laptop"}
    for c in args.codecs:
        name = "master" if c == "fp32" else c
        d = Path(args.root) / "cache" / f"{name}-train"
        if not (d / "manifest.json").exists():
            continue
        loader = BatchLoader(Store(d), 4096, seed=0)
        for i, _ in enumerate(loader.batches(epochs=1, prefetch=0)):
            if i >= args.batches - 1:
                break
        results[c] = loader.stats()
        s = results[c]
        print(f"{c:<10} disk {s['read_disk_MB_per_s']:>6.0f} MB/s  "
              f"decode {s['decode_MB_per_s']:>6.0f} MB/s  "
              f"pipeline {s['read_tokens_per_s']:>7.0f} tok/s")
    out = Path(args.root) / "eval" / "throughput_bench.json"
    write_json(out, results)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
