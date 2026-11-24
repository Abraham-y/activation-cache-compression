"""Paired loader-throughput measurement: std-int4 against rot-int4, same
session, same settings (prefetch=0, 256 batches of 4096, warm page cache).
Supports the claim that the rotation slows the loader; the main
throughput table (throughput_bench.json) predates the std-int4 store.
Writes <root>/eval/throughput_std_vs_rot.json
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.store import BatchLoader, Store  # noqa: E402
from acc.utils import write_json  # noqa: E402

def run(root, codec, batches):
    loader = BatchLoader(Store(root / "cache" / f"{codec}-train"), 4096, seed=0)
    for i, _ in enumerate(loader.batches(epochs=1, prefetch=0)):
        if i >= batches - 1:
            break
    return loader.stats()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--batches", type=int, default=256)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    root = Path(args.root)
    out = {"note": "paired, same session, prefetch=0, warm page cache; " + args.note}
    for codec in ("rot-int4", "std-int4", "rot-int4", "std-int4"):   # ABAB to expose drift
        s = run(root, codec, args.batches)
        out.setdefault(codec, []).append(s)
        print(f"{codec:<9} decode {s['decode_MB_per_s']:>6.0f} MB/s  pipeline {s['read_tokens_per_s']:>7.0f} tok/s")
    out["means"] = {c: {k: sum(r[k] for r in out[c]) / len(out[c])
                        for k in ("decode_MB_per_s", "read_tokens_per_s")}
                    for c in ("rot-int4", "std-int4")}
    write_json(root / "eval" / "throughput_std_vs_rot.json", out)
if __name__ == "__main__":
    main()
