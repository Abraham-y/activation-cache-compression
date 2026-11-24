"""Cache the SAME token set with a lower-precision model FORWARD.

The §7 claim is that the construct-validity failure is not about storage:
any data-side choice made without re-measuring is exposed to it. Storage
precision is the instance the paper measures; this script produces the
second instance, which is categorically different — it changes how the
cache is COMPUTED, not how it is written. No storage codec can recover
it, and the frozen judge cannot see it.

Some pipelines compute activations with a bf16 forward for speed, so
this is a choice real caches make. Everything else is held fixed: same
token file, same order, same layer, same shard boundaries, fp32 storage.
The only difference from the master is the dtype of the forward pass.

Output store is a plain fp32 store, so every existing CLI command treats
it as a codec named `fwdbf16` with no special-casing:

    python3 -m acc.cli --root runs_m1 --preset m1 train --codec fwdbf16 --seed 0

Usage: PYTHONPATH=src python3 scripts/forward_precision_cache.py \
           --root runs_m1 --dtype bf16 --name fwdbf16
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.cli import PRESETS, cache_dir, paths  # noqa: E402
from acc.model import cache_activations  # noqa: E402
from acc.utils import get_device, read_json, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--preset", default="m1", choices=list(PRESETS))
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--name", default="fwdbf16")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    p = PRESETS[args.preset]
    device = args.device or get_device()
    tokens_meta = read_json(paths(args.root)["tokens"] / "meta.json")

    master = read_json(cache_dir(args.root, "fp32", "train") / "manifest.json")
    out = {}
    for split in ("train", "eval"):
        d = cache_dir(args.root, args.name, split)
        if (d / "manifest.json").exists():
            print(f"skip {args.name}-{split}: manifest exists")
            out[split] = read_json(d / "manifest.json")
            continue
        t0 = time.perf_counter()
        m = cache_activations(
            paths(args.root)["tokens"] / f"tokens_{split}.npy", d,
            layer=p["layer"], batch_rows=p["batch_rows"],
            shard_tokens=p["shard_tokens"], device=device,
            model_name=p.get("model", "gpt2"),
            prepend_id=tokens_meta.get("eot_id", 50256),
            model_dtype=args.dtype,
        )
        print(f"cached {args.name}-{split}: {m['n_tokens']} tokens, "
              f"{m['throughput']['tokens_per_s']:.0f} tok/s, "
              f"{time.perf_counter()-t0:.0f}s")
        out[split] = m

    # Shard-for-shard alignment with the master is what makes every
    # downstream correlation valid (identical token positions). It is not
    # an assumption here — the same token file, shard size and fill order
    # produce it — but it is cheap to assert, so assert it.
    tr = out["train"]
    assert tr["n_tokens"] == master["n_tokens"], (
        f"token count differs from master: {tr['n_tokens']} vs "
        f"{master['n_tokens']} — comparisons would not be on identical "
        f"positions")
    assert len(tr["shards"]) == len(master["shards"]), "shard count differs"
    for a, b in zip(tr["shards"], master["shards"]):
        assert a["n_tokens"] == b["n_tokens"], "shard boundaries differ"
    print(f"alignment OK: {len(tr['shards'])} shards, "
          f"{tr['n_tokens']} tokens, identical boundaries to master")

    write_json(paths(args.root)["eval"] / f"forward_precision_{args.name}.json",
               {"name": args.name, "forward_dtype": args.dtype,
                "storage": "fp32 (identical to master)",
                "layer": p["layer"], "model": p.get("model", "gpt2"),
                "device": device,
                "aligned_with_master": True,
                "n_tokens_train": tr["n_tokens"],
                "n_shards_train": len(tr["shards"]),
                "cache_tokens_per_s": {
                    s: out[s]["throughput"]["tokens_per_s"] for s in out},
                "master_cache_tokens_per_s":
                    master.get("throughput", {}).get("tokens_per_s")})


if __name__ == "__main__":
    main()
