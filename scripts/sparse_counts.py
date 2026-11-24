"""Perturbed-value COUNTS for the two sparse rungs.

The paper's knee is in how many stored values differ, not how far, so the
counts have to be artifact-backed rather than asserted. Derived from the
store manifests and the codecs' own fractions.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _per_shard_total(manifest, fraction):
    """SparseUlpCodec applies k = round(n_shard_scalars * fraction) PER
    SHARD, so the corpus total is the SUM of per-shard rounds, not a
    single round over the whole corpus. Over the 77 master-train shards
    that is 15,335, not the 15,360 the whole-corpus form gives."""
    d = manifest.get("d") or manifest["codec_state"]["d"]
    return int(sum(max(1, round(sh["n_tokens"] * d * fraction))
                   for sh in manifest["shards"]))


def main():
    snap = ROOT / "cache_manifests_snapshot"
    m = None
    for c in ("runs_m1_cache_master-train_manifest.json",
              "runs_m1_cache_fp32-train_manifest.json"):
        if (snap / c).exists():
            m = json.loads((snap / c).read_text())
            break
    if m is None:
        p = ROOT / "runs_m1" / "cache" / "master-train" / "manifest.json"
        m = json.loads(p.read_text())
    n_tok = m["n_tokens"]
    d = m.get("d") or m.get("d_model")
    n_scalars = n_tok * d
    out = {
        "n_tokens": n_tok, "d_model": d, "n_scalars": n_scalars,
        "arms": {
            "oneulp": {"fraction": None, "n_perturbed": 1},
            "sparseulp6": {"fraction": 1e-6,
                           "n_perturbed": _per_shard_total(m, 1e-6)},
        },
        "note": ("Counts of stored scalars altered by one ulp. The knee the "
                 "paper brackets is in this quantity, not in magnitude."),
    }
    dest = ROOT / "runs_m1" / "eval" / "sparse_counts.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    print(f"  corpus: {n_tok:,} tokens x {d} = {n_scalars:,} scalars")
    for k, v in out["arms"].items():
        print(f"  {k:12} perturbed {v['n_perturbed']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
