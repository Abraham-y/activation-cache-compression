"""Every number the paper quotes from the 3x fresh-text / replay /
equal-bytes experiment (scripts/run_fresh_b3x.sh, pre-registered R1-R3
and P checks in its header), computed in one place from the pair JSONs and
training logs, plus the post-hoc resolution check (run_r3_resolution.sh).

Usage: python scripts/fresh_summary.py --root runs_m1
Writes <root>/eval/fresh_summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

PAIRS = ((0, 1), (0, 2), (1, 2))
CROSS = [(a, b) for a in range(3) for b in range(3) if a != b]


def stat(v):
    v = [float(x) for x in v]
    return {"mean": float(np.mean(v)), "min": min(v), "max": max(v),
            "half_range": (max(v) - min(v)) / 2, "values": v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    args = ap.parse_args()
    ev = Path(args.root) / "eval"
    sd = Path(args.root) / "saes"
    J = lambda n: json.loads((ev / n).read_text())
    S = lambda n: J(n)["survived"]
    out = {}

    f_fr = stat([S(f"pair_fp32_freshb3x_s{a}__fp32_freshb3x_s{b}.json") for a, b in PAIRS])
    f_rp = stat([S(f"pair_fp32_b3x_s{a}__fp32_b3x_s{b}.json") for a, b in PAIRS])
    out["R1"] = {"floor_fresh": f_fr, "floor_replay": f_rp,
                 "fresh_minus_replay": f_fr["mean"] - f_rp["mean"]}

    sf = stat([S(f"pair_fp32_freshb3x_s{k}__std-int4_freshb3x_s{k}.json") for k in range(3)])
    sr = stat([S(f"pair_fp32_b3x_s{k}__std-int4_b3x_s{k}.json") for k in range(3)])
    f16 = stat([S(f"pair_fp32_b3x_s{k}__fp16_b3x_s{k}.json") for k in range(3)])
    out["R2"] = {
        "std_int4_fresh": sf, "gap_fresh": sf["mean"] - f_fr["mean"],
        "ratio_fresh": sf["mean"] / f_fr["mean"],
        "every_seed_above_floor_max": bool(sf["min"] > f_fr["max"]),
        "std_int4_replay": sr, "gap_replay": sr["mean"] - f_rp["mean"],
        "ratio_replay": sr["mean"] / f_rp["mean"],
        "fp16_replay": f16, "fp16_ratio_replay": f16["mean"] / f_rp["mean"],
        "fresh_minus_replay_std_int4": sf["mean"] - sr["mean"]}

    def cross(cand):
        v = [S(f"pair_fp32_freshb3x_s{a}__{cand}_s{b}.json") for a, b in CROSS]
        fvu = [J(f"pair_fp32_freshb3x_s{a}__{cand}_s{b}.json")["fvu_cand"] for a, b in CROSS]
        return stat(v), float(np.mean(fvu))

    x16, fvu16 = cross("fp16_b3x")
    x4, fvu4 = cross("std-int4_mbb3x")
    x32, _ = cross("fp32_b3x")
    i4 = stat([S(f"pair_fp32_freshb3x_s{a}__int4_b3x_s0.json") for a in (1, 2)])
    lay = J("fresh_store_layout.json")
    out["R3"] = {
        "fp16_replay_20M": x16, "std_int4_mb": x4,
        "x4_minus_x16": x4["mean"] - x16["mean"],
        "ranges_disjoint": bool(x4["min"] > x16["max"] or x16["min"] > x4["max"]),
        "x16_over_floor": x16["mean"] / f_fr["mean"],
        "x4_over_floor": x4["mean"] / f_fr["mean"],
        "fvu_fp16": fvu16, "fvu_std_int4_mb": fvu4,
        "fvu_rel_diff": (fvu4 - fvu16) / fvu16,
        "mb_tokens": lay["mb_tokens"], "mb_bytes_over_fp16": lay["mb_bytes_over_fp16"],
        "mb_tokens_over_fp16": lay["mb_tokens_over_fp16"],
        "posthoc_resolution": {"naive_int4_replay": i4, "fp32_replay": x32,
                               "fp32_replay_over_floor": x32["mean"] / f_fr["mean"]}}

    out["fresh_vs_replay_fp32_same_seed"] = stat(
        [S(f"pair_fp32_freshb3x_s{k}__fp32_b3x_s{k}.json") for k in range(3)])

    def log(n):
        return {r["step"]: r for r in map(json.loads, open(sd / n / "train_log.jsonl"))}

    def p3(a, b, upto):
        la, lb = log(a), log(b)
        s = sorted(x for x in la if x in lb and x <= upto)
        same = all(la[x][k] == lb[x][k] for x in s for k in ("mse", "fvu", "l0", "dead_frac"))
        nxt = min(x for x in la if x in lb and x > upto)
        return {"logged_steps_equal": len(s), "through_step": max(s), "exact": bool(same),
                "next_logged_step": nxt, "mse_next": [la[nxt]["mse"], lb[nxt]["mse"]]}

    out["P3"] = {"fresh_vs_replay": p3("fp32_b3x_s0", "fp32_freshb3x_s0", 4800),
                 "mb_vs_fresh": p3("std-int4_freshb3x_s0", "std-int4_mbb3x_s0", 19100)}
    out["P1"] = J("fresh_forward_check.json")
    out["store"] = lay
    def manifest(store):
        """Live manifest, or its snapshot once the cache is reclaimed."""
        live = Path(args.root) / "cache" / store / "manifest.json"
        snap = (Path("cache_manifests_snapshot")
                / f"{args.root.replace('/', '_')}_cache_{store}_manifest.json")
        return json.loads((live if live.exists() else snap).read_text())

    out["store"]["n_shards_existing"] = len(manifest("master-train")["shards"])
    out["store"]["n_shards_new"] = len(manifest("master-freshtail-train")["shards"])
    out["tail_tokens"] = json.loads(
        (Path(args.root) / "tokens" / "meta_freshtail.json").read_text())
    out["stable"] = {k: v for k, v in J("fresh_stable.json").items() if k != "regimes"}
    (ev / "fresh_summary.json").write_text(json.dumps(out, indent=2))
    pct = lambda d: f"{100*d['mean']:.2f} ({100*d['min']:.2f}-{100*d['max']:.2f})"
    print("R1 floor fresh", pct(f_fr), "replay", pct(f_rp))
    print("R2 std-int4 fresh", pct(sf), "replay", pct(sr), "fp16 replay", pct(f16))
    print("R3 X16", pct(x16), "X4", pct(x4), "fp32 replay", pct(x32), "int4", pct(i4))
    print("P3", out["P3"])
    print(f"wrote {ev / 'fresh_summary.json'}")


if __name__ == "__main__":
    main()
