"""Fixed-population stable-set survival across training budgets.

Producer for eval/budget_fixedset.json (Fig. budget_scaling, tab:budget).
The naive per-tier stratification re-derives the stable set at each
budget, so the population grows as training stabilizes and the three
points compare different features. This holds the population fixed:
rank alive features by min cross-seed matched correlation (s0-s1,
s0-s2), take the same top-N at every tier, where N is the count of
features clearing r>0.9 at the 1x tier. Survival per codec is then
computed over that fixed-size set from the same-seed pair NPZs.

Usage: PYTHONPATH=src python3 scripts/budget_fixedset.py --root runs_m1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.utils import write_json  # noqa: E402

TIERS = {"1x": "", "3x": "_b3x", "10x": "_b10x"}
CODECS = ["fp16", "rot-int8", "rot-int4"]
STABLE_R = 0.9


def pair(eval_dir, ref, cand):
    with np.load(eval_dir / f"pair_{ref}__{cand}.npz") as z:
        return z["best_corr"].copy(), z["alive"].copy()


def token_visits_M(root, sfx):
    cfg = json.loads((root / "saes" / f"fp32{sfx}_s0" / "summary.json").read_text())["cfg"]
    return round(cfg["steps"] * cfg["batch_size"] / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--out", default=None,
                    help="output path (default <root>/eval/budget_fixedset.json)")
    args = ap.parse_args()
    root = Path(args.root)
    eval_dir = root / "eval"

    # Per-tier stability score: min matched correlation across both
    # cross-seed pairs, restricted to the tier's own alive set.
    scores, alives = {}, {}
    for tier, sfx in TIERS.items():
        c1, alive = pair(eval_dir, f"fp32{sfx}_s0", f"fp32{sfx}_s1")
        c2, _ = pair(eval_dir, f"fp32{sfx}_s0", f"fp32{sfx}_s2")
        scores[tier] = np.minimum(c1, c2)
        alives[tier] = alive

    # N is fixed by the 1x tier: alive features stable at r>0.9 there.
    n_fixed = int((scores["1x"][alives["1x"]] > STABLE_R).sum())

    out = {
        "method": "stable set = top-N features by min cross-seed matched "
                  "correlation, N fixed across tiers so the population "
                  "cannot drift with budget",
        "n_fixed": n_fixed,
        "binomial_se_pp": float(np.sqrt(0.9 * 0.1 / n_fixed) * 100),
        "tiers": {},
    }
    print(f"N = {n_fixed} (binomial SE {out['binomial_se_pp']:.2f}pp)")
    print(f"{'tier':<5} {'thresh':>7} {'n_alive':>8} "
          + "".join(f"{c:>10}" for c in CODECS) + f" {'gap pp':>7}")
    top_1x = None
    for tier, sfx in TIERS.items():
        idx = np.flatnonzero(alives[tier])
        top = idx[np.argsort(scores[tier][idx])[::-1][:n_fixed]]
        if top_1x is None:
            top_1x = set(top.tolist())
        row = {
            "token_visits_M": token_visits_M(root, sfx),
            "stability_threshold_at_N": float(scores[tier][top].min()),
            "n_alive": int(alives[tier].sum()),
            # how much of the 1x tier's set this tier's set reuses: the
            # sets are constant-SIZE, not constant-IDENTITY
            "overlap_with_1x_set": len(top_1x & set(top.tolist())),
        }
        for codec in CODECS:
            corr, _ = pair(eval_dir, f"fp32{sfx}_s0", f"{codec}{sfx}_s0")
            # raw count as well as the fraction: fp16 lands on the SAME
            # count at all three tiers, so the fraction prints identically
            # to full precision. That is a coincidence at the counting
            # level, not a copied number — the tiers' feature sets differ.
            row[codec] = float((corr[top] > STABLE_R).mean())
            row[f"{codec}_count"] = int((corr[top] > STABLE_R).sum())
        row["gap_fp16_minus_rotint4_pp"] = (row["fp16"] - row["rot-int4"]) * 100
        out["tiers"][tier] = row
        print(f"{tier:<5} {row['stability_threshold_at_N']:>7.4f} "
              f"{row['n_alive']:>8} "
              + "".join(f"{row[c]:>10.4f}" for c in CODECS)
              + f" {row['gap_fp16_minus_rotint4_pp']:>7.2f}")

    out_path = Path(args.out) if args.out else eval_dir / "budget_fixedset.json"
    write_json(out_path, out)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
