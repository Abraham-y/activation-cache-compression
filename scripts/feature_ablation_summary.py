"""Derived quantities from scripts/feature_ablation.py that the paper
quotes but the run itself did not write: agreement between the
correlational and causal criteria, and the pooled reseed values. No new
measurement; reads the run's npz only.

Usage: python scripts/feature_ablation_summary.py --root runs_m1
Writes <root>/eval/feature_ablation_summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RESEED = ["fp32_s1", "fp32_s2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--ref", default="fp32_s0")
    args = ap.parse_args()
    ev = Path(args.root) / "eval"
    d = json.loads((ev / f"feature_ablation_{args.ref}.json").read_text())
    z = np.load(ev / f"feature_ablation_{args.ref}.npz")
    out = {"ref": args.ref, "n_features": d["n_features"], "conds": {}}
    pooled_r, pooled_e = [], []
    for c in d["conds"]:
        r, e = z[f"{c}__best_corr_A"], z[f"{c}__effect_cos"]
        sv = r > 0.9
        row = {
            "frac_survived": float(sv.mean()),
            "frac_effect_gt_0.9": float((e > 0.9).mean()),
            "frac_effect_gt_0.9_among_survived":
                float((e[sv] > 0.9).mean()) if sv.any() else None,
            "frac_effect_gt_0.9_among_not_survived":
                float((e[~sv] > 0.9).mean()) if (~sv).any() else None,
            "criteria_agree": float(((r > 0.9) == (e > 0.9)).mean()),
            "median_effect_cos": float(np.median(e)),
            "median_dnext_r": float(np.median(z[f"{c}__dnext_r"])),
            "median_ctx_overlap": float(np.nanmedian(z[f"{c}__ctx"])),
        }
        out["conds"][c] = row
        if c != "fp32_s0_r1":
            pooled_r.append(r)
            pooled_e.append(e)
    rs = [out["conds"][c] for c in RESEED if c in out["conds"]]
    out["reseed_pooled"] = {k: float(np.mean([x[k] for x in rs]))
                            for k in ("frac_survived", "frac_effect_gt_0.9",
                                      "frac_effect_gt_0.9_among_survived",
                                      "median_effect_cos")}
    rho = spearmanr(np.concatenate(pooled_r), np.concatenate(pooled_e))
    out["spearman_bestcorr_vs_effect_cos"] = {
        "rho": float(rho.statistic), "n": int(sum(len(x) for x in pooled_r)),
        "excludes": "fp32_s0_r1 (bit-identical ceiling)"}
    agree = [v["criteria_agree"] for c, v in out["conds"].items() if c != "fp32_s0_r1"]
    out["criteria_agree_range"] = [float(min(agree)), float(max(agree))]
    (ev / "feature_ablation_summary.json").write_text(json.dumps(out, indent=2))
    for c, v in out["conds"].items():
        print(f"{c:>14}  eff>0.9 {v['frac_effect_gt_0.9']*100:5.1f}  "
              f"|surv {v['frac_effect_gt_0.9_among_survived']*100 if v['frac_effect_gt_0.9_among_survived'] is not None else float('nan'):5.1f}  "
              f"agree {v['criteria_agree']:.3f}")
    print("reseed pooled:", out["reseed_pooled"])
    print("spearman:", out["spearman_bestcorr_vs_effect_cos"])
    print(f"wrote {ev / 'feature_ablation_summary.json'}")


if __name__ == "__main__":
    main()
