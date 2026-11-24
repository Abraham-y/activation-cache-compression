"""Paired, within-run feature-level tests.

Every comparison in this paper is between two arms evaluated on the SAME
alive feature set against the SAME fp32 reference. That is a matched-pairs
design at n ~ 4,860, and collapsing it to two marginal percentages throws
the pairing away. McNemar recovers it, and a paired bootstrap puts an
interval on the ratios, which are otherwise quoted without one.

Features are not independent, so the p-values are optimistic in magnitude;
they are reported as effect-size evidence within a run, not as a claim
about the feature population. The value is that a single-seed cell stops
resting on an eyeballed tie band.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
T = 0.9


def surv(root, ref, cand, mask=None):
    d = np.load(ROOT / root / "eval" / f"pair_{ref}__{cand}.npz")
    m = d["alive"].astype(bool) if mask is None else mask
    return (d["best_corr"][m] > T), m


def mcnemar(a, b):
    """Exact McNemar on two boolean survival vectors over the same features."""
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return n01, n10, 1.0
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue
    return n01, n10, float(p)


def boot_ratio(num, den, reps=20000, seed=0):
    """Paired bootstrap over features for a ratio of two survival fractions."""
    rng = np.random.default_rng(seed)
    n = len(num)
    out = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        d = num[idx].mean(), den[idx].mean()
        out[i] = d[0] / d[1] if d[1] > 0 else np.nan
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(num.mean() / den.mean()), float(lo), float(hi)


def main():
    res = {"note": __doc__.strip(), "threshold": T, "forward_precision": {},
           "below_floor": {}, "ratio_cis": {}}

    # 1. the generalisation claim, per seed, paired within seed
    for s in (0, 1, 2):
        try:
            a, m = surv("runs_m1", f"fp32_s{s}", f"fwdbf16_s{s}")
            cand = f"rot-int8_s{s}" if s == 0 else f"rot-int8_k32_s{s}"
            b, _ = surv("runs_m1", f"fp32_s{s}", cand, mask=m)
        except FileNotFoundError:
            continue
        n01, n10, p = mcnemar(a, b)
        res["forward_precision"][f"seed{s}"] = {
            "bf16_only": n01, "rot_int8_only": n10,
            "delta_pp": 100 * (a.mean() - b.mean()), "mcnemar_p": p}

    # 2. below-floor / above-floor verdicts against the seed floor, per site
    for site, root, fl_cand in (("gpt2_l6", "runs_m1", "fp32_s1"),
                                ("pythia_l8", "runs_m4_L8", "fp32_s1"),
                                ("pythia_l16", "runs_m4_L16", "fp32_s1")):
        try:
            fl, m = surv(root, "fp32_s0", fl_cand)
        except FileNotFoundError:
            continue
        for c in ("rot-int4", "int4"):
            try:
                a, _ = surv(root, "fp32_s0", f"{c}_s0", mask=m)
            except FileNotFoundError:
                continue
            n01, n10, p = mcnemar(a, fl)
            r, lo, hi = boot_ratio(a, fl)
            res["below_floor"][f"{site}/{c}"] = {
                "ratio_to_floor": r, "ci95": [lo, hi],
                "codec_only": n01, "floor_only": n10, "mcnemar_p": p}

    # 3. bootstrap CIs on the ratios the paper quotes bare, all three sites
    for site, root in (("gpt2_l6", "runs_m1"), ("pythia_l8", "runs_m4_L8"),
                       ("pythia_l16", "runs_m4_L16")):
        try:
            fl, m = surv(root, "fp32_s0", "fp32_s1")
        except FileNotFoundError:
            continue
        for c in ("fp16", "rot-int8", "int8", "rot-int4", "int4"):
            f = ROOT / root / "eval" / f"pair_fp32_s0__{c}_s0.npz"
            if not f.exists():
                continue
            a, _ = surv(root, "fp32_s0", f"{c}_s0", mask=m)
            r, lo, hi = boot_ratio(a, fl)
            res["ratio_cis"][f"{site}/{c}"] = {
                "ratio": r, "ci95": [lo, hi], "excludes_1": bool(lo > 1 or hi < 1)}

    out = ROOT / "runs_m1" / "eval" / "paired_feature_tests.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}\n")
    print("FORWARD PRECISION (paired within each seed):")
    for k, v in res["forward_precision"].items():
        print(f"  {k}: delta {v['delta_pp']:+.2f} pp, discordant "
              f"{v['bf16_only']}/{v['rot_int8_only']}, McNemar p = {v['mcnemar_p']:.2g}")
    print("\nRATIO CIs (paired feature bootstrap):")
    for k, v in res["ratio_cis"].items():
        flag = "" if v["excludes_1"] else "   <- spans 1"
        print(f"  {k:22} {v['ratio']:.2f}x  95% CI "
              f"[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
