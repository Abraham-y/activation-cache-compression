"""Stable-feature version of the 3x fresh-vs-replay comparison: per-seed
results, with the stable-feature evaluation protocol unchanged.

The protocol is the paper's own (acc.cli cmd_stability and
scripts/budget_fixedset.py): with fp32 seed K as reference, a feature is
stable if its best match exceeds r = 0.9 in BOTH other fp32 seeds of the
same regime; the fixed-size variant takes the top N = 774 alive features
by their minimum cross-seed correlation (N is the 1x stable count, as in
tab:budget). Survival of a codec at seed K is then read from the same-seed
pair fp32_sK vs codec_sK, on all alive / stable / unstable features.
Regimes: replay (fp32_b3x, the existing 20M cache) and fresh
(fp32_freshb3x). Codec: std-int4 in both; fp16 in replay only.

READING, written 7 Oct 2026 before this ran: the pre-registered R2 was on
all alive features; this is its stable-stratum counterpart, descriptive.
If std-int4's stable-stratum survival in fresh and replay agree within
2 pp (seed means), the protocol gives the same answer with fresh data; a
larger gap is reported as a regime effect on stable features.

Usage: python scripts/fresh_stable.py --root runs_m1
Needs pair NPZs with every fp32 seed as reference (run_fresh_b3x.sh plus
the six extra pairs listed in REQUIRED). Writes <root>/eval/fresh_stable.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

REGIMES = {"replay": "fp32_b3x", "fresh": "fp32_freshb3x"}
CODECS = {"replay": ["std-int4_b3x", "fp16_b3x"], "fresh": ["std-int4_freshb3x"]}
N_FIXED = 774
R = 0.9


def load(ev, ref, cand):
    with np.load(ev / f"pair_{ref}__{cand}.npz") as z:
        return np.nan_to_num(z["best_corr"]), z["alive"].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    args = ap.parse_args()
    ev = Path(args.root) / "eval"
    out = {"protocol": "stable = best match > 0.9 in both other fp32 seeds; "
                       "fixed = top-774 by min cross-seed r", "regimes": {}}
    for reg, f32 in REGIMES.items():
        per_seed = []
        for k in range(3):
            others = [j for j in range(3) if j != k]
            cs = [load(ev, f"{f32}_s{k}", f"{f32}_s{j}") for j in others]
            alive = cs[0][1]
            mincorr = np.minimum(cs[0][0], cs[1][0])
            stable = alive & (mincorr > R)
            order = np.argsort(-np.where(alive, mincorr, -np.inf))
            fixed = np.zeros_like(alive)
            fixed[order[:N_FIXED]] = True
            row = {"seed": k, "n_alive": int(alive.sum()),
                   "stable_frac": float(stable[alive].mean()),
                   "n_stable": int(stable.sum())}
            for c in CODECS[reg]:
                corr, a2 = load(ev, f"{f32}_s{k}", f"{c}_s{k}")
                assert np.array_equal(a2, alive)
                s = corr > R
                row[c] = {"all": float(s[alive].mean()),
                          "stable": float(s[stable].mean()),
                          "unstable": float(s[alive & ~stable].mean()),
                          "fixed774": float(s[fixed].mean())}
            per_seed.append(row)
        summ = {"stable_frac_mean": float(np.mean([r["stable_frac"] for r in per_seed]))}
        for c in CODECS[reg]:
            for key in ("all", "stable", "unstable", "fixed774"):
                v = [r[c][key] for r in per_seed]
                summ[f"{c}_{key}_mean"] = float(np.mean(v))
                summ[f"{c}_{key}_range"] = [float(min(v)), float(max(v))]
        out["regimes"][reg] = {"per_seed": per_seed, "summary": summ}
    a = out["regimes"]["fresh"]["summary"]["std-int4_freshb3x_stable_mean"]
    b = out["regimes"]["replay"]["summary"]["std-int4_b3x_stable_mean"]
    out["std_int4_stable_fresh_minus_replay"] = a - b
    out["reading"] = ("agree within 2 pp" if abs(a - b) <= 0.02
                      else "regime effect on stable features")
    (ev / "fresh_stable.json").write_text(json.dumps(out, indent=2))
    for reg, d in out["regimes"].items():
        print(f"== {reg}: stable {100*d['summary']['stable_frac_mean']:.1f}% of alive")
        for r in d["per_seed"]:
            cells = "  ".join(f"{c}: all {100*r[c]['all']:.1f} stable {100*r[c]['stable']:.1f} "
                              f"unstable {100*r[c]['unstable']:.1f} fixed {100*r[c]['fixed774']:.1f}"
                              for c in CODECS[reg])
            print(f"  s{r['seed']} (stable n={r['n_stable']}): {cells}")
    print(f"std-int4 stable, fresh - replay: {100*(a-b):+.2f} pp -> {out['reading']}")


if __name__ == "__main__":
    main()
