"""How well do the cheap proxies ORDER the codecs, over every condition?

An internal audit caught the paper asserting Spearman rho = 1.000 without
computing it anywhere, over a hand-picked five of the fourteen released
GPT-2 conditions. Over all of them it is 0.991 --- and a free metric
(mean relative L2, no SAE, no encode pass, no eval token set) reaches
0.987. This producer computes both, per site, so the claim is generated
rather than typed.

It also isolates where the two diverge. Magnitude proxies rank the
quantization family fine; they inverts on a change of codec FAMILY, and
that is the only place the basis-aware proxy earns its cost.

Usage: PYTHONPATH=src python scripts/proxy_ranking.py
"""
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.utils import read_json, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SITES = [("GPT-2 L6", "runs_m1"), ("Pythia L8", "runs_m4_L8"),
         ("Pythia L16", "runs_m4_L16")]
PLOTTED = ["fp16", "rot-int8", "int8", "rot-int4", "int4"]


_NOT_A_CODEC = re.compile(r"^(noise\d+|sparseulp\d+|oneulp|clumpulp)$")


def main():
    out = {"note": "Spearman rho between each cheap proxy and same-seed "
                   "retrained survival. 'all' = every released condition at "
                   "that site; 'plotted' = the five codecs in the figure. "
                   "rel_l2 needs no SAE at all.", "sites": {}}
    for name, root in SITES:
        ev = ROOT / root / "eval"
        conds = []
        # CODEC conditions only. The glob also matches the noise/ulp
        # control arms, which are perturbations rather than codecs and
        # were never part of "all GPT-2 conditions"; sweeping them in
        # changes the reported rho on re-run (0.991 -> 0.976 for the
        # cosine) purely by changing the population. Their cosine values
        # are also the ones the old float32 metric saturated.
        for f in sorted(ev.glob("frozen_*__fp32_s0.json")):
            c = f.name[len("frozen_"):-len("__fp32_s0.json")]
            if _NOT_A_CODEC.match(c):
                continue
            p = ev / f"pair_fp32_s0__{c}_s0.json"
            if p.exists():
                d = read_json(f)
                conds.append((c, d["mean_corr"], d["direct"]["rel_l2_mean"],
                              d["direct"]["cos_mean"],
                              read_json(p)["survived"]))
        if len(conds) < 3:
            continue
        row = {"n_conditions": len(conds),
               "codecs": [c[0] for c in conds]}
        for key, idx, sign in (("frozen_r", 1, 1), ("rel_l2", 2, -1),
                               ("cos", 3, 1)):
            x = [sign * c[idx] for c in conds]
            y = [c[4] for c in conds]
            row[f"rho_{key}_all"] = float(spearmanr(x, y).statistic)
            sub = [i for i, c in enumerate(conds) if c[0] in PLOTTED]
            if len(sub) >= 3:
                row[f"rho_{key}_plotted"] = float(
                    spearmanr([x[i] for i in sub], [y[i] for i in sub]).statistic)
        out["sites"][name] = row
    # where magnitude and basis disagree: the projection family
    ev = ROOT / "runs_m1" / "eval"
    pairs = []
    for a, b in (("pca256", "int4"), ("rp256", "int4")):
        fa, fb = read_json(ev / f"frozen_{a}__fp32_s0.json"), read_json(ev / f"frozen_{b}__fp32_s0.json")
        sa = read_json(ev / f"pair_fp32_s0__{a}_s0.json")["survived"]
        sb = read_json(ev / f"pair_fp32_s0__{b}_s0.json")["survived"]
        pairs.append({
            "pair": f"{a} vs {b}",
            "rel_l2": [fa["direct"]["rel_l2_mean"], fb["direct"]["rel_l2_mean"]],
            "cos": [fa["direct"]["cos_mean"], fb["direct"]["cos_mean"]],
            "frozen_r": [fa["mean_corr"], fb["mean_corr"]],
            "retrained": [sa, sb],
            "magnitude_orders_correctly": bool(
                (fa["direct"]["rel_l2_mean"] > fb["direct"]["rel_l2_mean"]) == (sa < sb)),
            "frozen_orders_correctly": bool((fa["mean_corr"] < fb["mean_corr"]) == (sa < sb)),
        })
    out["family_change"] = pairs
    o = ROOT / "runs_m1" / "eval" / "proxy_ranking.json"
    write_json(o, out)
    for name, r in out["sites"].items():
        print(f"{name:<12} n={r['n_conditions']:>2}  frozen rho={r['rho_frozen_r_all']:.4f}"
              f"  rel-L2 rho={r['rho_rel_l2_all']:.4f}"
              f"  (plotted-5: {r.get('rho_frozen_r_plotted', float('nan')):.3f} /"
              f" {r.get('rho_rel_l2_plotted', float('nan')):.3f})")
    for p_ in pairs:
        print(f"  {p_['pair']}: magnitude correct? {p_['magnitude_orders_correctly']}"
              f"   frozen correct? {p_['frozen_orders_correctly']}")
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
