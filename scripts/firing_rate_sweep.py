"""Ratio-to-floor as the firing-rate cutoff defining the alive set moves.

The alive cutoff is a free parameter of the protocol (0.1% in the paper),
and it moves the reported level like every other conditioning choice.
Recomputed from released best_corr/firing_rate_ref arrays; no retraining.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
SITES = {"gpt2_l6": "runs_m1", "pythia_l8": "runs_m4_L8", "pythia_l16": "runs_m4_L16"}
CUTS = [0.001, 0.002, 0.005, 0.01, 0.02]


def sweep(root, codec="rot-int4"):
    ref = np.load(ROOT / root / "eval" / f"pair_fp32_s0__{codec}_s0.npz")
    if "firing_rate_ref" not in ref:
        return None
    fr = ref["firing_rate_ref"]
    floors = [np.load(ROOT / root / "eval" / f"pair_fp32_s0__fp32_s{s}.npz")
              for s in (1, 2)]
    out = {}
    for c in CUTS:
        m = fr > c
        if m.sum() < 20:
            continue
        cand = float(100 * np.mean(ref["best_corr"][m] > 0.9))
        fl = float(np.mean([100 * np.mean(f["best_corr"][m] > 0.9) for f in floors]))
        out[f"{c}"] = {"n_alive": int(m.sum()), "survived": cand,
                       "floor": fl, "ratio_to_floor": cand / fl}
    return out


def main():
    res = {"cuts": CUTS, "codec": "rot-int4", "sites": {}}
    for name, root in SITES.items():
        s = sweep(root)
        if s:
            res["sites"][name] = s
    out = ROOT / "runs_m1" / "eval" / "firing_rate_sweep.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    for site, d in res["sites"].items():
        lo, hi = min(d.values(), key=lambda r: r["ratio_to_floor"]), \
                 max(d.values(), key=lambda r: r["ratio_to_floor"])
        print(f"  {site:12} ratio {hi['ratio_to_floor']:.2f} -> "
              f"{lo['ratio_to_floor']:.2f}  (n {hi['n_alive']} -> {lo['n_alive']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
