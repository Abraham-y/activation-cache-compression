"""A seed floor for the stratified (stable-only) numbers.

The stratification in `acc.cli stability` defines a feature as stable if
its best match exceeds 0.9 in BOTH other fp32 seeds. That is fine for
showing damage concentrates on unstable features, but it leaves the
resulting percentage uncalibrated: on that set the seed comparator is
1.000 by construction, so the column has a ceiling of 1 and a floor of 1
and no unit. Read literally every codec then looks worse than process
noise, which is an artifact of the definition.

This computes the calibrated version. Define stability using ONE seed,
then use the HELD-OUT seed as the floor on exactly that population, and
report each codec against it. Both assignments (define on s1 / hold out
s2, and vice versa) are computed and reported.

Usage: PYTHONPATH=src python scripts/stable_floor.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.utils import write_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SITES = [("GPT-2 L6", "runs_m1"), ("Pythia L8", "runs_m4_L8"),
         ("Pythia L16", "runs_m4_L16")]
CODECS = ["fp16", "bf16", "rot-int8", "int8", "int8-o8", "rot-int6",
          "int6", "int6-o8", "rot-int4", "int4-o16", "int4", "pca256", "rp256"]
THRESH = 0.9


def corr(ev, ref, cand):
    p = ev / f"pair_{ref}__{cand}.npz"
    if not p.exists():
        return None, None
    with np.load(p) as z:
        return np.nan_to_num(z["best_corr"]).copy(), z["alive"].copy()


def main():
    out = {"threshold": THRESH,
           "note": ("Stable set defined by ONE fp32 seed; the HELD-OUT seed "
                    "supplies the floor on that same population. The "
                    "two-seed definition used elsewhere has a comparator of "
                    "1.000 by construction and therefore no floor."),
           "sites": {}}
    for name, root in SITES:
        ev = ROOT / root / "eval"
        c1, alive = corr(ev, "fp32_s0", "fp32_s1")
        c2, _ = corr(ev, "fp32_s0", "fp32_s2")
        if c1 is None or c2 is None:
            continue
        arms = []
        for lbl, defn, held in (("s1-defined", c1, c2), ("s2-defined", c2, c1)):
            stable = (defn > THRESH) & alive
            if stable.sum() == 0:
                continue
            floor = float((held[stable] > THRESH).mean())
            row = {"definition": lbl, "n_stable": int(stable.sum()),
                   "floor": floor, "codecs": {}}
            for c in CODECS:
                cc, _ = corr(ev, "fp32_s0", f"{c}_s0")
                if cc is None:
                    continue
                v = float((cc[stable] > THRESH).mean())
                row["codecs"][c] = {"survived_stable": v,
                                    "ratio_to_floor": v / floor if floor else None}
            arms.append(row)
        if not arms:
            continue
        mean = {}
        for c in arms[0]["codecs"]:
            if all(c in a["codecs"] for a in arms):
                mean[c] = {
                    "survived_stable": float(np.mean(
                        [a["codecs"][c]["survived_stable"] for a in arms])),
                    "ratio_to_floor": float(np.mean(
                        [a["codecs"][c]["ratio_to_floor"] for a in arms])),
                }
        out["sites"][name] = {
            "root": root, "arms": arms,
            "floor_mean": float(np.mean([a["floor"] for a in arms])),
            "floor_range": [float(min(a["floor"] for a in arms)),
                            float(max(a["floor"] for a in arms))],
            "mean_over_arms": mean,
        }
        f = out["sites"][name]
        print(f"{name:<12} floor {100*f['floor_mean']:.1f}% "
              f"({100*f['floor_range'][0]:.1f}-{100*f['floor_range'][1]:.1f})  "
              + "  ".join(f"{c} {100*v['survived_stable']:.1f}%"
                          f"={v['ratio_to_floor']:.2f}x"
                          for c, v in mean.items()
                          if c in ("rot-int8", "rot-int4", "int4")))
    o = ROOT / "runs_m1" / "eval" / "stable_floor.json"
    write_json(o, out)
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
