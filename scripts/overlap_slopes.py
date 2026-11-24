"""Slopes of the isotropic-noise ladder against the quantizer family,
computed over the SAME magnitude range.

The paper's resolution asymmetry compares a near-lossless rate to a
codec-range rate. Those come from different perturbation families, and
rates from different families over disjoint ranges need not be
commensurable. These two rungs (1e-3, 1e-2) put isotropic noise inside
the range real codecs occupy so the comparison can be checked directly.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
E = ROOT / "runs_m1" / "eval"


def arm(tag):
    fr = json.loads((E / f"frozen_{tag}__fp32_s0.json").read_text())
    pr = json.loads((E / f"pair_fp32_s0__{tag}_s0.json").read_text())
    return fr["direct"]["rel_l2_mean"], 100 * pr["survived"]


def slope(a, b):
    (ra, sa), (rb, sb) = a, b
    return (sa - sb) / (math.log10(rb) - math.log10(ra))


def main():
    out = {"note": __doc__.strip(), "arms": {}, "slopes_pp_per_decade": {}}
    for t in ("noise7", "noise6", "noise5", "noise4", "noise3", "noise2",
              "fp16", "rot-int8", "rot-int4", "int4"):
        f = E / f"frozen_{t}__fp32_s0.json"
        if f.exists():
            r, s = arm(t)
            out["arms"][t] = {"rel_l2": r, "survived": s}
    pairs = {
        "noise_1e-3_to_1e-2": ("noise3", "noise2"),
        "noise_1e-4_to_1e-2": ("noise4", "noise2"),
        "noise_1e-7_to_1e-4": ("noise7", "noise4"),
        "quantizer_fp16_to_rot-int8": ("fp16", "rot-int8"),
        "quantizer_fp16_to_rot-int4": ("fp16", "rot-int4"),
    }
    for k, (a, b) in pairs.items():
        if a in out["arms"] and b in out["arms"]:
            out["slopes_pp_per_decade"][k] = round(slope(arm(a), arm(b)), 4)
    dest = E / "overlap_slopes.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    for k, v in out["slopes_pp_per_decade"].items():
        print(f"  {k:32} {v:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
