"""Weight-space divergence between trained SAEs, on scale-appropriate stats.

max |delta| over a 9.4M-entry tensor is the least stable statistic
available, and reporting it made a one-scalar data perturbation look like
it moved the weights a lot. It did not: relative Frobenius against a
reseed is the right normaliser, and by that measure the same run sits at
4% of a reseed with every alive latent still best-matching itself.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent


def stats(ref: str, cand: str):
    a = torch.load(ROOT / f"runs_m1/saes/{ref}/sae.pt", map_location="cpu",
                   weights_only=True)["state_dict"]
    b = torch.load(ROOT / f"runs_m1/saes/{cand}/sae.pt", map_location="cpu",
                   weights_only=True)["state_dict"]
    out = {}
    for k in ("W_dec", "W_enc"):
        # float64: float32 norms over 9.4M entries drift by ~1e-3 relative
        ak, bk = a[k].double(), b[k].double()
        d = ak - bk
        out[k] = {
            "max_abs": float(d.abs().max()),
            "rel_frobenius": float(d.norm() / ak.norm()),
            "median_abs": float(d.abs().median()),
        }
    return out


def main():
    ref = "fp32_s0"
    arms = ["fp32_s0_r1", "oneulp_s0", "sparseulp6_s0", "noise7_s0",
            "fp16_s0", "fp32_s1"]
    res = {"reference": ref, "note": __doc__.strip(), "arms": {}}
    for c in arms:
        if (ROOT / f"runs_m1/saes/{c}/sae.pt").exists():
            res["arms"][c] = stats(ref, c)
    dest = ROOT / "runs_m1" / "eval" / "weight_divergence.json"
    dest.write_text(json.dumps(res, indent=2))
    print(f"wrote {dest}\n")
    print(f"{'arm':16} {'W_dec relFro':>13} {'max|d|':>9} {'median|d|':>11}")
    for c, v in res["arms"].items():
        w = v["W_dec"]
        print(f"{c:16} {w['rel_frobenius']:13.4f} {w['max_abs']:9.4f} "
              f"{w['median_abs']:11.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
