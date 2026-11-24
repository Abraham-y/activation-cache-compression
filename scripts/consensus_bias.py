"""Bias-vs-variance test for codec damage, from released pair NPZs only.

Question: seed noise averages out across seeds, but a codec's
damage might be the SAME features in every seed (a shared bias that
multi-seed protocols cannot average away). Among the fp32 consensus set
(features of fp32_s0 matched at r>0.9 in BOTH other fp32 seeds), we ask
whether the features a codec loses at seed 0 are also lost at seeds 1
and 2, against the rate independence would predict. Same-seed pairs
(fp32_si vs codec_si) isolate the codec; fp32 cross-seed best_idx maps
seed-0 features into seeds 1 and 2. No training.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "runs_m1" / "eval"
T = 0.9
CODECS = {"rot-int8": ["rot-int8_s0", "rot-int8_k32_s1", "rot-int8_k32_s2"],
          "rot-int4": ["rot-int4_s0", "rot-int4_s1", "rot-int4_s2"],
          "int4": ["int4_s0", "int4_s1", "int4_s2"]}


def npz(a, b):
    p = E / f"pair_{a}__{b}.npz"
    return np.load(p) if p.exists() else None


cross = {1: npz("fp32_s0", "fp32_s1"), 2: npz("fp32_s0", "fp32_s2")}
alive0 = cross[1]["alive"].astype(bool)
stable = alive0 & (cross[1]["best_corr"] > T) & (cross[2]["best_corr"] > T)
C = np.where(stable)[0]
out = {"threshold": T, "n_consensus": int(len(C)), "codecs": {}}
for name, arms in CODECS.items():
    lost = {}
    for s, arm in enumerate(arms):
        p = npz(f"fp32_s{s}", arm)
        if p is None:
            lost = None
            break
        idx = C if s == 0 else cross[s]["best_idx"][C]
        lost[s] = p["best_corr"][idx] <= T
    if lost is None:
        print(f"{name}: missing pair, skipped")
        continue
    l0, l1, l2 = lost[0], lost[1], lost[2]
    rates = [float(l.mean()) for l in (l0, l1, l2)]
    cond1 = float(l1[l0].mean())
    cond12 = float((l1 & l2)[l0].mean())
    all3 = float((l0 & l1 & l2).mean())
    out["codecs"][name] = {
        "arms": arms, "loss_rate_per_seed": rates, "n_lost_s0": int(l0.sum()),
        "p_lost_s1_given_lost_s0": cond1, "p_lost_s1_indep": rates[1],
        "p_lost_s1_and_s2_given_lost_s0": cond12,
        "p_lost_s1_and_s2_indep": rates[1] * rates[2],
        "frac_lost_all_three": all3,
        "frac_lost_all_three_indep": rates[0] * rates[1] * rates[2],
        "frac_lost_any": float((l0 | l1 | l2).mean())}
    print(f"{name}: loss/seed {[round(100*r,1) for r in rates]}  "
          f"P(lost s1|lost s0)={100*cond1:.1f}% vs indep {100*rates[1]:.1f}%  "
          f"lost in all three {100*all3:.2f}% vs indep {100*rates[0]*rates[1]*rates[2]:.3f}%")
(E / "consensus_bias.json").write_text(json.dumps(out, indent=1))
print(f"consensus set n={len(C)}; wrote consensus_bias.json")
