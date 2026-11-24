"""Task-matched prompts and a strength sweep for the causal steering arm.

WHY THIS RUNS. Two internal audit passes independently hit the same
defect: the
ag_news causal readout scores a World-vs-Sports direction using the EIGHT
SENTIMENT PROMPTS written for sst2 ("The movie was", "The acting felt",
...). tab:steering discloses this, but it means the paper's flagship
instrument-disagreement result rests on an off-task readout at a single
injection strength (5.0) with no sweep. The claim it supports -- PCA-256
is near-free causally while losing 92% of SAE features -- is stated in
the abstract.

This fixes both: news-lead prompts for the topic task, and a strength
sweep to show whether any of it is an artifact of strength = 5.0.

STAGE A: ag_news, task-matched prompts, full 400-split floor at the
reference strength, all codecs rescored. Directly replaces the off-task
numbers.
STAGE B: both tasks, strengths {1, 2, 5, 10}, within-call half-half floor.
Tests strength-dependence cheaply.

PRE-REGISTERED READING, fixed before this ran.
Current claims at stake: (i) naive int4 is the sole failure on both
tasks, (ii) PCA-256 clears the line on the mean floor but is "a pass with
a failure rate" (51/400 ag_news splits exceed 1 under x2).
  A. int4 still fails on ag_news with matched prompts, at every strength
     -> the below-floor verdict is prompt- and strength-robust. Report
     the matched numbers and keep the claim.
  B. int4 passes anywhere -> the "below the floor at every site tested"
     claim acquires a readout caveat and must be stated with it.
  C. PCA still clears the mean floor with matched prompts -> the
     instrument-disagreement headline survives the objection; report the
     matched failure rate in place of the sentiment-prompt one.
  D. PCA fails the mean floor with matched prompts -> the dissociation
     was an artifact of scoring a topic direction with sentiment prompts.
     That guts the abstract's second headline and MUST be reported as
     such, with the abstract and contribution 4 rewritten.
  E. Verdicts move with strength -> the single-strength number was
     underdetermined; report the sweep, not a point.
In every branch the matched-prompt numbers replace the off-task ones in
the paper. This is run to be reported, not to be believed only if it
is convenient.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acc.probes import load_probe_data  # noqa: E402
from acc.steering import (causal_effect, direction,  # noqa: E402
                          split_halves)

# sst2 contrasts negative vs positive review sentiment.
PROMPTS_SST2 = [
    "The movie was", "I thought the film", "Overall, this was",
    "My honest opinion is that it", "The acting felt",
    "Would I recommend it? It", "The script was", "In short, the picture",
]
# ag_news classes 0 and 1 are World vs Sports. News-lead framings, neutral
# between the two classes so neither is favoured by the prompt itself.
PROMPTS_AGNEWS = [
    "The report said", "According to officials", "In the news today",
    "The headline read", "Reporters covering the story",
    "The latest update is that", "Coverage of the event",
    "The story developing now",
]
PROMPTS = {"sst2": PROMPTS_SST2, "ag_news": PROMPTS_AGNEWS}

LAYER = 6
REF_STRENGTH = 5.0
STRENGTHS = [1.0, 2.0, 5.0, 10.0]
N_SPLITS_A = 400
N_SPLITS_B = 400   # was 40; must match N_SPLITS_A or the two stages
                   # quote ratios against differently-resampled floors
                   # (an 11% shift at strength 5.0), which is exactly the
                   # error this paper exists to warn about.
CODECS = ["fp16", "rot-int8", "rot-int4", "int4", "pca256"]


def _dirs(task):
    d = Path("runs_m1/probes") / task
    xf, yf = load_probe_data(d / "master-train")
    v_fp32 = direction(xf, yf)
    out = {}
    for c in CODECS:
        cd = d / f"{c}-train"
        if not (cd / "manifest.json").exists():
            continue
        xc, yc = load_probe_data(cd)
        if not np.array_equal(yf, yc):
            raise ValueError(f"label mismatch for {c}")
        out[c] = direction(xc, yc)
    return xf, yf, v_fp32, out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    prior = {}
    _pp = Path("runs_m1/eval/steering_matched_prompts.json")
    if _pp.exists():
        prior = json.loads(_pp.read_text())
    res = {"prompts": PROMPTS, "layer": LAYER,
           "note": "Task-matched prompts. Both stages use a 400-split "
                   "floor so their ratios are on one scale."}

    # ---------- STAGE A: matched prompts, full floor, reference strength
    for task in ("ag_news", "sst2"):
        d = Path("runs_m1/probes") / task
        if not (d / "master-train").exists():
            print(f"  {task}: no store, skipped"); continue
        if f"stageA_{task}" in prior:
            res[f"stageA_{task}"] = prior[f"stageA_{task}"]
            print(f"    {task}: stage A cached, reusing")
            continue
        xf, yf, v_fp32, vc = _dirs(task)
        P = PROMPTS[task]
        floors = []
        for k in range(N_SPLITS_A):
            ia, ib = split_halves(xf, yf, seed=k)
            da = causal_effect(direction(xf[ia], yf[ia]), P, LAYER,
                               REF_STRENGTH, model, tok, "cpu")
            db = causal_effect(direction(xf[ib], yf[ib]), P, LAYER,
                               REF_STRENGTH, model, tok, "cpu")
            floors.append(float(np.linalg.norm(da - db)))
            if (k + 1) % 50 == 0:
                print(f"    {task} floor {k+1}/{N_SPLITS_A}", flush=True)
        fl = np.array(floors)
        d_fp32 = causal_effect(v_fp32, P, LAYER, REF_STRENGTH, model, tok, "cpu")
        arms = {}
        for c, v in vc.items():
            dc = causal_effect(v, P, LAYER, REF_STRENGTH, model, tok, "cpu")
            num = float(np.linalg.norm(d_fp32 - dc))
            # x2 convention: the measured floor doubles a like-for-like
            # referent, so compare the numerator against floor/2.
            arms[c] = {
                "l2_effect": num,
                "ratio_at_mean_floor": num / float(fl.mean()),
                "ratio_x2_at_mean_floor": 2 * num / float(fl.mean()),
                "fails_x1_of_400": int((num > fl).sum()),
                "fails_x2_of_400": int((num > fl / 2).sum()),
            }
            print(f"    {task} {c:9} ratio {arms[c]['ratio_at_mean_floor']:.3f} "
                  f"x2 {arms[c]['ratio_x2_at_mean_floor']:.3f} "
                  f"fails {arms[c]['fails_x1_of_400']}/400 x1, "
                  f"{arms[c]['fails_x2_of_400']}/400 x2", flush=True)
        res[f"stageA_{task}"] = {
            "n_splits": N_SPLITS_A, "strength": REF_STRENGTH,
            "floor_mean": float(fl.mean()), "floor_min": float(fl.min()),
            "floor_max": float(fl.max()), "arms": arms}

    # ---------- STAGE B: strength sweep
    for task in ("ag_news", "sst2"):
        d = Path("runs_m1/probes") / task
        if not (d / "master-train").exists():
            continue
        xf, yf, v_fp32, vc = _dirs(task)
        P = PROMPTS[task]
        sweep = {}
        for s in STRENGTHS:
            floors = []
            for k in range(N_SPLITS_B):
                ia, ib = split_halves(xf, yf, seed=k)
                da = causal_effect(direction(xf[ia], yf[ia]), P, LAYER, s,
                                   model, tok, "cpu")
                db = causal_effect(direction(xf[ib], yf[ib]), P, LAYER, s,
                                   model, tok, "cpu")
                floors.append(float(np.linalg.norm(da - db)))
            fm = float(np.mean(floors))
            d_fp32 = causal_effect(v_fp32, P, LAYER, s, model, tok, "cpu")
            row = {}
            for c, v in vc.items():
                dc = causal_effect(v, P, LAYER, s, model, tok, "cpu")
                num = float(np.linalg.norm(d_fp32 - dc))
                row[c] = {"ratio": num / fm, "ratio_x2": 2 * num / fm}
            sweep[str(s)] = {"floor_mean": fm, "arms": row}
            print(f"    {task} strength {s}: " +
                  "  ".join(f"{c} {row[c]['ratio']:.2f}" for c in row),
                  flush=True)
        res[f"stageB_{task}"] = sweep

    p = Path("runs_m1/eval/steering_matched_prompts.json")
    p.write_text(json.dumps(res, indent=1))
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
