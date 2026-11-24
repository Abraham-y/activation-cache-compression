"""Replicate the causal steering floor over many splits.

Why. The steering floor was one draw at seed=0 while the SAE floor gets
three seed pairs and the probe floor got seven splits. An audit pass ran
eight splits and found the floor spans 2.8x, which means the single-draw
number the paper quotes -- and the strictest-convention "0.52 -> 1.05
FAIL" disclosure built on top of it -- may both be split artifacts. If
so the paper is confessing to a failure it does not have, which is as
much a reporting error as the overclaim would have been.

This recomputes the floor across N splits and rescores every codec
already on disk against the resulting distribution.

Pre-registered reading, fixed before running:
  - PCA's ag_news ratio stays under 1 at the MEAN floor under the
    strictest (x2) convention => the 1.05 disclosure is a single-split
    artifact and must be replaced with the replicated range.
  - It exceeds 1 at the mean => the disclosure stands as written.
  - int4's failure must survive at the most permissive draw or the
    below-floor verdict itself is in question.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acc.probes import load_probe_data  # noqa: E402
from acc.steering import (causal_effect, direction,  # noqa: E402
                          split_halves)

PROMPTS = [
    "The movie was", "I thought the film", "Overall, this was",
    "My honest opinion is that it", "The acting felt",
    "Would I recommend it? It", "The script was", "In short, the picture",
]
N_SPLITS = 400
LAYER, STRENGTH = 6, 5.0


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    out = {}
    for task in ("sst2", "ag_news"):
        d = Path("runs_m1/probes") / task
        if not (d / "master-train").exists():
            print(f"  {task}: no store, skipped")
            continue
        xf, yf = load_probe_data(d / "master-train")
        floors, dcos, ccos = [], [], []
        for k in range(N_SPLITS):
            ia, ib = split_halves(xf, yf, seed=k)
            va = direction(xf[ia], yf[ia])
            vb = direction(xf[ib], yf[ib])
            da = causal_effect(va, PROMPTS, LAYER, STRENGTH, model, tok, "cpu")
            db = causal_effect(vb, PROMPTS, LAYER, STRENGTH, model, tok, "cpu")
            floors.append(float(np.linalg.norm(da - db)))
            # record the COSINE floors too: two of tab:steering's three
            # columns were still scored against a single seed-0 draw,
            # with PCA passing by ~1e-4
            cs = lambda p_, q_: float(np.dot(p_, q_) / (
                np.linalg.norm(p_) * np.linalg.norm(q_) + 1e-30))
            dcos.append(cs(va, vb))
            ccos.append(cs(da, db))
        fl = np.array(floors)
        out[task] = {"n_splits": N_SPLITS, "floors": floors,
                     "mean": float(fl.mean()), "min": float(fl.min()),
                     "max": float(fl.max()),
                     "median": float(np.median(fl)),
                     "dir_cos_mean": float(np.mean(dcos)),
                     "dir_cos_min": float(np.min(dcos)),
                     "causal_cos_mean": float(np.mean(ccos)),
                     "causal_cos_min": float(np.min(ccos))}
        print(f"\n  {task}: floor L2 over {N_SPLITS} splits — "
              f"mean {fl.mean():.4f}  [{fl.min():.4f}, {fl.max():.4f}]  "
              f"spread {fl.max()/fl.min():.1f}x")

        for f in sorted(Path("runs_m1/eval").glob(f"steering_{task}_*.json")):
            c = json.load(open(f))
            if "causal" not in c:
                continue
            num = c["causal"]["l2_effect_fp32_vs_codec"]
            print(f"      {c['codec']:9} ratio at mean floor "
                  f"{num/fl.mean():.3f}  "
                  f"[{num/fl.max():.3f}, {num/fl.min():.3f}]   "
                  f"strict(x2) at mean {2*num/fl.mean():.3f}")
    p = Path("runs_m1/eval/steering_floor_splits.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
