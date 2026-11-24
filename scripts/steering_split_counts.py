"""Per-split failure counts for the causal steering test (tab:steering),
from the released effects and the 400 resample floors. A codec fails a
split when its causal L2 effect exceeds that split's floor (x1) or half of
it (x2, the like-for-like referent: the half-vs-half floor runs about twice
a full-data comparison). The table's ag_news rows use the original
sentiment prompts; the task-matched-prompt counts live in
steering_matched_prompts.json.

Usage: python scripts/steering_split_counts.py --root runs_m1
Writes <root>/eval/steering_split_counts.json
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    args = ap.parse_args()
    ev = Path(args.root) / "eval"
    fs = json.loads((ev / "steering_floor_splits.json").read_text())
    out = {}
    for task in ("sst2", "ag_news"):
        floors = fs[task]["floors"]
        for f in sorted(ev.glob(f"steering_{task}_*.json")):
            codec = f.stem[len(f"steering_{task}_"):]
            e = json.loads(f.read_text())["causal"]["l2_effect_fp32_vs_codec"]
            out[f"{task}/{codec}"] = {
                "n_splits": len(floors), "l2_effect": e,
                "fails_x1": sum(e > fl for fl in floors),
                "fails_x2": sum(e > fl / 2 for fl in floors)}
    (ev / "steering_split_counts.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"{k:>18}: fails x1 {v['fails_x1']:>3}/{v['n_splits']}  x2 {v['fails_x2']:>3}/{v['n_splits']}")


if __name__ == "__main__":
    main()
