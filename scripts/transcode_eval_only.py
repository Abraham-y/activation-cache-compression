"""Transcode only the EVAL split, calibrating from the TRAIN master.

`acc.cli transcode` always does both splits. For a frozen-basis eval we
need only the eval store, and at Pythia scale the train-side stores we
would not use are ~20 GB per codec — enough to matter on a laptop.

Calibration still comes from the train master, so the codec produced
here is the same codec the SAEs were trained on (same mu/sigma sample,
same seeded rotation). That is the whole reason the train master has to
be regenerated too, rather than calibrating on eval.

Usage: PYTHONPATH=src python scripts/transcode_eval_only.py \
           --root runs_m4_L8 --preset m4-pythia --codec rot-int4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.cli import PRESETS, cache_dir  # noqa: E402
from acc.store import transcode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--preset", default="m4-pythia", choices=list(PRESETS))
    ap.add_argument("--codec", required=True)
    args = ap.parse_args()
    p = PRESETS[args.preset]
    out = cache_dir(args.root, args.codec, "eval")
    if (out / "manifest.json").exists():
        print(f"skip {args.codec}-eval: manifest exists")
        return
    m = transcode(
        cache_dir(args.root, "fp32", "eval"), out, args.codec,
        calib_tokens=p["calib_tokens"],
        calib_master_dir=cache_dir(args.root, "fp32", "train"),
    )
    print(f"{args.codec}-eval: {m['n_tokens']} tokens, "
          f"{m['payload_bytes_per_token']:.1f} B/tok, "
          f"{m['transcode_seconds']:.1f}s")


if __name__ == "__main__":
    main()
