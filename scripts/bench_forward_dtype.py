"""Forward-pass throughput of GPT-2 small at fp32 and bf16 (the "2.21x
faster" bf16 forward in Section 4 and the Table 10 caption).

The paper's runs_m1/eval/forward_dtype_throughput.json came from an inline
loop run on 2026-08-13; this script re-implements that measurement so it
can be repeated: same model, layer 6 hidden states, batch shape (64 rows of
128 tokens plus the prepended EOT, as `acc cache --preset m1` uses), forward
only with no store write, and fp32 and bf16 interleaved in one session so
thermal drift hits both alike. Timings depend on the machine, so it writes
to a separate file by default rather than over the paper's artifact.

Usage: PYTHONPATH=src python scripts/bench_forward_dtype.py --root runs_m1
Writes <root>/eval/forward_dtype_throughput_rerun.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from acc.utils import get_device


def tokens_per_s(model, tokens, batch_rows, device, eot=50256):
    n = 0
    t0 = None
    with torch.inference_mode():
        for i, r0 in enumerate(range(0, tokens.shape[0], batch_rows)):
            rows = tokens[r0:r0 + batch_rows].astype(np.int64)
            ids = np.concatenate([np.full((rows.shape[0], 1), eot, np.int64), rows], 1)
            out = model(torch.from_numpy(ids).to(device), output_hidden_states=True)
            h = out.hidden_states[6][:, 1:, :].float().cpu()   # sync, as caching does
            if i == 0:          # first batch is warm-up, untimed
                t0 = time.perf_counter()
                continue
            n += h.shape[0] * h.shape[1]
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--rows", type=int, default=2048, help="token rows per run")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from transformers import AutoModel
    device = get_device()
    tokens = np.load(Path(args.root) / "tokens" / "tokens_eval.npy")[:args.rows]
    runs = {"fp32": [], "bf16": []}
    for _ in range(args.reps):
        for dt in ("fp32", "bf16"):
            m = AutoModel.from_pretrained("gpt2")
            if dt == "bf16":
                m = m.to(torch.bfloat16)
            m.eval().to(device)
            tps = tokens_per_s(m, tokens, 64, device)
            runs[dt].append(round(tps))
            print(f"{dt}: {tps:.0f} tok/s", flush=True)
            del m
    res = {dt: {"runs": v, "median": float(np.median(v))} for dt, v in runs.items()}
    res["speedup_bf16_over_fp32"] = res["bf16"]["median"] / res["fp32"]["median"]
    res["note"] = (f"GPT-2 small, {device}, batch_rows=64, layer 6; fp32 and bf16 "
                   "interleaved in one session; forward only, no store write")
    out = Path(args.out or Path(args.root) / "eval" / "forward_dtype_throughput_rerun.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"speedup {res['speedup_bf16_over_fp32']:.2f}x -> {out}")


if __name__ == "__main__":
    main()
