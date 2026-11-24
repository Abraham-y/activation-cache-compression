"""Producer for eval/streaming_comparison.json (the "why cache at all" numbers).

Streaming regenerates activations for every SAE; caching amortizes one
forward across all runs. The comparison needs two rates:
- forward_tok_s: MEASURED here — the model forward loop from
  acc.model.cache_activations (same batching, EOT prepend, layer grab)
  minus the store writer, timed after warmup.
- train_consumption_tok_s: DERIVED from recorded artifacts — mean
  whole-run rate (steps x batch_size / train_seconds) over the fp32
  seed trainings, so the number has on-disk provenance and does not
  drift with thermal noise.

Usage: PYTHONPATH=src python3 scripts/streaming_comparison.py --root runs_m1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.data import EOT  # noqa: E402
from acc.utils import get_device, write_json  # noqa: E402


def measure_forward(tokens_path, layer, batch_rows, device, model_name="gpt2",
                    warmup_batches=3, timed_batches=40):
    from transformers import AutoModel

    tokens = np.load(tokens_path, mmap_mode="r")
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    d = model.config.hidden_size
    n_tok = 0
    with torch.inference_mode():
        for b in range(warmup_batches + timed_batches):
            rows = np.asarray(tokens[b * batch_rows:(b + 1) * batch_rows],
                              dtype=np.int64)
            ids = np.concatenate(
                [np.full((rows.shape[0], 1), EOT, dtype=np.int64), rows], axis=1)
            input_ids = torch.from_numpy(ids).to(device)
            out = model(input_ids, output_hidden_states=True)
            h = out.hidden_states[layer][:, 1:, :]
            h.reshape(-1, d).float().cpu().numpy()  # include the transfer, as caching does
            if b == warmup_batches - 1:
                t0 = time.perf_counter()
                n_tok = 0
            n_tok += rows.size
    return n_tok / (time.perf_counter() - t0)


def train_rate_from_artifacts(root):
    rates = []
    for s in (0, 1, 2):
        p = root / "saes" / f"fp32_s{s}" / "summary.json"
        d = json.loads(p.read_text())
        rates.append(d["cfg"]["steps"] * d["cfg"]["batch_size"] / d["train_seconds"])
    return float(np.mean(rates)), rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--batch-rows", type=int, default=64)  # m1 preset value
    ap.add_argument("--timed-batches", type=int, default=40)
    ap.add_argument("--sweep-size", type=int, default=9)  # the M6 k x seed grid
    args = ap.parse_args()
    root = Path(args.root)
    device = get_device()

    train_tok_s, per_seed = train_rate_from_artifacts(root)
    print(f"train consumption (fp32 s0/s1/s2 artifacts): "
          f"{'/'.join(f'{r:.0f}' for r in per_seed)} -> mean {train_tok_s:.0f} tok/s")
    fwd_tok_s = measure_forward(root / "tokens" / "tokens_train.npy",
                                args.layer, args.batch_rows, device,
                                timed_batches=args.timed_batches)
    print(f"forward ({device}, batch_rows={args.batch_rows}): {fwd_tok_s:.0f} tok/s")

    out = {
        "forward_tok_s": round(fwd_tok_s, 1),
        "train_consumption_tok_s": round(train_tok_s, 1),
        "streaming_fraction_of_cached_speed": fwd_tok_s / train_tok_s,
        "sweep_size": args.sweep_size,
        "note": f"GPT-2 small, {device}. Forward measured by this script "
                f"(cache_activations loop minus the writer); train "
                f"consumption is the mean whole-run rate recorded in the "
                f"fp32 seed summaries. Streaming regenerates activations "
                f"for every SAE; caching amortizes one forward across all "
                f"runs.",
    }
    write_json(root / "eval" / "streaming_comparison.json", out)
    print(f"streaming fraction: {out['streaming_fraction_of_cached_speed']:.1%}; "
          f"wrote {root / 'eval' / 'streaming_comparison.json'}")


if __name__ == "__main__":
    main()
