"""Cross-matched null: what does best-of-d_sae matching return by chance?

Survival takes a MAXIMUM over d_sae candidate features, so a sceptic is
right to ask whether the 0.9 band is reachable by chance selection. The
paired null in `acc.cli null` cannot answer this — it correlates feature
i against feature i and never takes a maximum. (An earlier draft of the
paper claimed it did; it does not, and the claim was removed.)

This runs the real thing, through the same StreamingCross /
band_fractions path used for every survival number in the paper.

USE --permute. A constant offset is NOT an adequate misalignment: rows
are row_len (127) tokens, so a fixed lag preserves position-in-context
phase exactly and the cross-correlation is a function of lag mod 127.
Position-locked features stay correlated and own the maximum, inflating
the apparent chance ceiling to 0.55 — and at some phases two alive
features exceed 0.9, which would falsify the very claim this script is
meant to support. A token permutation destroys content alignment and
phase together and gives 0.11 — but that is an UNSTRUCTURED null. Use
--rowperm for the structured one: permuting whole rows preserves phase
and within-row co-activation while breaking content correspondence, and
chance then reaches 0.975 for 0.062% of alive features. Report both; the
row-level figure is the honest bound on argmax selection bias, and it is
negligible in rate even though its maximum is high.

Usage: PYTHONPATH=src python scripts/misalignment_null.py \
           --root runs_m1 --preset m1 --ref fp32_s0 --cand rot-int8_s0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.cli import PRESETS, cache_dir, paths  # noqa: E402
from acc.metrics import band_fractions, topk_to_csr  # noqa: E402
from acc.metrics import StreamingCross  # noqa: E402
from acc.sae import load_sae  # noqa: E402
from acc.store import Store  # noqa: E402
from acc.utils import get_device, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--preset", default="m1", choices=list(PRESETS))
    ap.add_argument("--ref", default="fp32_s0")
    ap.add_argument("--cand", default="rot-int8_s0")
    ap.add_argument("--offset", type=int, default=177_777,
                    help="tokens to misalign the candidate side by")
    ap.add_argument("--permute", action="store_true",
                    help="PREFERRED. Randomly permute the candidate side "
                         "instead of offsetting it. A constant offset "
                         "preserves position-in-context phase exactly "
                         "(rows are row_len tokens, so the correlation is a "
                         "function of lag mod row_len), which leaves "
                         "position-locked features correlated and inflates "
                         "the apparent chance maximum. A permutation "
                         "destroys content alignment AND phase.")
    ap.add_argument("--rowperm", action="store_true",
                    help="Permute whole ROWS (blocks of row_len tokens) "
                         "instead of individual tokens. This destroys "
                         "content alignment while PRESERVING "
                         "position-in-context phase and within-row feature "
                         "co-activation, so it is the structured null for "
                         "the argmax question: how far does best-of-d_sae "
                         "climb when the candidate dictionary is genuinely "
                         "co-structured with the reference but the content "
                         "does not correspond? It should sit between the "
                         "token permutation and the fixed lag.")
    ap.add_argument("--row-len", type=int, default=127,
                    help="tokens per row; must match the eval cache layout")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    p = PRESETS[args.preset]
    device = args.device or get_device()

    ref, _ = load_sae(paths(args.root)["saes"] / args.ref, device)
    cand, _ = load_sae(paths(args.root)["saes"] / args.cand, device)
    master = Store(cache_dir(args.root, "fp32", "eval"))

    need = p["eval_tokens"] + args.offset
    parts, have = [], 0
    for x, _ in master.iter_shards():
        parts.append(x)
        have += x.shape[0]
        if have >= need:
            break
    xs = np.concatenate(parts, axis=0)
    n = min(p["eval_tokens"], xs.shape[0] - args.offset)
    if n <= 0:
        sys.exit(f"need > {args.offset} eval tokens, have {xs.shape[0]}")
    a_side = xs[:n]
    if args.rowperm:
        # keep position-in-context phase and within-row co-activation,
        # break only which row a position belongs to
        rl = args.row_len
        nrows = n // rl
        m = nrows * rl
        a_side = xs[:m]
        rng = np.random.default_rng(0)
        blocks = xs[:m].reshape(nrows, rl, -1)
        b_side = blocks[rng.permutation(nrows)].reshape(m, -1)
        n = m
    elif args.permute:
        rng = np.random.default_rng(0)
        b_side = xs[:n][rng.permutation(n)]
    else:
        b_side = xs[args.offset:args.offset + n]

    cross = StreamingCross(ref.d_sae, cand.d_sae)
    for c0 in range(0, n, 8192):
        xa = torch.from_numpy(a_side[c0:c0 + 8192]).to(device)
        xb = torch.from_numpy(b_side[c0:c0 + 8192]).to(device)
        va, ia = ref.feature_acts(xa)
        vb, ib = cand.feature_acts(xb)
        cross.update(topk_to_csr(va.cpu().numpy(), ia.cpu().numpy(), ref.d_sae),
                     topk_to_csr(vb.cpu().numpy(), ib.cpu().numpy(), cand.d_sae))
    best, _ = cross.match()
    alive = cross.firing_rate_x() > 1e-3
    bands = band_fractions(best, alive)
    c = np.nan_to_num(best)[alive]
    out = {
        "mode": ("rowperm_null_cross" if args.rowperm
                 else "permutation_null_cross" if args.permute
                 else "misalignment_null_cross"),
        "ref": args.ref, "cand": args.cand,
        "offset_tokens": None if args.permute else args.offset,
        "permuted": bool(args.permute), "row_permuted": bool(args.rowperm),
        "row_len": args.row_len if args.rowperm else None, "n_eval_tokens": n,
        "n_alive": int(alive.sum()),
        "note": ("best-of-d_sae matching between two SAEs on MISALIGNED "
                 "token positions: every match here is chance selection. "
                 "Compare against the survival threshold of 0.9."),
        "mean_best_corr": float(c.mean()), "p99_best_corr": float(np.percentile(c, 99)),
        "max_best_corr": float(c.max()),
        "frac_above_0.9": float((c > 0.9).mean()),
        "frac_above_0.5": float((c > 0.5).mean()),
        **{k: v for k, v in bands.items() if k in ("survived", "damaged")},
    }
    # the mode/offset goes in the filename: a rerun with different
    # settings must not silently overwrite a published artifact
    sfx = ("rowperm" if args.rowperm
           else "perm" if args.permute else f"off{args.offset}")
    o = (paths(args.root)["eval"]
         / f"nullcross_{sfx}_{args.ref}__{args.cand}.json")
    write_json(o, out)
    print(f"misaligned best-of-{ref.d_sae} [{args.ref} vs {args.cand}, "
          f"{out['mode']}]: mean r={out['mean_best_corr']:.4f} "
          f"p99={out['p99_best_corr']:.4f} max={out['max_best_corr']:.4f} "
          f"frac>0.9={out['frac_above_0.9']:.5f} -> {o}")


if __name__ == "__main__":
    main()
