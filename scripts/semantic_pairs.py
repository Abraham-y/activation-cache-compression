"""Semantic deep-check: side-by-side top-activating contexts for matched
feature pairs, sampled across survival bands. The question this
answers: does a 'survived' feature MEAN the same thing, at the level of
actual text, and does a 'damaged' one visibly not?

Usage: PYTHONPATH=src python3 scripts/semantic_pairs.py \
          --root runs_m1 --cand rot-int4_s0 --tokens 150000
Writes docs/appendix-semantic-pairs.md + eval/semantic_pairs_<cand>.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.cli import cache_dir, paths  # noqa: E402
from acc.metrics import top_positions_per_feature  # noqa: E402
from acc.sae import load_sae  # noqa: E402
from acc.store import Store  # noqa: E402
from acc.utils import write_json  # noqa: E402


def collect_triples(sae, store, max_tokens, device):
    feats, vals, poss = [], [], []
    seen = 0
    for i in range(len(store.shards)):
        if seen >= max_tokens:
            break
        x, _ = store.read_shard(i)
        x = x[: max_tokens - seen]
        for c0 in range(0, x.shape[0], 8192):
            xb = torch.from_numpy(x[c0 : c0 + 8192]).to(device)
            v, j = sae.feature_acts(xb)
            v = v.cpu().numpy()
            j = j.cpu().numpy()
            fired = v > 0
            gpos = seen + c0 + np.arange(xb.shape[0])
            feats.append(j[fired].astype(np.int64))
            vals.append(v[fired])
            poss.append(np.broadcast_to(gpos[:, None], v.shape)[fired])
        seen += x.shape[0]
    return (np.concatenate(feats), np.concatenate(vals),
            np.concatenate(poss), seen)


def context(tok, rows, pos, row_len, window=8):
    r, c = pos // row_len, pos % row_len
    lo, hi = max(0, c - window), min(row_len, c + window + 1)
    ids = rows[r, lo:hi].tolist()
    mark = c - lo
    toks = [tok.decode([t]) for t in ids]
    toks[mark] = f"[[{toks[mark]}]]"
    return "".join(toks).replace("\n", "\\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--ref", default="fp32_s0")
    ap.add_argument("--cand", default="rot-int4_s0")
    ap.add_argument("--tokens", type=int, default=150_000)
    ap.add_argument("--per-band", type=int, nargs=3, default=[8, 6, 6])
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    device = "cpu"

    z = np.load(Path(args.root) / "eval" /
                f"pair_{args.ref}__{args.cand}.npz")
    corr = np.nan_to_num(z["best_corr"])
    alive, best_idx = z["alive"], z["best_idx"]
    rng = np.random.default_rng(0)
    bands = {
        "survived": np.nonzero(alive & (corr > 0.9))[0],
        "degraded": np.nonzero(alive & (corr >= 0.5) & (corr <= 0.9))[0],
        "damaged": np.nonzero(alive & (corr < 0.5))[0],
    }
    sample = {
        b: rng.choice(idx, size=min(n, len(idx)), replace=False)
        for (b, idx), n in zip(bands.items(), args.per_band)
    }

    ref, _ = load_sae(paths(args.root)["saes"] / args.ref, device)
    cand, _ = load_sae(paths(args.root)["saes"] / args.cand, device)
    master = Store(cache_dir(args.root, "fp32", "eval"))
    rf, rv, rp, n_tok = collect_triples(ref, master, args.tokens, device)
    cf, cv, cp, _ = collect_triples(cand, master, args.tokens, device)
    tops_ref = top_positions_per_feature(rf, rv, rp, ref.d_sae, args.top)
    tops_cand = top_positions_per_feature(cf, cv, cp, cand.d_sae, args.top)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    rows = np.load(paths(args.root)["tokens"] / "tokens_eval.npy")
    row_len = rows.shape[1]

    out_md = [f"# Matched-pair contexts: {args.ref} vs {args.cand}",
              f"({n_tok} eval tokens; [[token]] = activating position; "
              f"top-{args.top} positions per feature)\n"]
    summary = []
    for band, idxs in sample.items():
        out_md.append(f"\n## {band}\n")
        for f in idxs:
            g = int(best_idx[f])
            a, b = tops_ref[f], tops_cand[g]
            shared = len(np.intersect1d(a, b)) / max(min(len(a), len(b)), 1)
            summary.append({"band": band, "ref_feat": int(f),
                            "cand_feat": g, "corr": float(corr[f]),
                            "top_overlap": float(shared)})
            out_md.append(f"### ref#{f} ↔ cand#{g}  (r={corr[f]:.3f}, "
                          f"top-{args.top} overlap {shared:.0%})\n")
            out_md.append("| ref contexts | cand contexts |")
            out_md.append("|---|---|")
            for k in range(min(5, len(a), max(len(b), 1))):
                ca = context(tok, rows, int(a[k]), row_len) if k < len(a) else ""
                cb = context(tok, rows, int(b[k]), row_len) if k < len(b) else ""
                out_md.append(f"| {ca} | {cb} |")
            out_md.append("")
    md_path = Path(f"docs/appendix-semantic-pairs-{args.cand}.md")
    md_path.write_text("\n".join(out_md))
    write_json(Path(args.root) / "eval" /
               f"semantic_pairs_{args.cand}.json",
               {"ref": args.ref, "cand": args.cand, "n_tokens": n_tok,
                "pairs": summary})
    for band in sample:
        ov = [s["top_overlap"] for s in summary if s["band"] == band]
        print(f"{band:>9}: n={len(ov)} mean top-overlap "
              f"{np.mean(ov):.2f} range {min(ov):.2f}-{max(ov):.2f}")
    print(f"-> {md_path}")


if __name__ == "__main__":
    main()
