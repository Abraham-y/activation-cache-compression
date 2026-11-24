"""Do 'damaged' feature pairs mean different things, or the same thing
carved differently?

WHY THIS RUNS. The paper counts a feature as lost when its best match
falls below r=0.9, and reports that 77% of features turn over on a
reseed. Appendix B's hand-picked damaged pairs do not look like
strangers: ref#8926 fires on the noun after "one" (candidate, ball,
match, language, town) and its match cand#579 fires on the noun after
"one" (thing, candidate, person, type, reason). Same territory, carved
at a different angle. If that generalises, the survival metric
over-counts loss, and the seed-dependence literature's "reproducible
subspaces" reading (arXiv 2606.12138) applies to our numbers too.

The paper's existing semantic check measures TOP-POSITION overlap: the
same token in the same document. That is strict. Two features can fire
on the same KIND of token at different instances and score zero. This
measures token-TYPE overlap instead, over every alive feature rather
than a hand-picked sample, against a random-pair null.

PRE-REGISTERED READING, fixed before running.
Let J_pos be top-K position-overlap and J_tok be token-type Jaccard.
Null = each ref feature paired with a RANDOM alive candidate feature.
  A. damaged J_tok is at or near the null -> the turnover is real,
     damaged features genuinely differ, and the paper's framing stands
     unchanged.
  B. damaged J_tok sits well above the null while damaged J_pos does
     not -> damaged pairs fire on the same kinds of token at different
     instances. The metric over-counts loss and the paper must say so:
     "features turn over" would be partly a statement about basis
     re-partitioning, not destroyed content.
  C. survived J_tok is also near the null -> the measure is broken and
     nothing here is interpretable. (Sanity branch; would invalidate
     the check, not the paper.)
Report every branch. This is run to be reported, not to be believed
only if convenient.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.cli import cache_dir, paths           # noqa: E402
from acc.metrics import top_positions_per_feature  # noqa: E402
from acc.sae import load_sae                   # noqa: E402
from acc.store import Store                    # noqa: E402


def collect(sae, store, max_tokens, device):
    feats, vals, poss = [], [], []
    seen = 0
    for i in range(len(store.shards)):
        if seen >= max_tokens:
            break
        x, _ = store.read_shard(i)
        x = x[: max_tokens - seen]
        for c0 in range(0, x.shape[0], 8192):
            xb = torch.from_numpy(x[c0:c0 + 8192]).to(device)
            v, j = sae.feature_acts(xb)
            v, j = v.cpu().numpy(), j.cpu().numpy()
            fired = v > 0
            gpos = seen + c0 + np.arange(xb.shape[0])
            feats.append(j[fired].astype(np.int64))
            vals.append(v[fired])
            poss.append(np.broadcast_to(gpos[:, None], v.shape)[fired])
        seen += x.shape[0]
    return (np.concatenate(feats), np.concatenate(vals),
            np.concatenate(poss), seen)


def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--ref", default="fp32_s0")
    ap.add_argument("--cand", default="fp32_s1")
    ap.add_argument("--tokens", type=int, default=200_000)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()
    dev = "cpu"

    z = np.load(Path(args.root) / "eval" /
                f"pair_{args.ref}__{args.cand}.npz")
    corr = np.nan_to_num(z["best_corr"])
    alive, best_idx = z["alive"], z["best_idx"]

    ref, _ = load_sae(paths(args.root)["saes"] / args.ref, dev)
    cand, _ = load_sae(paths(args.root)["saes"] / args.cand, dev)
    master = Store(cache_dir(args.root, "fp32", "eval"))
    print("  collecting ref activations...", flush=True)
    rf, rv, rp, n_tok = collect(ref, master, args.tokens, dev)
    print("  collecting cand activations...", flush=True)
    cf, cv, cp, _ = collect(cand, master, args.tokens, dev)
    tops_r = top_positions_per_feature(rf, rv, rp, ref.d_sae, args.top)
    tops_c = top_positions_per_feature(cf, cv, cp, cand.d_sae, args.top)

    rows = np.load(paths(args.root)["tokens"] / "tokens_eval.npy").ravel()
    # Frequency control. The 10 commonest tokens are ~24% of the corpus,
    # so an overlap could in principle be two features both touching
    # stopwords. The random-pair null already prices that in, but we also
    # report the measure with the 50 commonest token types removed.
    u, c = np.unique(rows[:n_tok], return_counts=True)
    stop = set(u[np.argsort(-c)][:50].tolist())

    def tok_ids(positions, drop_stop=False):
        p = np.asarray(positions, dtype=np.int64)
        p = p[p < rows.size]
        ids = rows[p]
        if not drop_stop:
            return ids
        return np.array([x for x in ids.tolist() if x not in stop],
                        dtype=np.int64)

    bands = {"survived": alive & (corr > 0.9),
             "degraded": alive & (corr >= 0.5) & (corr <= 0.9),
             "damaged":  alive & (corr < 0.5)}
    alive_cand = np.unique(best_idx[alive])
    rng = np.random.default_rng(0)
    out = {"ref": args.ref, "cand": args.cand, "n_eval_tokens": int(n_tok),
           "top_k": args.top, "bands": {}}

    for band, mask in bands.items():
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            continue
        jp, jt, jp0, jt0, jts, jts0 = [], [], [], [], [], []
        for f in idx:
            g = int(best_idx[f])
            a, b = tops_r[f], tops_c[g]
            if len(a) == 0 or len(b) == 0:
                continue
            jp.append(jaccard(np.asarray(a), np.asarray(b)))
            jt.append(jaccard(tok_ids(a), tok_ids(b)))
            jts.append(jaccard(tok_ids(a, True), tok_ids(b, True)))
            gr = int(rng.choice(alive_cand))       # random-pair null
            br = tops_c[gr]
            if len(br):
                jp0.append(jaccard(np.asarray(a), np.asarray(br)))
                jt0.append(jaccard(tok_ids(a), tok_ids(br)))
                jts0.append(jaccard(tok_ids(a, True), tok_ids(br, True)))
        out["bands"][band] = {
            "n": len(jp),
            "position_jaccard_mean": float(np.mean(jp)),
            "tokentype_jaccard_mean": float(np.mean(jt)),
            "null_position_jaccard_mean": float(np.mean(jp0)),
            "null_tokentype_jaccard_mean": float(np.mean(jt0)),
            "tokentype_jaccard_nostop_mean": float(np.mean(jts)),
            "null_tokentype_jaccard_nostop_mean": float(np.mean(jts0)),
        }
        b = out["bands"][band]
        print(f"  {band:9} n={b['n']:5}  "
              f"pos {b['position_jaccard_mean']:.4f} "
              f"(null {b['null_position_jaccard_mean']:.4f})   "
              f"tok {b['tokentype_jaccard_mean']:.4f} "
              f"(null {b['null_tokentype_jaccard_mean']:.4f})   "
              f"nostop {b['tokentype_jaccard_nostop_mean']:.4f} "
              f"(null {b['null_tokentype_jaccard_nostop_mean']:.4f})",
              flush=True)

    p = Path(args.root) / "eval" / f"damaged_semantics_{args.cand}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
