"""Do matched features play the same causal role? Held-out contexts and
feature ablation, fp32 vs each retrained candidate, reseed as control.

WHY THIS RUNS. Camera-ready addition: for a sample of matched SAE
features, compare held-out activating contexts and feature-ablation
effects between fp32 and standardized int4, with re-seeding as the
control. The paper's matching is correlational (best
Pearson on the first 500k eval tokens). This asks whether a matched
partner also does the same thing to the model's predictions.

THE MEASURE. A fixed random sample of alive fp32_s0 features (seeded, the
same sample for every condition). Each feature's partner in a condition is
the paper's own best match from the eval-pair npz. Everything below is on
held-out eval tokens (rows starting at token 500,126; the matching used
tokens 0-499,999).
  - r_B: held-out Pearson between the feature and its partner.
  - ctx: overlap of their top-50 held-out activating positions
    (|A and B| / min, the paper's App. B measure).
  - Ablation. Take the 32 held-out rows where the reference feature
    peaks. In GPT-2's forward pass, at the SAE site (output of block 5,
    hidden_states[6]), subtract the feature's own contribution a_i(h) *
    W_dec[i] at every position, leaving the rest of the residual stream
    untouched (no reconstruction splice, so SAE error is not a
    confound). Do the same with the partner, using the candidate SAE, on
    the same rows. At the positions where the reference feature fires,
    sum the change in next-token log-probabilities over the vocabulary,
    centered across the vocabulary: that vector is the feature's effect.
    effect_cos is the cosine between the two effect vectors (0 if the
    partner never fires there, since it then does nothing). dnext_r is
    the Pearson between the two ablations' changes in the log-probability
    of the actual next token, over every position in those rows.

PIPELINE CHECKS (must pass before reading anything).
  1. Alignment: the site activations recomputed in the forward pass must
     match the cached fp32 eval activations for the same tokens (max abs
     diff < 1e-3 of the max abs value), and the reference feature's
     firing positions must match the cache with Jaccard >= 0.95 (TopK
     membership at the k-th boundary can flip under ~1e-6 differences).
  2. Ceiling: fp32_s0_r1 (retrained on identical bytes) must give median
     effect_cos > 0.99. Otherwise the measurement's own noise is too
     large to read.

PRE-REGISTERED READINGS, fixed 29 Sept 2026 before any condition ran.
Reseed = fp32_s1 and fp32_s2 pooled. Paired bootstrap over sampled
features (2,000 resamples) for unconditional contrasts; unpaired within a
band.
  F1. Does the r > 0.9 criterion track causal role? In every condition,
      the median effect_cos of survived pairs (r > 0.9) must exceed that
      of damaged pairs (r < 0.5). If it fails in any condition, the
      paper's criterion does not track what features do; say so up
      front.
  F2. The std-int4 question. Among survived pairs, std-int4's median
      effect_cos minus the reseed's: CI includes 0 or is positive ->
      "features std-int4 retains play the same causal role as seed-stable
      features do". CI entirely negative -> retained features are
      functionally weaker than seed-stable ones; report it.
  F3. Does the functional check confirm the codec verdict? Unconditional
      fraction of sampled features whose partner has effect_cos > 0.9,
      std-int4 minus reseed: CI entirely positive -> confirms; includes
      0 -> the functional check does not separate std-int4 from
      reseeding; entirely negative -> contradicts.
Report every branch that fires.

Usage: PYTHONPATH=src python scripts/feature_ablation.py --root runs_m1
Writes <root>/eval/feature_ablation_<ref>.{json,npz}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.cli import cache_dir, paths                     # noqa: E402
from acc.metrics import overlap_fraction, top_positions_per_feature  # noqa: E402
from acc.sae import load_sae                             # noqa: E402
from acc.store import Store                              # noqa: E402
from acc.utils import get_device, write_json             # noqa: E402

SPLIT = 500_000
CONDS = ["fp32_s0_r1", "sparseulp6_s0", "fp16_s0", "std-int4_s0",
         "int4_s0", "fp32_s1", "fp32_s2"]
RESEED = ["fp32_s1", "fp32_s2"]
LAYER = 6                 # hidden_states index; the hook sits on block LAYER-1
TOP_CTX = 50
N_ROWS = 32
ROW_BATCH = 16


def load_heldout(root, row_len):
    """fp32 site activations for whole rows starting at the first row
    boundary at or after SPLIT. Returns (x [n, d], first_row)."""
    first_row = -(-SPLIT // row_len)
    t0 = first_row * row_len
    store = Store(cache_dir(root, "fp32", "eval"))
    parts, seen = [], 0
    for s in range(len(store.shards)):
        x, _ = store.read_shard(s)
        lo, hi = max(t0 - seen, 0), x.shape[0]
        if lo < hi:
            parts.append(x[lo:hi])
        seen += x.shape[0]
    x = np.concatenate(parts)
    n_rows = x.shape[0] // row_len
    return x[: n_rows * row_len], first_row


def sparse_acts(sae, x, feats, device):
    """csr [n, len(feats)] of the given features' TopK activations."""
    col = np.full(sae.d_sae, -1, dtype=np.int64)
    col[feats] = np.arange(len(feats))
    colt = torch.from_numpy(col).to(device)
    rows_, cols_, vals_ = [], [], []
    for c0 in range(0, x.shape[0], 8192):
        xt = torch.from_numpy(x[c0 : c0 + 8192]).to(device)
        with torch.inference_mode():
            v, j = sae.feature_acts(xt)
            c = colt[j]
            keep = (c >= 0) & (v > 0)
            r = torch.nonzero(keep, as_tuple=True)[0]
        rows_.append(r.cpu().numpy() + c0)
        cols_.append(c[keep].cpu().numpy())
        vals_.append(v[keep].float().cpu().numpy())
    return sp.csr_matrix(
        (np.concatenate(vals_).astype(np.float64),
         (np.concatenate(rows_), np.concatenate(cols_))),
        shape=(x.shape[0], len(feats)))


def col_pearson(a, b):
    """Pearson between matching columns of two csr matrices [n, k]."""
    n = a.shape[0]
    sa, sb = np.asarray(a.sum(0)).ravel(), np.asarray(b.sum(0)).ravel()
    qa = np.asarray(a.multiply(a).sum(0)).ravel()
    qb = np.asarray(b.multiply(b).sum(0)).ravel()
    ab = np.asarray(a.multiply(b).sum(0)).ravel()
    cov = ab / n - sa * sb / n**2
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt((qa / n - (sa / n) ** 2) * (qb / n - (sb / n) ** 2))


def top_positions(csr, top_n):
    coo = csr.tocoo()
    return top_positions_per_feature(coo.col, coo.data, coo.row,
                                     csr.shape[1], top_n)


class Ablator:
    """Forward hook on block LAYER-1: optionally subtract one SAE feature's
    contribution at every non-sink position, and record where it fired."""

    def __init__(self):
        self.sae = None
        self.feat = None
        self.fired = None
        self.captured = None
        self.capture = False

    def __call__(self, _mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.capture:
            self.captured = h[:, 1:, :].detach().clone()
        if self.sae is None:
            return out
        B, T, d = h.shape
        x = h[:, 1:, :].reshape(-1, d)
        v, j = self.sae.feature_acts(x)
        a = (v * (j == self.feat)).sum(-1).view(B, T - 1, 1)
        self.fired = (a[..., 0] > 0)
        h2 = h.clone()
        h2[:, 1:, :] = h[:, 1:, :] - a * self.sae.W_dec[self.feat].view(1, 1, d)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--ref", default="fp32_s0")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--n-features", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="", help="output-name suffix (smoke runs)")
    args = ap.parse_args()
    conds = args.conds.split(",")
    device = args.device or get_device()
    t_start = time.perf_counter()
    P = paths(args.root)

    # ---- sample and partners
    pairs = {c: np.load(P["eval"] / f"pair_{args.ref}__{c}.npz") for c in conds}
    alive = pairs[conds[0]]["alive"]
    for c in conds:
        assert np.array_equal(pairs[c]["alive"], alive), f"alive mask differs: {c}"
    rng = np.random.default_rng(args.seed)
    F = np.sort(rng.choice(np.flatnonzero(alive), args.n_features, replace=False))
    partner = {c: pairs[c]["best_idx"][F] for c in conds}
    rA = {c: pairs[c]["best_corr"][F] for c in conds}

    # ---- held-out activations, contexts, r_B
    tokens = np.load(Path(args.root) / "tokens" / "tokens_eval.npy")
    row_len = tokens.shape[1]
    x, first_row = load_heldout(args.root, row_len)
    n_rows = x.shape[0] // row_len
    print(f"  held-out: {x.shape[0]} tokens, rows {first_row}..{first_row+n_rows-1}")
    saes = {args.ref: load_sae(P["saes"] / args.ref, device)[0]}
    for c in conds:
        saes[c] = load_sae(P["saes"] / c, device)[0]
    ref = saes[args.ref]
    a_ref = sparse_acts(ref, x, F, device)
    top_ref = top_positions(a_ref, TOP_CTX)
    r_B, ctx, cand_acts = {}, {}, {}
    for c in conds:
        uniq, inv = np.unique(partner[c], return_inverse=True)
        b = sparse_acts(saes[c], x, uniq, device)[:, inv]
        cand_acts[c] = b
        r_B[c] = col_pearson(a_ref, b.tocsr())
        top_c = top_positions(b.tocsr(), TOP_CTX)
        ctx[c] = np.array([overlap_fraction(top_ref[k], top_c[k]) for k in range(len(F))])
    print(f"  contexts done ({time.perf_counter()-t_start:.0f}s)")

    # ---- rows per feature: top N_ROWS rows by the ref feature's peak
    coo = a_ref.tocoo()
    peak = np.zeros((n_rows, len(F)))
    np.maximum.at(peak, (coo.row // row_len, coo.col), coo.data)
    rows_for = [np.argsort(-peak[:, k])[:N_ROWS] for k in range(len(F))]
    rows_for = [r[peak[r, k] > 0] for k, r in enumerate(rows_for)]

    # ---- model
    from transformers import GPT2LMHeadModel
    from acc.data import EOT
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(device)
    abl = Ablator()
    handle = model.transformer.h[LAYER - 1].register_forward_hook(abl)

    def run(ids):
        with torch.inference_mode():
            return torch.log_softmax(model(ids).logits.float(), dim=-1)

    def ids_for(rr):
        t = tokens[first_row + rr].astype(np.int64)
        t = np.concatenate([np.full((len(rr), 1), EOT, dtype=np.int64), t], 1)
        return torch.from_numpy(t).to(device)

    # alignment check on the first feature's rows
    ids0 = ids_for(rows_for[0])
    abl.capture, abl.sae = True, None
    run(ids0)
    abl.capture = False
    cached = torch.from_numpy(np.stack(
        [x[r * row_len : (r + 1) * row_len] for r in rows_for[0]])).to(device)
    align = float((abl.captured - cached).abs().max() / cached.abs().max())
    abl.sae, abl.feat = ref, int(F[0])
    run(ids0)
    fired_model = abl.fired.cpu().numpy()
    fired_cache = np.zeros_like(fired_model)
    sub = a_ref[:, 0].tocoo()
    pos_map = {r: i for i, r in enumerate(rows_for[0])}
    for p in sub.row:
        r = p // row_len
        if r in pos_map:
            fired_cache[pos_map[r], p % row_len] = True
    # Jaccard over fired positions: plain elementwise agreement would pass
    # a misaligned run, since almost every position is silent on both sides
    fire_agree = float((fired_model & fired_cache).sum()
                       / max((fired_model | fired_cache).sum(), 1))
    align_pass = align < 1e-3 and fire_agree >= 0.95
    print(f"  alignment: max rel diff {align:.2e}, firing agreement "
          f"{fire_agree:.4f} -> {'PASS' if align_pass else 'FAIL'}")
    if not align_pass:
        sys.exit(1)

    # ---- ablations
    K = len(F)
    keys = ["effect_cos", "dnext_r", "mag_ratio", "kl_cand", "cofire"]
    res = {c: {k: np.full(K, np.nan) for k in keys} for c in conds}
    kl_ref = np.full(K, np.nan)
    n_fire_ref = np.zeros(K, dtype=np.int64)

    def effect(sae, feat, batches, clean):
        E, dn, kl, cof, anyf = 0, [], 0.0, 0, False
        for (ids, fire), (lp_c, lpn_c) in zip(batches, clean):
            abl.sae, abl.feat = sae, int(feat)
            lp = run(ids)
            f_ab = abl.fired
            anyf = anyf or bool(f_ab.any())
            sel = torch.zeros(ids.shape[0], ids.shape[1], dtype=torch.bool, device=device)
            sel[:, 1:] = fire
            lpf, lcf = lp[sel], lp_c[sel]
            E = E + (lpf - lcf).sum(0)
            kl += float((lcf.exp() * (lcf - lpf)).sum())
            cof += int((f_ab & fire).sum())
            nxt = ids[:, 2:]
            dn.append((lp[:, 1:-1].gather(-1, nxt[..., None])[..., 0] - lpn_c).flatten())
        E = E - E.mean()
        dn = torch.cat(dn)
        if not anyf:   # partner silent on every row: exactly no effect
            E, dn, kl = torch.zeros_like(E), torch.zeros_like(dn), 0.0
        return E, dn, kl, cof

    for k in range(K):
        rr = rows_for[k]
        batches, clean = [], []
        for b0 in range(0, len(rr), ROW_BATCH):
            ids = ids_for(rr[b0 : b0 + ROW_BATCH])
            abl.sae = None
            lp_c = run(ids)
            batches.append([ids, None])      # firing mask filled in below
            clean.append((lp_c, lp_c[:, 1:-1].gather(-1, ids[:, 2:, None])[..., 0]))
        # The ref-ablation pass also yields the ref firing mask: the hook
        # encodes the stream it receives, which is clean up to the site.
        E_ref, dn_ref, kl_r, n_f = 0, [], 0.0, 0
        for bi, (ids, _) in enumerate(batches):
            abl.sae, abl.feat = ref, int(F[k])
            lp = run(ids)
            fire = abl.fired.clone()
            batches[bi][1] = fire
            lp_c = clean[bi][0]
            sel = torch.zeros(ids.shape[0], ids.shape[1], dtype=torch.bool, device=device)
            sel[:, 1:] = fire
            lpf, lcf = lp[sel], lp_c[sel]
            E_ref = E_ref + (lpf - lcf).sum(0)
            kl_r += float((lcf.exp() * (lcf - lpf)).sum())
            n_f += int(fire.sum())
            dn_ref.append((lp[:, 1:-1].gather(-1, ids[:, 2:, None])[..., 0]
                           - clean[bi][1]).flatten())
        E_ref = E_ref - E_ref.mean()
        dn_ref = torch.cat(dn_ref)
        n_fire_ref[k] = n_f
        kl_ref[k] = kl_r / max(n_f, 1)
        for c in conds:
            E_c, dn_c, kl_c, cof = effect(saes[c], partner[c][k], batches, clean)
            nr, nc = float(E_ref.norm()), float(E_c.norm())
            res[c]["effect_cos"][k] = float(E_ref @ E_c) / (nr * nc) if nc > 0 and nr > 0 else 0.0
            dr = torch.corrcoef(torch.stack([dn_ref, dn_c]))[0, 1]
            res[c]["dnext_r"][k] = float(dr) if torch.isfinite(dr) else 0.0
            res[c]["mag_ratio"][k] = nc / nr if nr > 0 else np.nan
            res[c]["kl_cand"][k] = kl_c / max(n_f, 1)
            res[c]["cofire"][k] = cof / max(n_f, 1)
        if k % 25 == 0:
            print(f"  feature {k+1}/{K}  ({time.perf_counter()-t_start:.0f}s)  "
                  + "  ".join(f"{c}:{res[c]['effect_cos'][k]:.2f}" for c in conds))
    handle.remove()

    # ---- summaries and pre-registered contrasts
    boot = np.random.default_rng(1)
    B_IDX = boot.integers(0, K, size=(2000, K))

    def band(c):
        r = rA[c]
        return np.where(r > 0.9, "survived", np.where(r < 0.5, "damaged", "degraded"))

    summ = {}
    for c in conds:
        bd = band(c)
        e = res[c]["effect_cos"]
        s = {"survived_frac_A": float((bd == "survived").mean()),
             "median_effect_cos": float(np.median(e)),
             "frac_effect_cos_gt_0.9": float((e > 0.9).mean()),
             "frac_effect_cos_gt_0.7": float((e > 0.7).mean()),
             "median_dnext_r": float(np.median(res[c]["dnext_r"])),
             "median_r_B": float(np.nanmedian(r_B[c])),
             "median_ctx_overlap": float(np.nanmedian(ctx[c]))}
        for b in ("survived", "degraded", "damaged"):
            m = bd == b
            s[f"n_{b}"] = int(m.sum())
            for key, arr in (("effect_cos", e), ("dnext_r", res[c]["dnext_r"]),
                             ("ctx_overlap", ctx[c]), ("mag_ratio", res[c]["mag_ratio"])):
                s[f"{b}_median_{key}"] = float(np.nanmedian(arr[m])) if m.any() else None
        summ[c] = s

    contrasts = {}
    ceiling_pass = summ.get("fp32_s0_r1", {}).get("median_effect_cos", 0) > 0.99
    contrasts["ceiling_pass"] = bool(ceiling_pass)
    f1 = {}
    for c in conds:
        sv, dm = summ[c]["survived_median_effect_cos"], summ[c]["damaged_median_effect_cos"]
        f1[c] = None if sv is None or dm is None else bool(sv > dm)
    contrasts["F1_survived_gt_damaged"] = f1
    rs = [c for c in RESEED if c in conds]
    if "std-int4_s0" in conds and rs:
        e_std = res["std-int4_s0"]["effect_cos"]
        e_rs = np.mean([res[c]["effect_cos"] for c in rs], axis=0)
        g_std = (e_std > 0.9).astype(float)
        g_rs = np.mean([(res[c]["effect_cos"] > 0.9) for c in rs], axis=0)
        d = g_std - g_rs
        bd = (d[B_IDX]).mean(1)
        contrasts["F3_frac_gt_0.9_std_minus_reseed"] = {
            "estimate": float(d.mean()),
            "ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]}
        dm_ = e_std - e_rs
        bm = np.median(dm_[B_IDX], axis=1)
        contrasts["F3b_median_paired_diff_effect_cos"] = {
            "estimate": float(np.median(dm_)),
            "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}
        # F2: survived pairs, unpaired bootstrap
        sv_std = e_std[band("std-int4_s0") == "survived"]
        sv_rs = np.concatenate([res[c]["effect_cos"][band(c) == "survived"] for c in rs])
        bs = [np.median(boot.choice(sv_std, sv_std.size)) - np.median(boot.choice(sv_rs, sv_rs.size))
              for _ in range(2000)]
        contrasts["F2_survived_median_std_minus_reseed"] = {
            "estimate": float(np.median(sv_std) - np.median(sv_rs)),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_std": int(sv_std.size), "n_reseed_pooled": int(sv_rs.size)}

    out = {"ref": args.ref, "conds": conds, "n_features": K, "seed": args.seed,
           "heldout_first_row": int(first_row), "heldout_tokens": int(x.shape[0]),
           "rows_per_feature": N_ROWS, "top_ctx": TOP_CTX,
           "alignment_max_rel_diff": align, "alignment_fire_agree": fire_agree,
           "median_kl_ref_per_fire": float(np.nanmedian(kl_ref)),
           "median_n_fire_ref": float(np.median(n_fire_ref)),
           "summary": summ, "contrasts": contrasts,
           "seconds": time.perf_counter() - t_start}
    stem = P["eval"] / f"feature_ablation_{args.ref}{args.tag}"
    write_json(stem.with_suffix(".json"), out)
    np.savez_compressed(
        stem.with_suffix(".npz"), features=F, kl_ref=kl_ref, n_fire_ref=n_fire_ref,
        **{f"{c}__partner": partner[c] for c in conds},
        **{f"{c}__best_corr_A": rA[c] for c in conds},
        **{f"{c}__r_B": r_B[c] for c in conds},
        **{f"{c}__ctx": ctx[c] for c in conds},
        **{f"{c}__{k}": res[c][k] for c in conds for k in keys})
    for c in conds:
        s = summ[c]
        print(f"  {c:>14}: surv {s['survived_frac_A']*100:5.1f}%  effect_cos med "
              f"{s['median_effect_cos']:.3f}  >0.9 {s['frac_effect_cos_gt_0.9']*100:5.1f}%  "
              f"surv-band {s['survived_median_effect_cos']}  dmg-band {s['damaged_median_effect_cos']}")
    print("  contrasts:", contrasts)
    print(f"  wrote {stem}.json  ({time.perf_counter()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
