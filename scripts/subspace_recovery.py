"""When a feature fails to reappear after retraining, is it gone, or re-carved?

WHY THIS RUNS. Camera-ready addition. The retraining
test counts a reference feature as lost when no single candidate feature
correlates with it above r = 0.9. Gerasimov et al. (arXiv 2606.12138) show
that across seeds, features that fail to reappear individually often lie in
subspaces that do reappear: the new dictionary cuts the same region along a
different basis. If that holds here, the count mixes two things: loss of
information and a change of basis. This script separates them.

THE MEASURE. For each alive reference feature i (firing rate > 1e-3 on the
matching tokens, as in the paper), take the m candidate features most
correlated with it on split A (the first 500,000 eval tokens, the same
tokens the paper's matching used). Fit an ordinary least-squares map from
those m candidate activations to feature i's activation on split A, then
score it on split B (eval tokens 500,000 onward, never used for matching or
fitting): held-out R^2_m. Feature i is "recovered at m" if R^2_m > 0.81,
which is the paper's r > 0.9 bar squared, so m = 1 reproduces the paper's
count on held-out tokens. S_m is the recovered fraction. Everything is
computed from float64 sufficient statistics (sums, cross-moments), so no
[N, d_sae] matrix is stored.

Also reported: the same with m random alive candidate features (null); the
held-out Pearson r of the paper's own best match; the fraction of each
reference decoder direction that lies in the span of its m partners'
decoder directions (random baseline ~ m/768); and Gerasimov's set-level
cross-subspace explained variance EV^(a->s)(r) (their App. F.1) for the
not-survived features.

PIPELINE CHECK (must pass before reading anything). Recomputing the paper's
best-match correlations from split-A moments must reproduce the eval-pair
npz: survived fraction within 0.05 pp. The script exits 1 otherwise.

PRE-REGISTERED READINGS, fixed 29 Sept 2026 before any condition ran.
Primary quantities: S_1 and S_4, gain G = S_4 - S_1. Single-seed
differences use the paper's 2 pp tie band.
  A. Seed turnover (fp32_s0 vs fp32_s1, fp32_s2). G >= 20 pp: basis
     ambiguity accounts for a large share of seed turnover at our site
     (Gerasimov transfers). G < 5 pp: seed turnover here is mostly not
     recoverable from a few partners; say so. 5-20: partial.
  B. The one-in-a-million ulp (sparseulp6, S_1 ~ 70%). S_4 >= 95%: the
     30-point loss is a change of basis over preserved structure, so the
     retraining count measures identifiability, not information. S_4 <
     85%: the perturbation changes what is represented. 85-95:
     indeterminate.
  C. The codec verdict at subspace level. std-int4 S_4 above the reseed
     S_4 by more than 2 pp: the verdict holds at subspace level. Within
     +-2 pp: std-int4 is indistinguishable from reseeding at subspace
     level, and the paper must say that its feature-level advantage does
     not carry to subspaces. Below: the verdict reverses at subspace level.
  D. Sanity. Naive int4 S_4 must sit more than 2 pp below std-int4 S_4;
     otherwise the measure cannot see a codec the paper already shows is
     damaging, and C is uninformative.
  E. Null. Random-partner S_4 < 1% in every condition; otherwise the
     measure is broken and nothing here is read.
Report every branch that fires.

Usage: PYTHONPATH=src python scripts/subspace_recovery.py \
          --root runs_m1 --ref fp32_s0 --cand fp32_s1
Writes <root>/eval/subspace_<ref>__<cand>.{json,npz}
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.cli import cache_dir, paths          # noqa: E402
from acc.metrics import band_fractions, topk_to_csr  # noqa: E402
from acc.sae import load_sae                  # noqa: E402
from acc.store import Store                   # noqa: E402
from acc.utils import get_device, write_json  # noqa: E402

SPLIT = 500_000          # eval-pair's eval_tokens: the matching tokens
ALIVE_RATE = 1e-3
BAR = 0.81               # r > 0.9, squared
MS = (1, 2, 4, 8)
EV_RANKS = (16, 64, 256)


class Moments:
    """Float64 sufficient statistics for one token split."""

    def __init__(self, m_ref, m_cand):
        self.n = 0
        self.sx = np.zeros(m_ref)
        self.sxx = np.zeros(m_ref)
        self.fx = np.zeros(m_ref, dtype=np.int64)
        self.sy = np.zeros(m_cand)
        self.fy = np.zeros(m_cand, dtype=np.int64)
        self.xy = np.zeros((m_ref, m_cand))
        self.yy = np.zeros((m_cand, m_cand))

    def update(self, a, b):
        self.n += a.shape[0]
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        self.sx += np.asarray(a64.sum(axis=0)).ravel()
        self.sxx += np.asarray(a64.multiply(a64).sum(axis=0)).ravel()
        self.fx += np.asarray((a > 0).sum(axis=0)).ravel()
        self.sy += np.asarray(b64.sum(axis=0)).ravel()
        self.fy += np.asarray((b > 0).sum(axis=0)).ravel()
        p = (a64.T @ b64).tocoo()
        self.xy[p.row, p.col] += p.data
        q = (b64.T @ b64).tocoo()
        self.yy[q.row, q.col] += q.data


def accumulate(ref, cand, store, device):
    A = Moments(ref.d_sae, cand.d_sae)
    B = Moments(ref.d_sae, cand.d_sae)
    seen = 0
    for s in range(len(store.shards)):
        x_all, _ = store.read_shard(s)
        for c0 in range(0, x_all.shape[0], 8192):
            x = x_all[c0 : c0 + 8192]
            g0 = seen + c0
            cut = int(np.clip(SPLIT - g0, 0, x.shape[0]))
            xt = torch.from_numpy(x).to(device)
            with torch.inference_mode():
                ra = ref.feature_acts(xt)
                ca = cand.feature_acts(xt)
            a = topk_to_csr(ra[0].cpu().numpy(), ra[1].cpu().numpy(), ref.d_sae)
            b = topk_to_csr(ca[0].cpu().numpy(), ca[1].cpu().numpy(), cand.d_sae)
            if cut > 0:
                A.update(a[:cut], b[:cut])
            if cut < x.shape[0]:
                B.update(a[cut:], b[cut:])
        seen += x_all.shape[0]
    return A, B


def corr_rows(M, rows, chunk=1024):
    """Pearson of ref rows `rows` against every cand feature, on M."""
    n = M.n
    mx, my = M.sx / n, M.sy / n
    vx = np.maximum(M.sxx / n - mx**2, 0)
    vy = np.maximum(np.diag(M.yy) / n - my**2, 0)
    sdy = np.sqrt(vy)
    out = np.empty((len(rows), M.xy.shape[1]))
    for r0 in range(0, len(rows), chunk):
        rr = rows[r0 : r0 + chunk]
        cov = M.xy[rr] / n - np.outer(mx[rr], my)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = cov / np.outer(np.sqrt(vx[rr]), sdy)
        c[:, sdy == 0] = -np.inf
        out[r0 : r0 + len(rr)] = c
    return out


def heldout_r2(A, B, rows, S):
    """OLS fit on A of ref feature rows[k] from cand features S[k] (n x m),
    scored on B. Returns R^2 on B, shape (n,)."""
    nA, nB = A.n, B.n
    m = S.shape[1]
    # split-A covariances
    mxA = A.sy[S] / nA                                   # (n, m)
    myA = A.sx[rows] / nA                                # (n,)
    CxxA = A.yy[S[:, :, None], S[:, None, :]] / nA - mxA[:, :, None] * mxA[:, None, :]
    cxyA = A.xy[rows[:, None], S] / nA - myA[:, None] * mxA
    tr = np.trace(CxxA, axis1=1, axis2=2) / m
    ridge = (1e-9 * tr + 1e-30)[:, None, None] * np.eye(m)[None]
    beta = np.linalg.solve(CxxA + ridge, cxyA[:, :, None])[:, :, 0]
    alpha = myA - np.einsum("nm,nm->n", beta, mxA)
    # split-B raw moments
    Ey = B.sx[rows] / nB
    Eyy = B.sxx[rows] / nB
    Ex = B.sy[S] / nB
    Exx = B.yy[S[:, :, None], S[:, None, :]] / nB
    Exy = B.xy[rows[:, None], S] / nB
    mse = (Eyy - 2 * alpha * Ey - 2 * np.einsum("nm,nm->n", beta, Exy)
           + alpha**2 + 2 * alpha * np.einsum("nm,nm->n", beta, Ex)
           + np.einsum("nm,nmk,nk->n", beta, Exx, beta))
    var = Eyy - Ey**2
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1 - mse / var


def heldout_pearson(B, rows, cols):
    n = B.n
    mx, my = B.sx[rows] / n, B.sy[cols] / n
    vx = np.maximum(B.sxx[rows] / n - mx**2, 0)
    vy = np.maximum(B.yy[cols, cols] / n - my**2, 0)
    cov = B.xy[rows, cols] / n - mx * my
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vx * vy)


def span_fraction(Wr, Wc, rows, S):
    """||proj of unit ref decoder row onto span(cand rows S)||^2."""
    d = Wr[rows] / np.linalg.norm(Wr[rows], axis=1, keepdims=True)
    V = Wc[S]                                            # (n, m, 768)
    G = np.einsum("nmd,nkd->nmk", V, V)
    b = np.einsum("nmd,nd->nm", V, d)
    m = S.shape[1]
    G = G + 1e-12 * np.eye(m)[None]
    sol = np.linalg.solve(G, b[:, :, None])[:, :, 0]
    return np.einsum("nm,nm->n", b, sol)


def ev_cross(Xa, Xs, r):
    """Gerasimov App. F.1: ||P_a(r) X_s||_F^2 / ||X_s||_F^2, P_a from the
    top-r left singular vectors of Xa (columns = unit decoder vectors)."""
    U, _, _ = np.linalg.svd(Xa, full_matrices=False)
    P = U[:, :r]
    return float(np.square(P.T @ Xs).sum() / np.square(Xs).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--ref", default="fp32_s0")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = args.device or get_device()
    t0 = time.perf_counter()
    P = paths(args.root)
    ref, _ = load_sae(P["saes"] / args.ref, device)
    cand, _ = load_sae(P["saes"] / args.cand, device)
    store = Store(cache_dir(args.root, "fp32", "eval"))
    A, B = accumulate(ref, cand, store, device)
    print(f"  moments: A n={A.n}  B n={B.n}  ({time.perf_counter()-t0:.0f}s)")
    assert A.n == SPLIT, A.n

    # ---- pipeline check against the paper's eval-pair npz
    pair = np.load(P["eval"] / f"pair_{args.ref}__{args.cand}.npz")
    alive = A.fx / A.n > ALIVE_RATE
    rows = np.flatnonzero(alive)
    C = corr_rows(A, rows)
    best_idx = np.argmax(C, axis=1)
    best_corr = C[np.arange(len(rows)), best_idx]
    full = np.full(ref.d_sae, np.nan)
    full[rows] = best_corr
    ours = band_fractions(full, alive)
    theirs = band_fractions(pair["best_corr"], pair["alive"])
    check = {
        "alive_masks_equal": bool(np.array_equal(alive, pair["alive"])),
        "survived_ours": ours["survived"], "survived_npz": theirs["survived"],
        "max_abs_corr_diff": float(np.nanmax(np.abs(best_corr - pair["best_corr"][rows]))),
        "best_idx_agree": float((best_idx == pair["best_idx"][rows]).mean()),
    }
    check["pass"] = bool(check["alive_masks_equal"]
                         and abs(ours["survived"] - theirs["survived"]) < 5e-4)
    print(f"  check: survived {ours['survived']*100:.2f} vs npz "
          f"{theirs['survived']*100:.2f}, max|dr| {check['max_abs_corr_diff']:.2e}, "
          f"idx agree {check['best_idx_agree']:.4f} -> "
          f"{'PASS' if check['pass'] else 'FAIL'}")

    # ---- partners: top-8 cand features by split-A correlation
    top = np.argpartition(-C, max(MS), axis=1)[:, : max(MS)]
    order = np.take_along_axis(
        top, np.argsort(-np.take_along_axis(C, top, axis=1), axis=1), axis=1)
    assert (order[:, 0] == best_idx).all()
    cand_alive = np.flatnonzero(A.fy / A.n > ALIVE_RATE)
    rng = np.random.default_rng(args.seed)
    rand = np.stack([rng.choice(cand_alive, max(MS), replace=False)
                     for _ in rows])
    survived = best_corr > 0.9

    r2, r2_null, span, span_null = {}, {}, {}, {}
    Wr = ref.W_dec.detach().cpu().numpy().astype(np.float64)
    Wc = cand.W_dec.detach().cpu().numpy().astype(np.float64)
    for m in MS:
        r2[m] = heldout_r2(A, B, rows, order[:, :m])
        r2_null[m] = heldout_r2(A, B, rows, rand[:, :m])
        span[m] = span_fraction(Wr, Wc, rows, order[:, :m])
        span_null[m] = span_fraction(Wr, Wc, rows, rand[:, :m])
    r_b = heldout_pearson(B, rows, best_idx)

    def frac(v, bar=BAR, mask=None):
        v = v if mask is None else v[mask]
        return float(np.mean(np.nan_to_num(v, nan=-np.inf) > bar)) if v.size else float("nan")

    summary = {"n_alive": int(len(rows)),
               "survived_split_A": float(survived.mean()),
               "survived_heldout_r_B": frac(r_b, 0.9)}
    for m in MS:
        summary[f"S_{m}"] = frac(r2[m])
        summary[f"S_{m}_null"] = frac(r2_null[m])
        summary[f"median_R2_{m}"] = float(np.nanmedian(r2[m]))
        summary[f"recovered_{m}_among_not_survived"] = frac(r2[m], mask=~survived)
        for bar in (0.5, 0.9):
            summary[f"S_{m}_bar{bar}"] = frac(r2[m], bar)
        summary[f"span_median_{m}"] = float(np.median(span[m]))
        summary[f"span_median_{m}_null"] = float(np.median(span_null[m]))
        summary[f"span_median_{m}_not_survived"] = (
            float(np.median(span[m][~survived])) if (~survived).any() else float("nan"))
    summary["G_4"] = summary["S_4"] - summary["S_1"]

    # ---- Gerasimov set-level EV for not-survived features
    matched_c = np.unique(best_idx[survived])
    U_c = np.setdiff1d(cand_alive, matched_c)
    Xr_U = (Wr[rows[~survived]] / np.linalg.norm(Wr[rows[~survived]], axis=1, keepdims=True)).T
    Xr_S = (Wr[rows[survived]] / np.linalg.norm(Wr[rows[survived]], axis=1, keepdims=True)).T
    Xc_U = (Wc[U_c] / np.linalg.norm(Wc[U_c], axis=1, keepdims=True)).T
    ev = {}
    if Xr_U.shape[1] >= max(EV_RANKS) and Xc_U.shape[1] > 0:
        for r in EV_RANKS:
            ev[str(r)] = {
                "within_ref": ev_cross(Xr_U, Xr_U, r),
                "ref_to_cand": ev_cross(Xr_U, Xc_U, r),
                "cand_to_ref": ev_cross(Xc_U, Xr_U, r),
                "stable_ref_to_cand_U": (ev_cross(Xr_S, Xc_U, r)
                                         if Xr_S.shape[1] >= r else None),
                "random_baseline": r / Wr.shape[1],
            }
    summary["n_not_survived_ref"] = int((~survived).sum())
    summary["n_unmatched_cand"] = int(len(U_c))

    out = {"ref": args.ref, "cand": args.cand, "split_tokens": SPLIT,
           "n_A": A.n, "n_B": B.n, "alive_rate": ALIVE_RATE, "bar_R2": BAR,
           "ms": list(MS), "check": check, **summary, "ev_set_level": ev,
           "seconds": time.perf_counter() - t0}
    stem = P["eval"] / f"subspace_{args.ref}__{args.cand}"
    write_json(stem.with_suffix(".json"), out)
    np.savez_compressed(
        stem.with_suffix(".npz"), rows=rows, best_idx=best_idx,
        best_corr_A=best_corr, r_B=r_b, partners=order,
        **{f"r2_{m}": r2[m] for m in MS},
        **{f"r2_null_{m}": r2_null[m] for m in MS},
        **{f"span_{m}": span[m] for m in MS},
        **{f"span_null_{m}": span_null[m] for m in MS})
    print(f"  {args.cand}: S_1 {summary['S_1']*100:.1f}  S_2 {summary['S_2']*100:.1f}  "
          f"S_4 {summary['S_4']*100:.1f}  S_8 {summary['S_8']*100:.1f}  "
          f"(null S_4 {summary['S_4_null']*100:.2f})  "
          f"recovered@4 among lost {summary['recovered_4_among_not_survived']*100:.1f}%")
    print(f"  wrote {stem}.json  ({time.perf_counter()-t0:.0f}s)")
    if not check["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
