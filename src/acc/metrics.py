"""Streaming feature-fidelity metrics.

Two comparison modes:

1. Paired (StreamingPaired): the same feature index on both sides, for
   frozen-SAE distortion (Duan-style) and the fp16-vs-fp16 null check.
   Accumulates per-feature sufficient statistics in float64 — O(d_sae)
   memory, never materializes [N, d_sae].

2. Cross (StreamingCross): all-pairs statistics between two different SAEs
   whose feature indices don't correspond (independently trained). Uses the
   TopK sparsity: per-chunk sparse products accumulate into a dense float64
   [m_ref, m_cand] raw-moment matrix; correlations and best matches are
   computed row-chunked at the end.

Banding (Duan): survived > 0.9, degraded 0.5–0.9, damaged < 0.5, restricted
to reference features with firing rate > min_rate.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

SURVIVED = 0.9
DAMAGED = 0.5


class StreamingPaired:
    def __init__(self, m: int):
        self.m = m
        self.n = 0
        self.sx = np.zeros(m)
        self.sy = np.zeros(m)
        self.sxx = np.zeros(m)
        self.syy = np.zeros(m)
        self.sxy = np.zeros(m)
        self.fire_x = np.zeros(m, dtype=np.int64)
        self.fire_y = np.zeros(m, dtype=np.int64)

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        assert a.shape == b.shape and a.shape[1] == self.m
        self.n += a.shape[0]
        self.sx += a.sum(axis=0, dtype=np.float64)
        self.sy += b.sum(axis=0, dtype=np.float64)
        self.sxx += np.einsum("ij,ij->j", a, a, dtype=np.float64)
        self.syy += np.einsum("ij,ij->j", b, b, dtype=np.float64)
        self.sxy += np.einsum("ij,ij->j", a, b, dtype=np.float64)
        self.fire_x += (a > 0).sum(axis=0)
        self.fire_y += (b > 0).sum(axis=0)

    def pearson(self) -> np.ndarray:
        n = self.n
        cov = self.sxy / n - (self.sx / n) * (self.sy / n)
        vx = np.maximum(self.sxx / n - (self.sx / n) ** 2, 0)
        vy = np.maximum(self.syy / n - (self.sy / n) ** 2, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cov / np.sqrt(vx * vy)
        r[(vx == 0) | (vy == 0)] = np.nan
        return r

    def firing_rate_x(self) -> np.ndarray:
        return self.fire_x / max(self.n, 1)


class StreamingCross:
    def __init__(self, m_ref: int, m_cand: int):
        self.m_ref, self.m_cand = m_ref, m_cand
        self.n = 0
        self.sx = np.zeros(m_ref)
        self.sxx = np.zeros(m_ref)
        self.sy = np.zeros(m_cand)
        self.syy = np.zeros(m_cand)
        self.sxy = np.zeros((m_ref, m_cand))
        self.fire_x = np.zeros(m_ref, dtype=np.int64)

    def update(self, a_sp: sp.csr_matrix, b_sp: sp.csr_matrix) -> None:
        assert a_sp.shape[0] == b_sp.shape[0]
        self.n += a_sp.shape[0]
        a64 = a_sp.astype(np.float64)
        b64 = b_sp.astype(np.float64)
        self.sx += np.asarray(a64.sum(axis=0)).ravel()
        self.sy += np.asarray(b64.sum(axis=0)).ravel()
        self.sxx += np.asarray(a64.multiply(a64).sum(axis=0)).ravel()
        self.syy += np.asarray(b64.multiply(b64).sum(axis=0)).ravel()
        prod = (a64.T @ b64).tocoo()
        self.sxy[prod.row, prod.col] += prod.data
        self.fire_x += np.asarray((a_sp > 0).sum(axis=0)).ravel()

    def match(self, row_chunk: int = 1024):
        """For each ref feature, the best-correlated cand feature.

        Returns (best_corr [m_ref], best_idx [m_ref]); nan for zero-variance
        ref features. Cross-terms with zero co-firing still get exact Pearson
        because the raw moment matrix includes the (dense) mean correction.
        """
        n = self.n
        mx, my = self.sx / n, self.sy / n
        vx = np.maximum(self.sxx / n - mx**2, 0)
        vy = np.maximum(self.syy / n - my**2, 0)
        sd_y = np.sqrt(vy)
        best_corr = np.full(self.m_ref, np.nan)
        best_idx = np.zeros(self.m_ref, dtype=np.int64)
        valid_y = sd_y > 0
        for r0 in range(0, self.m_ref, row_chunk):
            r1 = min(r0 + row_chunk, self.m_ref)
            cov = self.sxy[r0:r1] / n - np.outer(mx[r0:r1], my)
            sd_x = np.sqrt(vx[r0:r1])
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = cov / np.outer(sd_x, sd_y)
            corr[:, ~valid_y] = -np.inf
            idx = np.argmax(corr, axis=1)
            vals = corr[np.arange(r1 - r0), idx]
            vals[sd_x == 0] = np.nan
            best_corr[r0:r1] = vals
            best_idx[r0:r1] = idx
        return best_corr, best_idx

    def firing_rate_x(self) -> np.ndarray:
        return self.fire_x / max(self.n, 1)


def topk_to_csr(vals: np.ndarray, idx: np.ndarray, m: int) -> sp.csr_matrix:
    """Build csr [n, m] from TopK (vals [n,k], idx [n,k]). Zero vals dropped."""
    n, k = vals.shape
    rows = np.repeat(np.arange(n), k)
    mat = sp.csr_matrix(
        (vals.ravel().astype(np.float64), (rows, idx.ravel().astype(np.int64))),
        shape=(n, m),
    )
    mat.eliminate_zeros()
    return mat


def band_fractions(corr: np.ndarray, alive_mask: np.ndarray) -> dict:
    """Survived/degraded/damaged fractions over alive reference features.

    NaN correlations among alive features count as damaged (a feature whose
    match has zero variance is as gone as one with corr < 0.5) and are also
    reported separately.
    """
    c = corr[alive_mask]
    n = c.size
    if n == 0:
        return {"n_alive": 0}
    nan = np.isnan(c)
    return {
        "n_alive": int(n),
        "mean_corr": float(np.nanmean(c)) if (~nan).any() else float("nan"),
        "median_corr": float(np.nanmedian(c)) if (~nan).any() else float("nan"),
        "survived": float(((c > SURVIVED) & ~nan).sum() / n),
        "degraded": float(((c >= DAMAGED) & (c <= SURVIVED) & ~nan).sum() / n),
        "damaged": float(((c < DAMAGED) | nan).sum() / n),
        "nan_frac": float(nan.sum() / n),
        "corr_deciles": [
            float(v) for v in np.nanpercentile(c[~nan], np.arange(0, 101, 10))
        ] if (~nan).any() else [],
    }


def decoder_cosine_match(w_ref: np.ndarray, w_cand: np.ndarray,
                         row_chunk: int = 2048):
    """Best |cosine| match of decoder directions (Paulo & Belrose style)."""
    a = w_ref / np.linalg.norm(w_ref, axis=1, keepdims=True).clip(min=1e-12)
    b = w_cand / np.linalg.norm(w_cand, axis=1, keepdims=True).clip(min=1e-12)
    best = np.empty(a.shape[0])
    for r0 in range(0, a.shape[0], row_chunk):
        sims = a[r0 : r0 + row_chunk] @ b.T
        best[r0 : r0 + row_chunk] = np.abs(sims).max(axis=1)
    return best


def top_positions_per_feature(feat: np.ndarray, val: np.ndarray,
                              pos: np.ndarray, m: int, top_n: int):
    """From firing triples (feature, value, position), the top-`top_n`
    activating positions per feature, value-descending. Returns a list of
    m arrays (possibly shorter than top_n, empty for silent features)."""
    order = np.lexsort((-val, feat))
    f_sorted = feat[order]
    starts = np.searchsorted(f_sorted, np.arange(m), side="left")
    ends = np.searchsorted(f_sorted, np.arange(m), side="right")
    return [pos[order[s : min(s + top_n, e)]] for s, e in zip(starts, ends)]


def overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """|A ∩ B| / min(|A|, |B|) over position sets; nan if either empty."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return len(np.intersect1d(a, b)) / min(len(a), len(b))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    ua, ub = np.unique(a), np.unique(b)
    return len(np.intersect1d(ua, ub)) / len(np.union1d(ua, ub))


def direct_distortion_arrays(x: np.ndarray, x_hat: np.ndarray):
    """Per-token relative L2 error and cosine similarity.

    The cosine MUST accumulate in float64. A d=768 dot product of two
    nearly-identical float32 vectors saturates at about 1 - 5.4e-7, so in
    float32 every high-fidelity arm reports the same cosine regardless of
    its true distortion: four noise rungs spanning 1e-4 to 1e-7 all
    recorded cos_min = 0.9999993443489075, identical to sixteen digits,
    and eight of twenty-four arms recorded cos_mean of exactly 1.0. The
    relative L2 is unaffected (x - x_hat is exact by Sterbenz for nearby
    values), which is why the rate-distortion figures were sound while
    the cosine column was not.
    """
    xd = x.astype(np.float64, copy=False)
    xh = x_hat.astype(np.float64, copy=False)
    num = np.linalg.norm(xd - xh, axis=1)
    den = np.linalg.norm(xd, axis=1).clip(min=1e-12)
    cos = np.einsum("ij,ij->i", xd, xh) / (
        den * np.linalg.norm(xh, axis=1).clip(min=1e-12)
    )
    return num / den, cos


def direct_distortion(x: np.ndarray, x_hat: np.ndarray) -> dict:
    """Raw activation-space distortion of a codec, no SAE involved.
    Order statistics (p99, min) are only global if called on ALL tokens at
    once — for shard-streamed eval use direct_distortion_arrays and
    aggregate the concatenated arrays."""
    rel, cos = direct_distortion_arrays(x, x_hat)
    return {
        "rel_l2_mean": float(rel.mean()),
        "rel_l2_p99": float(np.percentile(rel, 99)),
        "cos_mean": float(cos.mean()),
        "cos_min": float(cos.min()),
    }
