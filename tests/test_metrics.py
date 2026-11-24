import numpy as np
import pytest
import scipy.sparse as sp

from acc.metrics import (StreamingCross, StreamingPaired, band_fractions,
                         decoder_cosine_match, topk_to_csr)

RNG = np.random.default_rng(1)


def test_streaming_paired_matches_numpy():
    a = RNG.standard_normal((5000, 32)).astype(np.float32)
    b = (0.8 * a + 0.2 * RNG.standard_normal((5000, 32))).astype(np.float32)
    s = StreamingPaired(32)
    for c0 in range(0, 5000, 700):  # uneven chunks
        s.update(a[c0 : c0 + 700], b[c0 : c0 + 700])
    ref = np.array([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(32)])
    np.testing.assert_allclose(s.pearson(), ref, atol=1e-6)


def test_streaming_paired_null_is_exactly_one():
    a = np.abs(RNG.standard_normal((3000, 16)).astype(np.float32))
    s = StreamingPaired(16)
    s.update(a, a.copy())
    r = s.pearson()
    assert np.all(r > 0.999999999)


def test_streaming_paired_zero_variance_is_nan():
    a = RNG.standard_normal((100, 3)).astype(np.float32)
    a[:, 1] = 0.0
    s = StreamingPaired(3)
    s.update(a, a.copy())
    r = s.pearson()
    assert np.isnan(r[1]) and not np.isnan(r[0])


def _sparse_acts(n, m, k, rng):
    idx = np.stack([rng.choice(m, size=k, replace=False) for _ in range(n)])
    vals = np.abs(rng.standard_normal((n, k))).astype(np.float32)
    return vals, idx


def test_cross_recovers_permutation():
    n, m, k = 4000, 64, 8
    vals, idx = _sparse_acts(n, m, k, RNG)
    a = topk_to_csr(vals, idx, m)
    perm = RNG.permutation(m)
    b = a[:, perm]  # column c of B is column perm[c] of A
    cross = StreamingCross(m, m)
    for c0 in range(0, n, 900):
        cross.update(a[c0 : c0 + 900].tocsr(), b[c0 : c0 + 900].tocsr())
    best_corr, best_idx = cross.match(row_chunk=17)
    fired = np.asarray((a > 0).sum(axis=0)).ravel() > 0
    # A's feature j appears in B at the column where perm == j
    expect = np.argsort(perm)
    assert np.all(best_corr[fired] > 0.999999)
    assert np.all(best_idx[fired] == expect[fired])


def test_cross_matches_dense_pearson():
    n, ma, mb = 2000, 24, 20
    a_dense = np.abs(RNG.standard_normal((n, ma))).astype(np.float32)
    a_dense[a_dense < 1.0] = 0  # sparsify
    b_dense = np.abs(RNG.standard_normal((n, mb))).astype(np.float32)
    b_dense[b_dense < 1.0] = 0
    cross = StreamingCross(ma, mb)
    cross.update(sp.csr_matrix(a_dense), sp.csr_matrix(b_dense))
    best_corr, best_idx = cross.match()
    with np.errstate(invalid="ignore"):
        full = np.corrcoef(a_dense.T, b_dense.T)[:ma, ma:]
    for j in range(ma):
        if np.isnan(best_corr[j]):
            continue
        assert best_corr[j] == pytest.approx(np.nanmax(full[j]), abs=1e-8)
        assert best_idx[j] == np.nanargmax(full[j])


def test_band_fractions():
    corr = np.array([0.95, 0.91, 0.7, 0.3, np.nan, 0.99])
    alive = np.array([True, True, True, True, True, False])
    b = band_fractions(corr, alive)
    assert b["n_alive"] == 5
    assert b["survived"] == pytest.approx(2 / 5)
    assert b["degraded"] == pytest.approx(1 / 5)
    assert b["damaged"] == pytest.approx(2 / 5)  # 0.3 and the nan
    total = b["survived"] + b["degraded"] + b["damaged"]
    assert total == pytest.approx(1.0)


def test_decoder_cosine_match_identity():
    w = RNG.standard_normal((40, 16)).astype(np.float32)
    best = decoder_cosine_match(w, -w[RNG.permutation(40)])
    np.testing.assert_allclose(best, 1.0, atol=1e-6)


def test_topk_to_csr_drops_zeros():
    vals = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    idx = np.array([[0, 3], [1, 2]])
    m = topk_to_csr(vals, idx, 5)
    assert m.nnz == 3
    assert m[0, 0] == 1.0 and m[0, 3] == 0.0
