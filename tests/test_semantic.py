import numpy as np
import pytest

from acc.metrics import jaccard, overlap_fraction, top_positions_per_feature

RNG = np.random.default_rng(11)


def test_top_positions_orders_by_value():
    feat = np.array([0, 0, 0, 1, 2, 2])
    val = np.array([1.0, 3.0, 2.0, 5.0, 0.5, 0.7], dtype=np.float32)
    pos = np.array([10, 11, 12, 20, 30, 31])
    tops = top_positions_per_feature(feat, val, pos, m=4, top_n=2)
    np.testing.assert_array_equal(tops[0], [11, 12])  # top-2 by value
    np.testing.assert_array_equal(tops[1], [20])
    np.testing.assert_array_equal(tops[2], [31, 30])
    assert len(tops[3]) == 0  # silent feature


def test_top_positions_random_agrees_with_bruteforce():
    n, m, top_n = 5000, 37, 8
    feat = RNG.integers(0, m, n)
    val = RNG.random(n).astype(np.float32)
    pos = RNG.permutation(100_000)[:n]
    tops = top_positions_per_feature(feat, val, pos, m, top_n)
    for f in range(m):
        mask = feat == f
        expect = pos[mask][np.argsort(-val[mask])][:top_n]
        np.testing.assert_array_equal(tops[f], expect)


def test_overlap_fraction():
    assert overlap_fraction(np.array([1, 2, 3]), np.array([2, 3, 4])) == \
        pytest.approx(2 / 3)
    assert overlap_fraction(np.array([1, 2]), np.array([3, 4])) == 0.0
    assert np.isnan(overlap_fraction(np.array([]), np.array([1])))


def test_jaccard_on_token_ids():
    a = np.array([5, 5, 7, 9])
    b = np.array([5, 7, 7, 11])
    assert jaccard(a, b) == pytest.approx(2 / 4)  # {5,7} over {5,7,9,11}
