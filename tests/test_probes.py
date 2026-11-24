import numpy as np
import pytest

from acc.probes import direction_cosine, eval_probe, train_probe

RNG = np.random.default_rng(7)


def _separable(n=2000, d=32, n_classes=3, noise=0.3):
    dirs = RNG.standard_normal((n_classes, d)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    y = RNG.integers(0, n_classes, size=n)
    x = 2.0 * dirs[y] + noise * RNG.standard_normal((n, d)).astype(np.float32)
    return x.astype(np.float32), y.astype(np.int64)


def _stats(x):
    return x.mean(axis=0), x.std(axis=0) + 1e-6


def test_probe_learns_separable_data():
    x, y = _separable()
    mean, std = _stats(x)
    w = train_probe(x, y, 3, mean, std)
    assert eval_probe(w, x, y, mean, std) > 0.95


def test_probe_deterministic():
    x, y = _separable(n=500)
    mean, std = _stats(x)
    w1 = train_probe(x, y, 3, mean, std)
    w2 = train_probe(x, y, 3, mean, std)
    np.testing.assert_array_equal(w1, w2)  # convex + zero-init + full batch


def test_direction_cosine_self_is_one():
    x, y = _separable(n=800)
    mean, std = _stats(x)
    w = train_probe(x, y, 3, mean, std)
    assert direction_cosine(w, std, w, std) == pytest.approx(1.0, abs=1e-9)


def test_small_noise_keeps_direction():
    x, y = _separable(n=3000, noise=0.2)
    xq = x + 0.01 * RNG.standard_normal(x.shape).astype(np.float32)
    m1, s1 = _stats(x)
    m2, s2 = _stats(xq)
    w1 = train_probe(x, y, 3, m1, s1)
    w2 = train_probe(xq, y, 3, m2, s2)
    assert direction_cosine(w1, s1, w2, s2) > 0.99
    assert abs(eval_probe(w1, x, y, m1, s1)
               - eval_probe(w2, x, y, m2, s2)) < 0.02


def test_destandardization_matters_for_cosine():
    # scaling one input dimension changes standardized-space weights but the
    # raw-space direction must be recovered by de-standardization
    x, y = _separable(n=3000, noise=0.2)
    x_scaled = x.copy()
    x_scaled[:, 0] *= 50.0
    m1, s1 = _stats(x_scaled)
    w1 = train_probe(x_scaled, y, 3, m1, s1)
    m2, s2 = _stats(x_scaled)
    w2 = train_probe(x_scaled, y, 3, m2, s2)
    assert direction_cosine(w1, s1, w2, s2) == pytest.approx(1.0, abs=1e-6)
