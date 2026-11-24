"""Precision-adequacy tests for the distortion metrics.

These exist because a float32 cosine shipped in the released artifacts.
A d=768 dot product of two nearly-identical float32 vectors saturates at
about 1 - 5.4e-7, so every high-fidelity arm reported the same cosine
regardless of its true distortion: four noise rungs spanning rel-L2 1e-4
to 1e-7 all recorded cos_min = 0.9999993443489075, identical to sixteen
digits, and 8 of 24 arms recorded cos_mean of exactly 1.0. The 86 tests
in place at the time all passed, because none of them compared a metric
against a higher-precision reference.

The rule these encode: a metric that must resolve differences far below
float32 epsilon has to accumulate in float64, and the test that proves it
must use an independent float64 reference rather than the metric itself.
"""
import numpy as np
import pytest

from acc.metrics import direct_distortion_arrays

D = 768
N = 4096


def _reference(x, x_hat):
    """Independent float64 implementation. Deliberately not vectorised
    the same way as the metric under test."""
    xd = np.asarray(x, dtype=np.float64)
    xh = np.asarray(x_hat, dtype=np.float64)
    rel = np.array([np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)
                    for a, b in zip(xd, xh)])
    cos = np.array([float(np.dot(a, b) /
                          (max(np.linalg.norm(a), 1e-12) *
                           max(np.linalg.norm(b), 1e-12)))
                    for a, b in zip(xd, xh)])
    return rel, cos


def _data(seed=0):
    rng = np.random.default_rng(seed)
    # heavy-tailed per-channel scale, like real residual-stream activations
    scale = rng.gamma(2.0, 1.0, (1, D)).astype(np.float32)
    return (rng.standard_normal((N, D)).astype(np.float32) * scale)


@pytest.mark.parametrize("rel_target", [1e-2, 1e-4, 1e-6, 1e-7])
def test_cosine_resolves_perturbations_below_float32_epsilon(rel_target):
    """The cosine must track the perturbation, not saturate at the f32 floor.

    At rel 1e-6 and 1e-7 a float32 accumulation returns a constant, which
    is precisely how four distinct noise rungs came to share a cos_min.
    """
    x = _data()
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(x.shape).astype(np.float32)
    noise *= (rel_target * np.linalg.norm(x) / np.linalg.norm(noise))
    x_hat = (x + noise).astype(np.float32)

    _, cos = direct_distortion_arrays(x, x_hat)
    _, cos_ref = _reference(x, x_hat)

    # agrees with an independent float64 reference
    assert np.allclose(cos, cos_ref, atol=1e-12), (
        f"cosine disagrees with float64 reference at rel={rel_target}")
    # and is strictly below 1: it must not have saturated
    assert cos.mean() < 1.0, (
        f"cos_mean saturated to {cos.mean()!r} at rel={rel_target}; "
        "this is the float32 failure the metric was fixed for")


def test_cosine_orders_two_arms_a_decade_apart():
    """Two arms one decade apart in distortion must not tie.

    The released artifacts had four rungs spanning four decades reporting
    an identical cosine, so ordering — the thing the paper's rho claims
    rest on — was not resolvable from that column.
    """
    x = _data()
    rng = np.random.default_rng(2)
    out = []
    for rel in (1e-6, 1e-7):
        n = rng.standard_normal(x.shape).astype(np.float32)
        n *= (rel * np.linalg.norm(x) / np.linalg.norm(n))
        _, cos = direct_distortion_arrays(x, (x + n).astype(np.float32))
        out.append(cos.mean())
    # cosine is SECOND order in the perturbation: cos ~ 1 - rel^2/2, so a
    # decade in rel is two decades in (1 - cos). At rel 1e-6 that is
    # 1 - 5e-13, which float32 (eps ~ 1e-7) cannot represent at all and
    # rounds to exactly 1.0 -- the whole reason this metric needs float64.
    assert out[0] < out[1], "coarser perturbation must give the lower cosine"
    gap = out[1] - out[0]
    assert gap > 1e-14, f"arms tied at float64 resolution: {out}"
    # and confirm float32 really would have tied them
    f32 = [np.float32(v) for v in out]
    assert f32[0] == f32[1] == np.float32(1.0), (
        "expected float32 to collapse both arms to 1.0; if this fails the "
        "regression guard below is no longer meaningful")


def test_relative_l2_matches_float64_reference():
    """rel-L2 was always sound (x - x_hat is exact by Sterbenz for nearby
    values) — pin that so a future refactor cannot quietly break it."""
    x = _data()
    rng = np.random.default_rng(3)
    n = rng.standard_normal(x.shape).astype(np.float32)
    n *= (1e-5 * np.linalg.norm(x) / np.linalg.norm(n))
    rel, _ = direct_distortion_arrays(x, (x + n).astype(np.float32))
    rel_ref, _ = _reference(x, (x + n).astype(np.float32))
    assert np.allclose(rel, rel_ref, rtol=1e-9)


def test_identical_inputs_give_exactly_one():
    x = _data()
    rel, cos = direct_distortion_arrays(x, x.copy())
    assert np.all(rel == 0.0)
    assert np.allclose(cos, 1.0, atol=1e-12)
