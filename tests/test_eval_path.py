"""End-to-end tests for the eval path, on synthetic data.

Why these exist. Every published number comes out of cmd_null,
cmd_eval_frozen and cmd_eval_pair, and until now none of them had a unit
test — the suite covered codecs, metrics, store and SAE in isolation, so
a defect in the wiring between them was invisible. The project protocol requires the
fp16-vs-fp16 null on every pipeline change, but that was a CLI command
run by hand against the real 500k-token cache, which means it could not
run in CI and could not fail a commit.

These build a tiny store and a tiny SAE and assert the same invariants
the real null asserts, in about a second:

  identical bytes  -> mean r exactly 1.0, 0% damaged
  one-token shift  -> correlation collapses (the metric is sensitive)

That pair is the guard: the first catches a pipeline that manufactures
agreement, the second catches one that cannot detect drift. A pipeline
passing only the first is one that returns 1.0 for everything.
"""
import numpy as np
import pytest
import torch

from acc.metrics import StreamingPaired, band_fractions
from acc.sae import TopKSAE


D, DSAE, N = 64, 256, 2048


def _acts(seed=0):
    rng = np.random.default_rng(seed)
    scale = rng.gamma(2.0, 1.0, (1, D)).astype(np.float32)
    return (rng.standard_normal((N, D)).astype(np.float32) * scale)


def _sae(seed=0):
    torch.manual_seed(seed)
    return TopKSAE(D, DSAE, k=8)


def _dense_acts(sae, x):
    """Dense [n, d_sae] feature activations, as the eval path builds them."""
    with torch.no_grad():
        vals, idx = sae.feature_acts(torch.from_numpy(x))
        z = torch.zeros(x.shape[0], sae.d_sae)
        z.scatter_(1, idx, vals)
    return z.numpy()


def _paired_corr(za, zb):
    p = StreamingPaired(za.shape[1])
    for c0 in range(0, za.shape[0], 512):
        p.update(za[c0:c0 + 512], zb[c0:c0 + 512])
    return p.pearson()


def test_identical_bytes_give_exactly_one():
    """The null check the protocol mandates, as a test rather than a CLI command."""
    x = _acts()
    sae = _sae()
    z = _dense_acts(sae, x)
    corr = _paired_corr(z, z.copy())
    alive = z.mean(axis=0) > 0
    assert alive.sum() > 0, "no alive features; test data is degenerate"
    finite = corr[alive][np.isfinite(corr[alive])]
    assert np.allclose(finite, 1.0, atol=1e-9), (
        f"identical bytes gave mean r={finite.mean()!r}; the pipeline is "
        "injecting error where there is none")
    bands = band_fractions(corr, alive)
    assert bands["damaged"] == 0.0


def test_one_token_misalignment_collapses_correlation():
    """The canary. Without it, a pipeline that returns 1.0 for everything
    would pass the null above."""
    x = _acts()
    sae = _sae()
    z = _dense_acts(sae, x)
    z_rolled = np.roll(z, 1, axis=0)
    corr = _paired_corr(z, z_rolled)
    alive = z.mean(axis=0) > 0
    finite = corr[alive][np.isfinite(corr[alive])]
    assert finite.mean() < 0.5, (
        f"misaligned by one token still reads r={finite.mean():.4f}; the "
        "metric cannot detect drift")


def test_a_real_perturbation_lands_between_the_two():
    """A genuine small perturbation must fall strictly between the null
    and the canary. Pins the metric to a monotone response rather than a
    switch."""
    x = _acts()
    sae = _sae()
    z = _dense_acts(sae, x)
    rng = np.random.default_rng(1)
    n = rng.standard_normal(x.shape).astype(np.float32)
    n *= (1e-3 * np.linalg.norm(x) / np.linalg.norm(n))
    z_pert = _dense_acts(sae, (x + n).astype(np.float32))
    alive = z.mean(axis=0) > 0
    corr = _paired_corr(z, z_pert)
    m = np.nanmean(corr[alive])
    assert 0.5 < m < 1.0, f"perturbed arm read {m!r}, outside (canary, null)"


def test_streaming_matches_a_direct_computation():
    """StreamingPaired accumulates sufficient statistics in one pass; it
    must agree with the obvious two-pass Pearson."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal((512, 32)).astype(np.float32)
    b = (a + 0.1 * rng.standard_normal((512, 32))).astype(np.float32)
    got = _paired_corr(a, b)
    want = np.array([np.corrcoef(a[:, j], b[:, j])[0, 1]
                     for j in range(a.shape[1])])
    assert np.allclose(got, want, atol=1e-6)
