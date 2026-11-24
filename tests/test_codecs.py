import numpy as np
import pytest

from acc.codecs import (get_codec, pack_uint, theoretical_bytes_per_token,
                        unpack_uint, Codec)

RNG = np.random.default_rng(0)
D = 768


def heavy_tailed(n=2048, d=D):
    """Synthetic activations with outlier channels, like real residuals."""
    x = RNG.standard_normal((n, d)).astype(np.float32)
    x[:, 17] *= 40.0
    x[:, 401] *= 25.0
    return x


@pytest.mark.parametrize("bits", [4, 6])
def test_pack_roundtrip(bits):
    q = RNG.integers(0, 2**bits, size=(100, D), dtype=np.uint8)
    packed = pack_uint(q, bits)
    assert packed.shape == (100, D * bits // 8)
    np.testing.assert_array_equal(unpack_uint(packed, bits, D), q)


@pytest.mark.parametrize("bits", [4, 6])
def test_fast_pack_matches_bitpath_reference(bits):
    from acc.codecs import _pack_uint_bitpath, _unpack_uint_bitpath
    q = RNG.integers(0, 2**bits, size=(64, D), dtype=np.uint8)
    fast = pack_uint(q, bits)
    ref = _pack_uint_bitpath(q, bits)
    np.testing.assert_array_equal(fast, ref)  # on-disk format unchanged
    np.testing.assert_array_equal(unpack_uint(ref, bits, D),
                                  _unpack_uint_bitpath(ref, bits, D))


def test_fp32_exact():
    x = heavy_tailed()
    c = get_codec("fp32")
    np.testing.assert_array_equal(c.decode(c.encode(x)), x)


def test_fp16_error_bound():
    x = heavy_tailed()
    c = get_codec("fp16")
    err = np.abs(c.decode(c.encode(x)) - x)
    assert (err <= np.abs(x) * 2**-10 + 1e-7).all()


def test_bf16_matches_truncation_semantics():
    c = get_codec("bf16")
    x = np.array([[1.0, -2.5, 3.14159e5, 1e-20]], dtype=np.float32)
    dec = c.decode(c.encode(x))
    assert (np.abs(dec - x) <= np.abs(x) * 2**-8).all()
    exact = np.array([[1.0, -2.5]], dtype=np.float32)
    np.testing.assert_array_equal(c.decode(c.encode(exact)), exact)


@pytest.mark.parametrize("name", ["int8", "int6", "int4"])
def test_quant_error_bounded_by_scale(name):
    x = heavy_tailed()
    c = get_codec(name)
    dec = c.decode(c.encode(x))
    scale = np.abs(x).max(axis=1) / c.levels
    assert (np.abs(dec - x) <= scale[:, None] * 0.5 + 1e-6).all()


def test_quant_zero_row_no_nan():
    x = np.zeros((4, D), dtype=np.float32)
    for name in ["int8", "int4", "fp16"]:
        dec = get_codec(name).decode(get_codec(name).encode(x))
        assert np.isfinite(dec).all()
        np.testing.assert_allclose(dec, 0.0, atol=1e-9)


def test_outlier_codec_protects_outlier_channels():
    x = heavy_tailed()
    plain = get_codec("int4")
    outl = get_codec("int4-o8")
    outl.fit(x)
    assert 17 in outl.outlier_idx and 401 in outl.outlier_idx
    dec_plain = plain.decode(plain.encode(x))
    dec_outl = outl.decode(outl.encode(x))
    # non-outlier channels must get better, since scales shrink drastically
    mask = np.ones(D, bool)
    mask[outl.outlier_idx] = False
    err_plain = np.abs(dec_plain[:, mask] - x[:, mask]).mean()
    err_outl = np.abs(dec_outl[:, mask] - x[:, mask]).mean()
    assert err_outl < err_plain / 5
    # outlier channels themselves are fp16-exact-ish
    err_out_ch = np.abs(dec_outl[:, ~mask] - x[:, ~mask])
    assert (err_out_ch <= np.abs(x[:, ~mask]) * 2**-10 + 1e-6).all()


def test_outlier_state_roundtrip():
    x = heavy_tailed()
    c = get_codec("int8-o8")
    c.fit(x)
    payload = c.encode(x)
    c2 = get_codec("int8-o8")
    c2.load_state({**c.state(), "d": D}, c.state_arrays())
    np.testing.assert_array_equal(c.decode(payload), c2.decode(payload))


def test_pca_full_rank_near_exact():
    x = heavy_tailed(512)
    c = get_codec(f"pca{D}")
    c.fit(x)
    dec = c.decode(c.encode(x))
    rel = np.linalg.norm(dec - x, axis=1) / np.linalg.norm(x, axis=1)
    assert rel.max() < 0.01  # limited only by fp16 coefficient storage


def test_pca_beats_rp_at_same_rank():
    x = heavy_tailed(4096)
    err = {}
    for name in ["pca256", "rp256"]:
        c = get_codec(name)
        c.fit(x)
        dec = c.decode(c.encode(x))
        err[name] = float(np.linalg.norm(dec - x) / np.linalg.norm(x))
    assert err["pca256"] < err["rp256"]


@pytest.mark.parametrize("name,expect", [
    ("fp32", 3072), ("fp16", 1536), ("bf16", 1536),
    ("int8", 772), ("int6", 580), ("int4", 388),
    ("int8-o8", 780), ("pca256", 512),
])
def test_measured_bytes_match_theory(name, expect):
    assert theoretical_bytes_per_token(name, D) == expect
    x = heavy_tailed(256)
    c = get_codec(name)
    c.fit(x)
    measured = Codec.payload_nbytes(c.encode(x)) / 256
    assert measured == pytest.approx(expect, abs=0.01)


def test_rot_codec_beats_plain_int_on_heavy_tails():
    x = heavy_tailed(2048)
    plain = get_codec("int4")
    rot = get_codec("rot-int4")
    rot.fit(x)
    err_plain = np.abs(plain.decode(plain.encode(x)) - x).mean()
    err_rot = np.abs(rot.decode(rot.encode(x)) - x).mean()
    assert err_rot < err_plain / 3  # rotation spreads the outlier channels


def test_rot_codec_bytes_match_plain_int():
    x = heavy_tailed(256)
    for bits in (4, 6, 8):
        c = get_codec(f"rot-int{bits}")
        c.fit(x)
        measured = Codec.payload_nbytes(c.encode(x)) / 256
        assert measured == theoretical_bytes_per_token(f"int{bits}", D)
        assert measured == theoretical_bytes_per_token(f"rot-int{bits}", D)


def test_rot_codec_state_roundtrip():
    x = heavy_tailed(512)
    c = get_codec("rot-int6")
    c.fit(x)
    payload = c.encode(x)
    c2 = get_codec("rot-int6")
    c2.load_state(c.state(), c.state_arrays())
    np.testing.assert_array_equal(c.decode(payload), c2.decode(payload))


def test_rot_int8_high_fidelity():
    x = heavy_tailed(1024)
    c = get_codec("rot-int8")
    c.fit(x)
    dec = c.decode(c.encode(x))
    rel = np.linalg.norm(dec - x) / np.linalg.norm(x)
    assert rel < 0.01


def test_decode_encode_deterministic():
    x = heavy_tailed(256)
    for name in ["fp16", "int8", "int6", "int4-o8", "pca128"]:
        c = get_codec(name)
        c.fit(x)
        a = c.decode(c.encode(x))
        b = c.decode(c.encode(x))
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("exponent", [4, 5, 6])
def test_noise_codec_hits_exact_relative_dose(exponent):
    """The dose-response control must land ON its target, not near it.

    Per-token noise norm is rescaled to rel * ||x||, so the realized
    rel-L2 is exact per token rather than exact in expectation. If this
    drifts, the x-axis of the dose-response curve is wrong.
    """
    x = heavy_tailed()
    c = get_codec(f"noise{exponent}")
    x_hat = c.decode(c.encode(x))
    rel = (np.linalg.norm(x - x_hat, axis=1)
           / np.linalg.norm(x, axis=1))
    np.testing.assert_allclose(rel, 10.0 ** (-exponent), rtol=2e-2)


def test_noise_codec_is_deterministic_and_advances_per_shard():
    x = heavy_tailed()
    a, b = get_codec("noise6"), get_codec("noise6")
    # same shard index -> bit-identical, so a rerun reproduces the store
    np.testing.assert_array_equal(a.encode(x)["x"], b.encode(x)["x"])
    # successive shards get independent noise, not a repeated pattern
    c = get_codec("noise6")
    first, second = c.encode(x)["x"], c.encode(x)["x"]
    assert not np.array_equal(first, second)


def test_noise_codec_is_not_a_compression_codec():
    """It stores fp32 and must never be read as a rate-distortion point."""
    assert theoretical_bytes_per_token("noise6", D) == 4 * D
