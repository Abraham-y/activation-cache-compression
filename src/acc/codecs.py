"""Lossy activation codecs.

Every codec maps float32 [n_tokens, d] to a dict of named numpy arrays
("payload") and back. Bytes per token is always MEASURED from the encoded
payload, never assumed from the nominal bit width. Codec calibration state
(outlier channel indices, PCA basis) is stored separately and its size is
reported once per store, amortized across all tokens.

Registry names:
    fp32, fp16, bf16
    int{8,6,4}          per-token symmetric quantization
    int{8,6,4}-o{K}     same, with K outlier channels kept in fp16
    pca{R}              PCA projection to R dims, stored fp16
    rp{R}               orthonormalized Gaussian random projection to R dims, fp16
    noise{E}            fp32 storage + random perturbation at rel-L2 = 1e-E

`noise{E}` is not a compression codec and is not on the rate-distortion
curve: it stores full fp32 and costs 4d bytes/token. It exists as the
dose-response control for the audit objection that retrained feature
survival measures training chaos rather than information loss. The
design has a discontinuity without it -- zero perturbation gives a
bit-exact 1.0000 ceiling, and the smallest perturbation ever tested
(fp16, rel-L2 2e-4) already flips ~40% of features, with nothing in
between. Placing a point three decades below fp16 says whether survival
is continuous in perturbation magnitude down to negligible doses (in
which case a survival number is a property of SGD's sensitivity, not of
the codec) or whether fp16 sits on a real shoulder.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np

Payload = dict


class Codec(ABC):
    name: str = ""
    needs_fit = False

    def fit(self, sample: np.ndarray) -> None:
        """Calibrate from a sample of fp32 activations. No-op by default."""

    @abstractmethod
    def encode(self, x: np.ndarray) -> Payload: ...

    @abstractmethod
    def decode(self, payload: Payload) -> np.ndarray: ...

    def state(self) -> dict:
        return {}

    def state_arrays(self) -> Payload:
        return {}

    def load_state(self, state: dict, arrays: Payload) -> None:
        pass

    @staticmethod
    def payload_nbytes(payload: Payload) -> int:
        return int(sum(a.nbytes for a in payload.values()))

    def state_nbytes(self) -> int:
        return int(sum(a.nbytes for a in self.state_arrays().values()))


def _f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even truncation of fp32 to bf16, stored as uint16."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def _bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


class FloatCodec(Codec):
    def __init__(self, dtype: str):
        assert dtype in ("fp32", "fp16", "bf16")
        self.name = dtype
        self.dtype = dtype

    def encode(self, x):
        if self.dtype == "fp32":
            return {"x": np.ascontiguousarray(x, dtype=np.float32)}
        if self.dtype == "fp16":
            return {"x": x.astype(np.float16)}
        return {"x": _f32_to_bf16_bits(x)}

    def decode(self, payload):
        if self.dtype == "bf16":
            return _bf16_bits_to_f32(payload["x"]).reshape(payload["x"].shape)
        return payload["x"].astype(np.float32)

    def decode_torch(self, payload, device):
        """GPU decode from the raw payload; fp16 -> fp32 is exact, so this
        is bit-identical to decode()."""
        import torch
        assert self.dtype in ("fp32", "fp16"), "decode_torch: fp32/fp16 only"
        x = torch.from_numpy(np.ascontiguousarray(payload["x"])).to(device)
        return x.to(torch.float32)

    def state(self):
        return {"dtype": self.dtype}


class NoiseCodec(Codec):
    """fp32 storage plus an isotropic perturbation of exact relative norm.

    Per token the stored vector is x + n, where n is drawn isotropically
    and rescaled so ||n|| / ||x|| equals `rel` EXACTLY. Fixing the norm
    rather than the noise variance makes the dose exact per token instead
    of exact in expectation, so the treatment lands on a single point of
    the distortion axis with no spread of its own to disentangle later.

    Direction is random; only the magnitude is controlled. Decode is the
    identity -- the perturbation is baked in at encode time, so a store
    written by this codec is an ordinary fp32 store and every downstream
    command treats it like one.

    Determinism: shards are seeded by `seed` plus a shard counter, and
    transcode writes a store in a single ordered pass, so re-running
    reproduces the store bit-for-bit.
    """

    def __init__(self, exponent: int, seed: int = 0):
        # seed > 0 is an independent DRAW of the same dose (name suffix
        # "d{seed}"), used to estimate draw-to-draw variance of a rung
        self.name = f"noise{exponent}" + (f"d{seed}" if seed else "")
        self.exponent = exponent
        self.rel = 10.0 ** (-exponent)
        self.seed = seed
        self._shard = 0

    def encode(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        rng = np.random.default_rng((self.seed, self.exponent, self._shard))
        self._shard += 1
        n = rng.standard_normal(x.shape, dtype=np.float32)
        # scale each row so ||n|| == rel * ||x||; zero rows stay zero
        nn = np.linalg.norm(n, axis=1, keepdims=True).clip(min=1e-30)
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        return {"x": (x + n * (self.rel * xn / nn)).astype(np.float32)}

    def decode(self, payload):
        return payload["x"].astype(np.float32)

    def state(self):
        return {"rel": self.rel, "exponent": self.exponent, "seed": self.seed}


def _pack_uint_bitpath(q: np.ndarray, bits: int) -> np.ndarray:
    """Generic reference implementation via numpy bit arrays (slow)."""
    n, d = q.shape
    bit_arr = np.unpackbits(q[:, :, None], axis=2, count=8)[:, :, 8 - bits:]
    return np.packbits(bit_arr.reshape(n, d * bits), axis=1)


def _unpack_uint_bitpath(packed: np.ndarray, bits: int, d: int) -> np.ndarray:
    n = packed.shape[0]
    bit_arr = np.unpackbits(packed, axis=1, count=d * bits).reshape(n, d, bits)
    full = np.zeros((n, d, 8), dtype=np.uint8)
    full[:, :, 8 - bits:] = bit_arr
    return np.packbits(full, axis=2)[:, :, 0]


def pack_uint(q: np.ndarray, bits: int) -> np.ndarray:
    """Pack uint8 values < 2**bits, shape [n, d], into [n, d*bits//8] bytes.

    Big-endian within the row bitstream, identical to the numpy bit-array
    reference path (tested equal). Requires d*bits % 8 == 0 so rows stay
    byte-aligned (holds for d=768 at 4 and 6 bits, and for 768-K outlier
    channels with K a multiple of 4). Shift arithmetic instead of
    unpackbits/packbits: the decode sits on the SAE training path.
    """
    n, d = q.shape
    assert (d * bits) % 8 == 0, f"d={d}, bits={bits} not byte-aligned"
    assert q.dtype == np.uint8
    if bits == 4:
        return (q[:, 0::2] << 4) | q[:, 1::2]
    if bits == 6:
        g = q.reshape(n, d // 4, 4)
        out = np.empty((n, d // 4, 3), dtype=np.uint8)
        out[:, :, 0] = (g[:, :, 0] << 2) | (g[:, :, 1] >> 4)
        out[:, :, 1] = ((g[:, :, 1] & 15) << 4) | (g[:, :, 2] >> 2)
        out[:, :, 2] = ((g[:, :, 2] & 3) << 6) | g[:, :, 3]
        return out.reshape(n, d * 6 // 8)
    return _pack_uint_bitpath(q, bits)


def unpack_uint(packed: np.ndarray, bits: int, d: int) -> np.ndarray:
    n = packed.shape[0]
    if bits == 4:
        out = np.empty((n, d), dtype=np.uint8)
        out[:, 0::2] = packed >> 4
        out[:, 1::2] = packed & 15
        return out
    if bits == 6:
        b = packed.reshape(n, d // 4, 3)
        out = np.empty((n, d // 4, 4), dtype=np.uint8)
        out[:, :, 0] = b[:, :, 0] >> 2
        out[:, :, 1] = ((b[:, :, 0] & 3) << 4) | (b[:, :, 1] >> 4)
        out[:, :, 2] = ((b[:, :, 1] & 15) << 2) | (b[:, :, 2] >> 6)
        out[:, :, 3] = b[:, :, 2] & 63
        return out.reshape(n, d)
    return _unpack_uint_bitpath(packed, bits, d)


class SparseUlpCodec(Codec):
    """fp32 storage, one ULP flipped on a SPARSE random fraction of values.

    Why this exists. The paper's plateau argument is bounded below by
    fp32's own resolution: a DENSE codec must move every value it touches
    by at least one ulp, so its mean relative L2 cannot fall below about
    6e-8, and no dense codec reaches the region where agreement might
    recover. A SPARSE perturbation escapes that bound -- touch a fraction
    f of values and the mean relative L2 falls as sqrt(f) -- which is
    exactly the region the paper otherwise declines to populate.

    So this codec answers the one question sec:dose leaves open: is the
    approach to the bit-exact ceiling CONTINUOUS with a knee somewhere
    below 1e-7, or is it a STEP at zero where any perturbation at all
    costs the full ~34pp? A dense ladder cannot distinguish them; a
    sparse one can. It is also the mixed-precision case the paper names
    as untested, since a codec that keeps most channels exact and
    quantizes a few has precisely this shape.

    `fraction` is the share of scalar entries perturbed; each selected
    entry moves by exactly one ulp, sign random. Determinism matches
    NoiseCodec: seeded per shard, single ordered pass, bit-reproducible.
    """

    def __init__(self, exponent: int, seed: int = 0):
        # seed > 0 is an independent draw (placement + directions) of the
        # same per-shard count; name carries "d{seed}"
        self.name = f"sparseulp{exponent}" + (f"d{seed}" if seed else "")
        self.exponent = exponent
        self.fraction = 10.0 ** (-exponent)
        self.seed = seed
        self._shard = 0

    def encode(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        rng = np.random.default_rng((self.seed, 900 + self.exponent, self._shard))
        self._shard += 1
        out = x.copy()
        n = x.size
        k = max(1, int(round(n * self.fraction)))
        idx = rng.choice(n, size=k, replace=False)
        flat = out.reshape(-1)
        # one ulp up or down, taken on the float32 grid so the move is the
        # smallest representable change to that value
        signs = rng.integers(0, 2, size=k).astype(np.int8) * 2 - 1
        tgt = np.where(signs > 0, np.float32(np.inf), np.float32(-np.inf))
        flat[idx] = np.nextafter(flat[idx], tgt, dtype=np.float32)
        return {"x": out}

    def decode(self, payload):
        return payload["x"].astype(np.float32)


class OneUlpCodec(Codec):
    """fp32 storage with EXACTLY ONE scalar in the whole store moved one ulp.

    The minimal possible nonzero perturbation of a cache. sparseulp6 moves
    a millionth of entries -- about 15,000 scalars across a 20M-token
    store, so roughly three tokens in every training batch carry a
    difference. That leaves open whether the 30-point drop needs a
    perturbation present throughout training, or whether a single altered
    bit anywhere is enough. This codec answers it: one scalar, one ulp, in
    one shard, and every other byte of the store bit-identical to fp32.

    It is the SAE-side analogue of Summers & Dinneen's single-weight-bit
    experiment, applied to the training data rather than the weights.

    Deterministic: fires only on the shard index given by `shard`, at a
    fixed offset within it, so a rerun reproduces the store bit-for-bit.
    """

    def __init__(self, shard: int = 0, offset: int = 12345, seed: int = 0):
        self.name = "oneulp"
        self.shard = shard
        self.offset = offset
        self.seed = seed
        self._shard = 0
        self.fired = False

    def encode(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        i = self._shard
        self._shard += 1
        if i != self.shard:
            return {"x": x}
        out = x.copy()
        flat = out.reshape(-1)
        j = self.offset % flat.size
        flat[j] = np.nextafter(flat[j], np.float32(np.inf), dtype=np.float32)
        self.fired = True
        return {"x": out}

    def decode(self, payload):
        return payload["x"].astype(np.float32)


class ClumpUlpCodec(Codec):
    """Same COUNT as sparseulp6, confined to ONE shard: the locality control.

    sparseulp6 scatters ~15,360 one-ulp changes uniformly over a 20M-token
    corpus, which puts a perturbed token in essentially every optimizer
    step. oneulp changes a single scalar, perturbing one step. So those two
    arms vary count and steps-affected TOGETHER, and "how many values
    change" is confounded with "how many training steps see a change".

    This codec breaks the tie: the same 15,360 changes, all inside a single
    shard, so the count matches sparseulp6 exactly while the number of
    affected steps drops by the shard fraction. If agreement returns toward
    100%, the governing variable is how often the optimizer meets a
    perturbation, not how many values differ -- and the paper's framing has
    to change from count to frequency. If it stays near 70%, count is the
    right variable and the confound is ruled out.
    """

    def __init__(self, n_perturb: int = 15360, shard: int = 0, seed: int = 0):
        self.name = "clumpulp"
        self.n_perturb = n_perturb
        self.shard = shard
        self.seed = seed
        self._shard = 0

    def encode(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        i = self._shard
        self._shard += 1
        if i != self.shard:
            return {"x": x}
        out = x.copy()
        flat = out.reshape(-1)
        rng = np.random.default_rng((self.seed, 777, i))
        k = min(self.n_perturb, flat.size)
        idx = rng.choice(flat.size, size=k, replace=False)
        signs = rng.integers(0, 2, size=k).astype(np.int8) * 2 - 1
        tgt = np.where(signs > 0, np.float32(np.inf), np.float32(-np.inf))
        flat[idx] = np.nextafter(flat[idx], tgt, dtype=np.float32)
        return {"x": out}

    def decode(self, payload):
        return payload["x"].astype(np.float32)


class QuantCodec(Codec):
    """Per-token symmetric integer quantization, optional fp16 outlier channels.

    Outlier channels are chosen once, at fit time, as the K channels with the
    largest mean |activation| over the calibration sample (activations are
    heavy-tailed with a few extreme channels; per Adacc, naive low-bit
    quantization of those destroys the signal). Scale is per-token fp32.
    """

    ENC_CHUNK = 32768  # tokens per packing chunk, bounds bit-array temp memory

    def __init__(self, bits: int, outlier_k: int = 0):
        assert bits in (4, 6, 8)
        self.bits = bits
        self.outlier_k = outlier_k
        self.levels = 2 ** (bits - 1) - 1
        self.name = f"int{bits}" + (f"-o{outlier_k}" if outlier_k else "")
        self.needs_fit = outlier_k > 0
        self.outlier_idx: np.ndarray | None = None
        self.keep_idx: np.ndarray | None = None

    def fit(self, sample):
        if self.outlier_k == 0:
            return
        mean_abs = np.abs(sample).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][: self.outlier_k]
        self.outlier_idx = np.sort(idx).astype(np.int64)
        d = sample.shape[1]
        mask = np.ones(d, dtype=bool)
        mask[self.outlier_idx] = False
        self.keep_idx = np.nonzero(mask)[0].astype(np.int64)

    def _quant_part(self, x):
        if self.outlier_k:
            assert self.outlier_idx is not None, f"{self.name} not fitted"
            return x[:, self.keep_idx]
        return x

    def encode(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = {}
        if self.outlier_k:
            out["outliers"] = x[:, self.outlier_idx].astype(np.float16)
        xq = self._quant_part(x)
        scale = np.abs(xq).max(axis=1) / self.levels
        scale = np.maximum(scale, 1e-12).astype(np.float32)
        q = np.rint(xq / scale[:, None]).clip(-self.levels, self.levels)
        if self.bits == 8:
            out["q"] = q.astype(np.int8)
        else:
            off = 1 << (self.bits - 1)
            qu = (q + off).astype(np.uint8)
            chunks = [
                pack_uint(qu[i : i + self.ENC_CHUNK], self.bits)
                for i in range(0, qu.shape[0], self.ENC_CHUNK)
            ]
            out["q"] = np.concatenate(chunks, axis=0)
        out["scale"] = scale
        return out

    def decode(self, payload):
        scale = payload["scale"].astype(np.float32)
        n = scale.shape[0]
        d_q = (
            payload["q"].shape[1]
            if self.bits == 8
            else payload["q"].shape[1] * 8 // self.bits
        )
        if self.bits == 8:
            q = payload["q"].astype(np.float32)
        else:
            off = 1 << (self.bits - 1)
            qu = unpack_uint(payload["q"], self.bits, d_q)
            q = qu.astype(np.float32) - off
        xq = q * scale[:, None]
        if not self.outlier_k:
            return xq
        d = d_q + self.outlier_k
        x = np.empty((n, d), dtype=np.float32)
        x[:, self.keep_idx] = xq
        x[:, self.outlier_idx] = payload["outliers"].astype(np.float32)
        return x

    def state(self):
        return {"bits": self.bits, "outlier_k": self.outlier_k}

    def state_arrays(self):
        if self.outlier_k and self.outlier_idx is not None:
            return {"outlier_idx": self.outlier_idx}
        return {}

    def load_state(self, state, arrays):
        if self.outlier_k:
            self.outlier_idx = arrays["outlier_idx"].astype(np.int64)
            d = state["d"]
            mask = np.ones(d, dtype=bool)
            mask[self.outlier_idx] = False
            self.keep_idx = np.nonzero(mask)[0].astype(np.int64)


class RotQuantCodec(Codec):
    """Incoherence-processed per-token quantization (QuaRot/KVarN-style).

    Calibrated per-channel standardization, then a fixed random rotation,
    then per-token symmetric int quantization. The rotation spreads outlier
    channels across all dimensions so no channel dominates the per-token
    scale — the alternative to keeping explicit fp16 outlier channels.
    Same bytes/token as the plain int codec at equal bits.
    """

    def __init__(self, bits: int):
        assert bits in (4, 6, 8)
        self.bits = bits
        self.levels = 2 ** (bits - 1) - 1
        self.name = f"rot-int{bits}"
        self.needs_fit = True
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.rot: np.ndarray | None = None  # [d, d] orthogonal

    def fit(self, sample):
        d = sample.shape[1]
        self.mu = sample.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.sigma = (sample.std(axis=0, dtype=np.float64) + 1e-6).astype(np.float32)
        rng = np.random.default_rng(777)
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        self.rot = q.astype(np.float32)

    def encode(self, x):
        assert self.rot is not None, f"{self.name} not fitted"
        n, d = x.shape
        chunk = QuantCodec.ENC_CHUNK
        scale = np.empty(n, dtype=np.float32)
        q_parts = []
        for i in range(0, n, chunk):  # bounds transient temps to chunk size
            z = ((np.ascontiguousarray(x[i : i + chunk], dtype=np.float32)
                  - self.mu) / self.sigma) @ self.rot
            s = np.maximum(np.abs(z).max(axis=1) / self.levels, 1e-12)
            scale[i : i + chunk] = s
            q = np.rint(z / s[:, None]).clip(-self.levels, self.levels)
            if self.bits == 8:
                q_parts.append(q.astype(np.int8))
            else:
                off = 1 << (self.bits - 1)
                q_parts.append(pack_uint((q + off).astype(np.uint8), self.bits))
        return {"q": np.concatenate(q_parts, axis=0), "scale": scale}

    def decode(self, payload):
        d = self.rot.shape[0]
        scale = payload["scale"].astype(np.float32)
        if self.bits == 8:
            q = payload["q"].astype(np.float32)
        else:
            off = 1 << (self.bits - 1)
            q = unpack_uint(payload["q"], self.bits, d).astype(np.float32) - off
        z = q * scale[:, None]
        return (z @ self.rot.T) * self.sigma + self.mu

    def is_identity_rotation(self) -> bool:
        return self.rot is not None and np.array_equal(
            self.rot, np.eye(self.rot.shape[0], dtype=np.float32))

    def decode_nomatmul(self, payload):
        """decode() without the d x d matmul, for an identity rotation
        (std-int). Bit-identical to decode(): z @ I adds exact zeros, so
        the matmul changes no bit (tests/test_codecs.py). It is the whole
        difference in CPU decode cost between this and decode()."""
        assert self.is_identity_rotation(), f"{self.name}: rotation is not the identity"
        d = self.rot.shape[0]
        scale = payload["scale"].astype(np.float32)
        if self.bits == 8:
            q = payload["q"].astype(np.float32)
        else:
            off = 1 << (self.bits - 1)
            q = unpack_uint(payload["q"], self.bits, d).astype(np.float32) - off
        z = q * scale[:, None]
        return z * self.sigma + self.mu

    def decode_torch(self, payload, device):
        """GPU decode of an identity-rotation store from its raw payload:
        same operations in the same order as decode_nomatmul, one IEEE
        rounding each, so bit-identical to the CPU path (tested). Only the
        identity rotation is supported: a GPU matmul would accumulate in a
        different order and break bit-identity."""
        import torch
        assert self.is_identity_rotation(), f"{self.name}: rotation is not the identity"
        assert self.bits == 4, "decode_torch implements the 4-bit unpack"
        d = self.rot.shape[0]
        packed = torch.from_numpy(np.ascontiguousarray(payload["q"])).to(device)
        scale = torch.from_numpy(payload["scale"].astype(np.float32)).to(device)
        n = packed.shape[0]
        q = torch.stack((packed >> 4, packed & 15), dim=2).reshape(n, d)
        z = (q.to(torch.float32) - 8.0) * scale[:, None]
        if not hasattr(self, "_mu_t") or self._mu_t.device != torch.device(device):
            self._mu_t = torch.from_numpy(self.mu).to(device)
            self._sigma_t = torch.from_numpy(self.sigma).to(device)
        return z * self._sigma_t + self._mu_t

    def state(self):
        return {"bits": self.bits}

    def state_arrays(self):
        if self.rot is None:
            return {}
        return {"mu": self.mu, "sigma": self.sigma, "rot": self.rot}

    def load_state(self, state, arrays):
        self.mu = arrays["mu"].astype(np.float32)
        self.sigma = arrays["sigma"].astype(np.float32)
        self.rot = arrays["rot"].astype(np.float32)


class ProjCodec(Codec):
    """Dimension reduction to R components, coefficients stored fp16.

    kind='pca': eigenbasis of the calibration covariance (top-R).
    kind='rp': fixed orthonormalized Gaussian random projection (seeded by R),
    the non-adaptive baseline.
    """

    def __init__(self, kind: str, r: int):
        assert kind in ("pca", "rp")
        self.kind = kind
        self.r = r
        self.name = f"{kind}{r}"
        self.needs_fit = True
        self.mean: np.ndarray | None = None
        self.basis: np.ndarray | None = None  # [d, r], orthonormal columns

    def fit(self, sample):
        d = sample.shape[1]
        assert self.r <= d
        self.mean = sample.mean(axis=0, dtype=np.float64).astype(np.float32)
        if self.kind == "pca":
            xc = sample.astype(np.float64) - self.mean.astype(np.float64)
            cov = xc.T @ xc / xc.shape[0]
            _, vecs = np.linalg.eigh(cov)
            self.basis = vecs[:, ::-1][:, : self.r].astype(np.float32)
        else:
            rng = np.random.default_rng(12345 + self.r)
            a = rng.standard_normal((d, self.r))
            q, _ = np.linalg.qr(a)
            self.basis = q.astype(np.float32)

    def encode(self, x):
        assert self.basis is not None, f"{self.name} not fitted"
        z = (x.astype(np.float32) - self.mean) @ self.basis
        return {"z": z.astype(np.float16)}

    def decode(self, payload):
        return payload["z"].astype(np.float32) @ self.basis.T + self.mean

    def state(self):
        return {"kind": self.kind, "r": self.r}

    def state_arrays(self):
        if self.basis is None:
            return {}
        return {"mean": self.mean, "basis": self.basis}

    def load_state(self, state, arrays):
        self.mean = arrays["mean"].astype(np.float32)
        self.basis = arrays["basis"].astype(np.float32)


_FLOAT_RE = re.compile(r"^(fp32|fp16|bf16)$")
_QUANT_RE = re.compile(r"^int([468])(?:-o(\d+))?$")
class StdQuantCodec(RotQuantCodec):
    """Ablation: per-channel standardization then per-token quantization,
    WITHOUT the rotation.

    RotQuantCodec does standardize-then-rotate, so every "rotation repairs
    int4" number in the paper actually measures both operations at once.
    Per-channel scaling is the textbook first-line fix for activation
    outliers and is far cheaper than a dense d x d decode matmul, so which
    of the two carries the repair is the difference between two very
    different recommendations. This subclass is byte-identical to
    rot-int{bits} except that the rotation is the identity, which isolates
    the rotation exactly.
    """

    def __init__(self, bits: int):
        super().__init__(bits)
        self.name = f"std-int{bits}"

    def fit(self, sample):
        super().fit(sample)
        d = sample.shape[1]
        self.rot = np.eye(d, dtype=np.float32)   # identity: no rotation


_ROT_RE = re.compile(r"^rot-int([468])$")
_STD_RE = re.compile(r"^std-int([468])$")
_PROJ_RE = re.compile(r"^(pca|rp)(\d+)$")
_NOISE_RE = re.compile(r"^noise(\d+)(?:d(\d+))?$")
_SPARSE_RE = re.compile(r"^sparseulp(\d+)(?:d(\d+))?$")


def get_codec(name: str) -> Codec:
    if _FLOAT_RE.match(name):
        return FloatCodec(name)
    m = _STD_RE.match(name)
    if m:
        return StdQuantCodec(int(m.group(1)))
    m = _ROT_RE.match(name)
    if m:
        return RotQuantCodec(int(m.group(1)))
    m = _QUANT_RE.match(name)
    if m:
        return QuantCodec(int(m.group(1)), int(m.group(2) or 0))
    m = _PROJ_RE.match(name)
    if m:
        return ProjCodec(m.group(1), int(m.group(2)))
    if name == "oneulp":
        return OneUlpCodec()
    if name == "clumpulp":
        return ClumpUlpCodec()
    m = _SPARSE_RE.match(name)
    if m:
        return SparseUlpCodec(int(m.group(1)), seed=int(m.group(2) or 0))
    m = _NOISE_RE.match(name)
    if m:
        return NoiseCodec(int(m.group(1)), seed=int(m.group(2) or 0))
    raise ValueError(f"unknown codec: {name}")


def theoretical_bytes_per_token(name: str, d: int) -> float:
    """Nominal payload bytes/token; the pipeline always reports measured too."""
    if name == "fp32":
        return 4 * d
    if name in ("fp16", "bf16"):
        return 2 * d
    m = _STD_RE.match(name)
    if m:
        return d * int(m.group(1)) / 8 + 4       # same layout as rot-int*
    m = _ROT_RE.match(name)
    if m:
        return d * int(m.group(1)) / 8 + 4
    m = _QUANT_RE.match(name)
    if m:
        bits, k = int(m.group(1)), int(m.group(2) or 0)
        return (d - k) * bits / 8 + 2 * k + 4  # + per-token fp32 scale
    m = _PROJ_RE.match(name)
    if m:
        return 2 * int(m.group(2))
    if name in ("oneulp", "clumpulp") or _SPARSE_RE.match(name):
        return 4.0 * d          # fp32 storage; only a few entries differ
    if _NOISE_RE.match(name):
        return 4 * d  # a control, not a codec: fp32 storage, no saving
    raise ValueError(f"unknown codec: {name}")
