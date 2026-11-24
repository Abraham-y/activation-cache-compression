import numpy as np
import pytest

from acc.codecs import get_codec
from acc.store import BatchLoader, Store, StoreWriter, transcode

RNG = np.random.default_rng(2)
D = 64


def make_master(dir, n_shards=3, shard_tokens=500):
    w = StoreWriter(dir, get_codec("fp32"), D)
    shards = []
    for _ in range(n_shards):
        x = RNG.standard_normal((shard_tokens, D)).astype(np.float32)
        x[:, 5] *= 30.0
        shards.append(x)
        w.write_shard(x)
    w.close()
    return np.concatenate(shards)


def test_store_roundtrip_exact(tmp_path):
    x = make_master(tmp_path / "m")
    s = Store(tmp_path / "m")
    back = np.concatenate([xs for xs, _ in s.iter_shards()])
    np.testing.assert_array_equal(back, x)
    assert s.manifest["payload_bytes_per_token"] == 4 * D


@pytest.mark.parametrize("codec", ["fp16", "int8", "int6", "int4-o8", "pca32"])
def test_transcode_alignment_and_fidelity(tmp_path, codec):
    x = make_master(tmp_path / "m")
    man = transcode(tmp_path / "m", tmp_path / codec, codec, calib_tokens=600)
    s = Store(tmp_path / codec)
    assert [sh["n_tokens"] for sh in s.shards] == \
           [sh["n_tokens"] for sh in Store(tmp_path / "m").shards]
    back = np.concatenate([xs for xs, _ in s.iter_shards()])
    assert back.shape == x.shape
    rel = np.linalg.norm(back - x) / np.linalg.norm(x)
    assert rel < 0.5
    assert man["payload_bytes_per_token"] < 4 * D


def test_loader_deterministic_and_codec_invariant(tmp_path):
    make_master(tmp_path / "m")
    transcode(tmp_path / "m", tmp_path / "fp16", "fp16")

    def batches(store_dir, seed):
        loader = BatchLoader(Store(store_dir), batch_size=100, seed=seed,
                            group_shards=2)
        return [np.asarray(b) for b in loader.batches(epochs=1)]

    a1 = batches(tmp_path / "m", seed=7)
    a2 = batches(tmp_path / "m", seed=7)
    for x, y in zip(a1, a2):
        np.testing.assert_array_equal(x, y)  # same seed, same order
    b1 = batches(tmp_path / "fp16", seed=7)
    assert len(a1) == len(b1)
    for x, y in zip(a1, b1):
        # identical shuffle across codecs: same rows, fp16-sized error only
        np.testing.assert_allclose(x, y, atol=0.05, rtol=2**-10)
    a3 = batches(tmp_path / "m", seed=8)
    assert any(not np.array_equal(x, y) for x, y in zip(a1, a3))


def test_loader_multiple_epochs_reshuffle(tmp_path):
    make_master(tmp_path / "m", n_shards=1, shard_tokens=400)
    loader = BatchLoader(Store(tmp_path / "m"), batch_size=200, seed=0)
    batches = [b.clone() for b in loader.batches(epochs=2)]
    assert len(batches) == 4
    assert not np.array_equal(batches[0].numpy(), batches[2].numpy())


def test_prefetch_stream_identical_to_sync(tmp_path):
    make_master(tmp_path / "m", n_shards=3, shard_tokens=500)
    sync = [np.asarray(b) for b in BatchLoader(
        Store(tmp_path / "m"), 100, seed=7, group_shards=2
    ).batches(epochs=2, prefetch=0)]
    pre = [np.asarray(b) for b in BatchLoader(
        Store(tmp_path / "m"), 100, seed=7, group_shards=2
    ).batches(epochs=2, prefetch=3)]
    assert len(sync) == len(pre)
    for a, b in zip(sync, pre):
        np.testing.assert_array_equal(a, b)


def test_prefetch_propagates_producer_errors(tmp_path):
    make_master(tmp_path / "m", n_shards=2, shard_tokens=300)
    store = Store(tmp_path / "m")
    store.shards[1]["file"] = "missing_shard.npz"
    loader = BatchLoader(store, 100, seed=0, group_shards=1)
    import pytest as _pytest
    with _pytest.raises(FileNotFoundError):
        for _ in loader.batches(epochs=1, prefetch=2):
            pass


def test_loader_counts_all_tokens(tmp_path):
    make_master(tmp_path / "m", n_shards=3, shard_tokens=500)
    loader = BatchLoader(Store(tmp_path / "m"), batch_size=100, seed=0,
                         group_shards=2)
    n = sum(b.shape[0] for b in loader.batches(epochs=1))
    assert n == 1500  # 1500 divisible by 100: leftover buffer carries over
    assert loader.stats()["disk_bytes_read"] > 0


# --- decode modes (camera-ready timing experiment): every mode must yield
# the same batch stream bit for bit, so timing differences are pure loader
# cost. Uneven shard sizes force leftovers across groups and epochs.

def _uneven_master(dir):
    w = StoreWriter(dir, get_codec("fp32"), D)
    for n in (530, 270, 610, 150, 333):
        x = RNG.standard_normal((n, D)).astype(np.float32)
        x[:, 5] *= 30.0
        w.write_shard(x)
    w.close()


def _devices():
    import torch
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("codec", ["std-int4", "fp16", "fp32"])
def test_decode_modes_yield_identical_batches(tmp_path, codec):
    _uneven_master(tmp_path / "m")
    sdir = tmp_path / "m" if codec == "fp32" else tmp_path / codec
    if codec != "fp32":
        transcode(tmp_path / "m", sdir, codec, calib_tokens=600)

    def stream(decode, device, prefetch):
        loader = BatchLoader(Store(sdir), batch_size=128, seed=3,
                             group_shards=2, device=device, decode=decode)
        return [b.cpu().numpy() for b in loader.batches(epochs=2, prefetch=prefetch)]

    ref = stream("cpu", "cpu", 2)
    assert len(ref) > 0
    modes = ["gpu"] + (["cpu-nomatmul"] if codec.startswith("std") else [])
    for mode in modes:
        for dev in _devices() if mode == "gpu" else ["cpu"]:
            for prefetch in (0, 2):
                got = stream(mode, dev, prefetch)
                assert len(got) == len(ref), (mode, dev)
                for a, b in zip(ref, got):
                    np.testing.assert_array_equal(a, b)


def test_std_decode_nomatmul_matches_decode_exactly():
    c = get_codec("std-int4")
    x = RNG.standard_normal((2000, D)).astype(np.float32) * np.linspace(0.1, 40, D, dtype=np.float32)
    c.fit(x[:1000])
    p = c.encode(x)
    np.testing.assert_array_equal(c.decode(p), c.decode_nomatmul(p))


def test_nomatmul_refuses_a_real_rotation():
    c = get_codec("rot-int4")
    c.fit(RNG.standard_normal((500, D)).astype(np.float32))
    with pytest.raises(AssertionError):
        c.decode_nomatmul(c.encode(RNG.standard_normal((10, D)).astype(np.float32)))
