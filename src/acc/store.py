"""Sharded compressed activation store.

Layout of a store directory:
    manifest.json           codec name/state, shard list, measured bytes
    codec_state.npz         calibration arrays (only if codec needs them)
    shard_00000.npz ...     encoded payload arrays

Alignment invariant: a transcoded store copies the master's shard boundaries
and row order exactly. Together with the seeded loader this guarantees every
codec's SAE sees the same tokens in the same order — codec comparisons at
equal seed differ only in the bytes, nothing else.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .codecs import Codec, get_codec
from .utils import read_json, write_json


class StoreWriter:
    def __init__(self, dir: str | Path, codec: Codec, d: int, source: str = ""):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.codec = codec
        self.d = d
        self.source = source
        self.shards = []
        self.n_tokens = 0
        self.payload_bytes = 0

    def write_shard(self, x_f32: np.ndarray) -> None:
        """Encode one shard of fp32 activations [n, d] and write it."""
        assert x_f32.shape[1] == self.d
        payload = self.codec.encode(x_f32)
        idx = len(self.shards)
        fname = f"shard_{idx:05d}.npz"
        np.savez(self.dir / fname, **payload)
        nbytes = Codec.payload_nbytes(payload)
        disk = (self.dir / fname).stat().st_size
        self.shards.append(
            {"file": fname, "n_tokens": int(x_f32.shape[0]),
             "payload_bytes": nbytes, "disk_bytes": disk}
        )
        self.n_tokens += int(x_f32.shape[0])
        self.payload_bytes += nbytes

    def close(self) -> dict:
        state_arrays = self.codec.state_arrays()
        state_file = None
        if state_arrays:
            state_file = "codec_state.npz"
            np.savez(self.dir / state_file, **state_arrays)
        manifest = {
            "codec": self.codec.name,
            "codec_state": {**self.codec.state(), "d": self.d},
            "codec_state_file": state_file,
            "codec_state_bytes": self.codec.state_nbytes(),
            "d": self.d,
            "n_tokens": self.n_tokens,
            "source": self.source,
            "payload_bytes": self.payload_bytes,
            "payload_bytes_per_token": self.payload_bytes / max(self.n_tokens, 1),
            "disk_bytes_per_token":
                sum(s["disk_bytes"] for s in self.shards) / max(self.n_tokens, 1),
            "shards": self.shards,
        }
        write_json(self.dir / "manifest.json", manifest)
        return manifest


class Store:
    def __init__(self, dir: str | Path):
        self.dir = Path(dir)
        self.manifest = read_json(self.dir / "manifest.json")
        self.codec = get_codec(self.manifest["codec"])
        state_file = self.manifest.get("codec_state_file")
        arrays = {}
        if state_file:
            with np.load(self.dir / state_file) as z:
                arrays = {k: z[k] for k in z.files}
        self.codec.load_state(self.manifest["codec_state"], arrays)
        self.d = self.manifest["d"]
        self.n_tokens = self.manifest["n_tokens"]
        self.shards = self.manifest["shards"]
        self.disk_seconds = 0.0
        self.decode_seconds = 0.0

    def read_shard(self, i: int, nomatmul: bool = False) -> tuple[np.ndarray, int]:
        """Returns (decoded fp32 [n, d], disk bytes read). Accumulates
        disk_seconds (file -> arrays) and decode_seconds separately so
        throughput reporting can distinguish disk from codec CPU.
        nomatmul: use the codec's matmul-free decode (identity-rotation
        std-int stores only; bit-identical)."""
        payload, disk = self.read_payload(i)
        t1 = time.perf_counter()
        x = (self.codec.decode_nomatmul(payload) if nomatmul
             else self.codec.decode(payload))
        self.decode_seconds += time.perf_counter() - t1
        return x, disk

    def read_payload(self, i: int) -> tuple[dict, int]:
        """The shard's raw encoded arrays, undecoded (for GPU-side decode)."""
        meta = self.shards[i]
        t0 = time.perf_counter()
        with np.load(self.dir / meta["file"]) as z:
            payload = {k: z[k] for k in z.files}
        self.disk_seconds += time.perf_counter() - t0
        return payload, meta["disk_bytes"]

    def iter_shards(self):
        for i in range(len(self.shards)):
            yield self.read_shard(i)


class BatchLoader:
    """Deterministic shuffled batch stream for SAE training.

    Reads shards sequentially (keeps disk access sequential and measurable),
    decodes, groups `group_shards` shards into a buffer, permutes rows with a
    seeded RNG, yields fixed-size batches. The permutation depends only on
    (seed, epoch, buffer position, row count) — identical for every codec
    store transcoded from the same master.
    """

    DECODE_MODES = ("cpu", "cpu-nomatmul", "gpu")

    def __init__(self, store: Store, batch_size: int, seed: int,
                 group_shards: int = 4, device: str = "cpu",
                 shuffle: bool = True, decode: str = "cpu"):
        """decode: 'cpu' (codec.decode, the default every paper run used),
        'cpu-nomatmul' (identity-rotation stores without the d x d matmul),
        or 'gpu' (raw payload to the device, decoded there). All three yield
        the same batch stream bit for bit (tests/test_store.py); only where
        the work happens differs."""
        assert decode in self.DECODE_MODES, decode
        self.store = store
        self.batch_size = batch_size
        self.seed = seed
        self.group_shards = group_shards
        self.device = device
        self.shuffle = shuffle
        self.decode = decode
        self.tokens_read = 0
        self.disk_bytes_read = 0
        self.read_seconds = 0.0

    def _group_buffers(self, epochs):
        """Sequential decoded+shuffled group buffers. Runs in ONE producer
        (thread or inline), so RNG consumption order — and therefore the
        batch stream — is identical with and without prefetch."""
        epoch = 0
        while epochs is None or epoch < epochs:
            rng = np.random.default_rng((self.seed, epoch))
            n_shards = len(self.store.shards)
            leftover = None
            for g0 in range(0, n_shards, self.group_shards):
                t0 = time.perf_counter()
                parts = []
                for i in range(g0, min(g0 + self.group_shards, n_shards)):
                    x, disk = self.store.read_shard(
                        i, nomatmul=self.decode == "cpu-nomatmul")
                    self.disk_bytes_read += disk
                    parts.append(x)
                buf = np.concatenate(parts, axis=0)
                if leftover is not None:
                    buf = np.concatenate([leftover, buf], axis=0)
                    leftover = None
                if self.shuffle:
                    buf = buf[rng.permutation(buf.shape[0])]
                self.read_seconds += time.perf_counter() - t0
                n_full = buf.shape[0] // self.batch_size * self.batch_size
                if n_full < buf.shape[0]:
                    leftover = buf[n_full:]
                yield buf[:n_full]
            epoch += 1

    def batches(self, epochs: int | None = 1, prefetch: int = 2):
        """Yield fp32 tensors [batch_size, d] on device. epochs=None →
        forever. prefetch>0 decodes+shuffles up to that many group buffers
        ahead in a producer thread, overlapping loader CPU with training —
        the batch stream is bit-identical to prefetch=0.
        decode='gpu': the producer only reads raw payloads and draws the
        permutation; decode, leftover handling and the shuffle run on the
        device in this (consumer) thread."""
        if self.decode == "gpu":
            state = {"left": None}
            for plan in self._prefetched(self._gpu_plans(epochs), prefetch):
                yield from self._emit_tensor(self._gpu_decode(plan, state))
            return
        for buf in self._prefetched(self._group_buffers(epochs), prefetch):
            yield from self._emit(buf)

    def _prefetched(self, gen, prefetch):
        if prefetch <= 0:
            yield from gen
            return
        import queue as _queue
        import threading
        q: _queue.Queue = _queue.Queue(maxsize=prefetch)
        _END, _ERR = object(), object()

        def producer():
            try:
                for item in gen:
                    q.put(item)
                q.put(_END)
            except BaseException as e:  # surfaced to the consumer
                q.put((_ERR, e))

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is _END:
                break
            if isinstance(item, tuple) and item[0] is _ERR:
                raise item[1]
            yield item

    def _gpu_plans(self, epochs):
        """Disk reads and RNG only, mirroring _group_buffers exactly: the
        same permutation length (leftover rows + group rows), the same RNG
        draw per group, leftover dropped at each epoch boundary."""
        epoch = 0
        while epochs is None or epoch < epochs:
            rng = np.random.default_rng((self.seed, epoch))
            n_shards = len(self.store.shards)
            left = 0
            for g0 in range(0, n_shards, self.group_shards):
                t0 = time.perf_counter()
                parts, rows = [], 0
                for i in range(g0, min(g0 + self.group_shards, n_shards)):
                    payload, disk = self.store.read_payload(i)
                    self.disk_bytes_read += disk
                    parts.append(payload)
                    rows += self.store.shards[i]["n_tokens"]
                total = left + rows
                perm = rng.permutation(total) if self.shuffle else None
                n_full = total // self.batch_size * self.batch_size
                self.read_seconds += time.perf_counter() - t0
                yield (parts, perm, n_full, g0 == 0)
                left = total - n_full
            epoch += 1

    def _gpu_decode(self, plan, state):
        parts, perm, n_full, epoch_start = plan
        if epoch_start:
            state["left"] = None
        buf = torch.cat([self.store.codec.decode_torch(p, self.device)
                         for p in parts], dim=0)
        if state["left"] is not None:
            buf = torch.cat([state["left"], buf], dim=0)
        if perm is not None:
            buf = buf[torch.from_numpy(perm).to(self.device)]
        state["left"] = buf[n_full:] if n_full < buf.shape[0] else None
        return buf[:n_full]

    def _emit_tensor(self, buf):
        for b0 in range(0, buf.shape[0], self.batch_size):
            self.tokens_read += self.batch_size
            yield buf[b0 : b0 + self.batch_size]

    def _emit(self, buf):
        for b0 in range(0, buf.shape[0], self.batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(buf[b0 : b0 + self.batch_size])
            ).to(self.device)
            self.tokens_read += self.batch_size
            yield batch

    def stats(self) -> dict:
        """Pipeline time (read_seconds) covers disk + decode + buffer
        shuffle; disk and decode are also broken out so 'disk throughput'
        means what it says and is comparable across codecs."""
        s = max(self.read_seconds, 1e-9)
        return {
            "decode_mode": self.decode,
            "tokens_read": self.tokens_read,
            "disk_bytes_read": self.disk_bytes_read,
            "read_seconds": self.read_seconds,
            "disk_seconds": self.store.disk_seconds,
            "decode_seconds": self.store.decode_seconds,
            "read_disk_MB_per_s":
                self.disk_bytes_read / max(self.store.disk_seconds, 1e-9) / 1e6,
            "decode_MB_per_s":
                self.disk_bytes_read / max(self.store.decode_seconds, 1e-9) / 1e6,
            "pipeline_MB_per_s": self.disk_bytes_read / s / 1e6,
            "read_tokens_per_s": self.tokens_read / s,
        }


def transcode(master_dir: str | Path, out_dir: str | Path, codec_name: str,
              calib_tokens: int = 262144, calib_master_dir=None) -> dict:
    """Re-encode a master fp32 store under `codec_name`, shard-for-shard.

    Calibration (outlier channels, PCA basis) uses the first `calib_tokens`
    of `calib_master_dir` (defaults to the master being transcoded) so that
    train and eval splits share identical codec state.
    """
    master = Store(master_dir)
    codec = get_codec(codec_name)
    if codec.needs_fit:
        calib = Store(calib_master_dir) if calib_master_dir else master
        parts, have = [], 0
        for x, _ in calib.iter_shards():
            parts.append(x)
            have += x.shape[0]
            if have >= calib_tokens:
                break
        sample = np.concatenate(parts, axis=0)[:calib_tokens]
        codec.fit(sample)
    writer = StoreWriter(out_dir, codec, master.d, source=str(master_dir))
    t0 = time.perf_counter()
    for x, _ in master.iter_shards():
        writer.write_shard(x)
    manifest = writer.close()
    manifest["transcode_seconds"] = time.perf_counter() - t0
    return manifest
