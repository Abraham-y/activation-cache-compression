"""SAELens drop-in adapter: train a SAELens SAE from a compressed cache.

SAELens >= 6.x trains from a ``DataProvider``, which is just
``Iterator[torch.Tensor]`` yielding flat ``[batch, d_in]`` activation batches
(see ``sae_lens.training.types``). ``CompressedActivationsStore`` implements
that protocol (plus the ``next_batch`` / checkpoint methods of SAELens's own
``ActivationsStore``) on top of our codec-compressed store, so it can be
passed directly to ``SAETrainer`` in place of SAELens's activations store.

Usage — train a SAELens SAE from a compressed cache::

    from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig
    from sae_lens.config import SAETrainerConfig, LoggingConfig
    from sae_lens.training.sae_trainer import SAETrainer
    from acc.saelens_adapter import CompressedActivationsStore

    store = CompressedActivationsStore("runs/cache_rot-int4", batch_size=4096, seed=0)
    sae = StandardTrainingSAE(StandardTrainingSAEConfig(d_in=store.d_in, d_sae=16384))
    cfg = SAETrainerConfig(
        total_training_samples=100_000_000, train_batch_size_samples=4096,
        lr=3e-4, lr_end=3e-5,  # SAETrainerConfig requires lr_end explicitly
        device="cuda", logger=LoggingConfig(log_to_wandb=False))
    sae = SAETrainer(cfg=cfg, sae=sae, data_provider=store).fit()

This module deliberately does NOT import ``sae_lens``: the adapter only has
to satisfy SAELens's iterator protocol, so ``acc`` keeps zero dependency on
it. It works with or without sae-lens installed (verified against
sae-lens 6.49.1).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .store import BatchLoader, Store

_STATE_FILENAME = "compressed_activations_store_state.json"


class CompressedActivationsStore:
    """Drop-in replacement for SAELens's ActivationsStore backed by a
    compressed :class:`acc.store.Store`.

    Yields fp32 (or ``dtype``) tensors of shape ``[batch_size, d_in]`` on
    ``device``, cycling the cache indefinitely with a per-epoch reshuffle.
    The batch stream is a deterministic function of
    (store, seed, batch_size, group_shards, shuffle) — it reuses
    :class:`acc.store.BatchLoader`'s seeded shuffle, so two adapters built
    with the same arguments yield bit-identical streams, and stores
    transcoded from the same master shard-for-shard yield the same rows in
    the same order under every codec.
    """

    def __init__(
        self,
        store_dir: str | Path | Store,
        batch_size: int,
        seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        group_shards: int = 4,
        prefetch: int = 2,
        shuffle: bool = True,
    ):
        self.store = store_dir if isinstance(store_dir, Store) else Store(store_dir)
        if self.store.n_tokens < batch_size:
            raise ValueError(
                f"store has {self.store.n_tokens} tokens, fewer than one "
                f"batch of {batch_size}; no full batch can ever be yielded"
            )
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        self.group_shards = group_shards
        self.prefetch = prefetch
        self.shuffle = shuffle
        self.d_in = self.store.d
        self.batches_yielded = 0
        self._start_stream()

    def _start_stream(self) -> None:
        """(Re)create the underlying loader + infinite batch iterator."""
        self.loader = BatchLoader(
            self.store,
            batch_size=self.batch_size,
            seed=self.seed,
            group_shards=self.group_shards,
            device="cpu",
            shuffle=self.shuffle,
        )
        self._iterator = self.loader.batches(epochs=None, prefetch=self.prefetch)

    # -- DataProvider protocol (what SAETrainer actually consumes) ----------

    def __iter__(self) -> "CompressedActivationsStore":
        return self

    def __next__(self) -> torch.Tensor:
        batch = next(self._iterator)  # fp32 [batch_size, d_in] on cpu
        self.batches_yielded += 1
        return batch.to(device=self.device, dtype=self.dtype)

    # -- ActivationsStore-compatible surface --------------------------------

    def next_batch(self) -> torch.Tensor:
        """Alias matching sae_lens ActivationsStore.next_batch()."""
        return self.__next__()

    def state_dict(self) -> dict:
        return {
            "batches_yielded": self.batches_yielded,
            "seed": self.seed,
            "batch_size": self.batch_size,
        }

    def save_to_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Save resume state (position in the deterministic batch stream)."""
        path = Path(checkpoint_path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / _STATE_FILENAME, "w") as f:
            json.dump(self.state_dict(), f)

    def load_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Restore position by replaying the deterministic stream.

        The stream is a pure function of (store, seed, batch_size, ...), so
        seeking = restarting and discarding ``batches_yielded`` batches. This
        decodes the skipped shards again (cost proportional to position) but
        guarantees the resumed stream is bit-identical to an uninterrupted
        run.
        """
        with open(Path(checkpoint_path) / _STATE_FILENAME) as f:
            state = json.load(f)
        if state["seed"] != self.seed or state["batch_size"] != self.batch_size:
            raise ValueError(
                "checkpoint was written with seed="
                f"{state['seed']}, batch_size={state['batch_size']}; this "
                f"store has seed={self.seed}, batch_size={self.batch_size} — "
                "the stream would not line up"
            )
        target = state["batches_yielded"]
        self._start_stream()
        self.batches_yielded = 0
        for _ in range(target):
            next(self._iterator)
            self.batches_yielded += 1

    def stats(self) -> dict:
        """Loader throughput stats (see acc.store.BatchLoader.stats)."""
        return self.loader.stats()
