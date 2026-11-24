"""Integration tests: CompressedActivationsStore feeding a real SAELens trainer.

Skipped entirely when sae-lens is not installed; the adapter itself has no
sae_lens dependency (verified by test_adapter_imports_without_sae_lens).
"""
import numpy as np
import pytest
import torch

from acc.codecs import get_codec
from acc.saelens_adapter import CompressedActivationsStore
from acc.store import Store, StoreWriter, transcode

sae_lens = pytest.importorskip("sae_lens")

from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig  # noqa: E402
from sae_lens.config import LoggingConfig, SAETrainerConfig  # noqa: E402
from sae_lens.training.sae_trainer import SAETrainer  # noqa: E402

D = 64
BATCH = 256


@pytest.fixture(scope="module")
def store_dirs(tmp_path_factory):
    """Master fp32 store (4 shards x 2048 tokens, d=64) + rot-int4 transcode."""
    root = tmp_path_factory.mktemp("stores")
    rng = np.random.default_rng(0)
    w = StoreWriter(root / "master", get_codec("fp32"), D)
    for _ in range(4):
        x = rng.standard_normal((2048, D)).astype(np.float32)
        x[:, 5] *= 30.0  # heavy-tailed channel, like real activations
        w.write_shard(x)
    w.close()
    transcode(root / "master", root / "rot-int4", "rot-int4", calib_tokens=4096)
    return root


def make_adapter(store_dirs, name="master", seed=0, **kw):
    return CompressedActivationsStore(store_dirs / name, batch_size=BATCH,
                                      seed=seed, **kw)


def test_shape_dtype_device(store_dirs):
    store = make_adapter(store_dirs)
    batch = next(store)
    # SAELens 6.x DataProvider contract: flat [batch, d_in] tensors, and the
    # trainer moves them to the SAE device itself (SAETrainer.step).
    assert batch.shape == (BATCH, D)
    assert batch.dtype == torch.float32
    assert batch.device.type == "cpu"
    assert store.d_in == D
    # configurable dtype
    bf16 = next(make_adapter(store_dirs, dtype=torch.bfloat16))
    assert bf16.dtype == torch.bfloat16


def test_deterministic_given_seed(store_dirs):
    a = make_adapter(store_dirs, seed=7)
    b = make_adapter(store_dirs, seed=7)
    for _ in range(3):
        torch.testing.assert_close(next(a), next(b), rtol=0, atol=0)
    c = make_adapter(store_dirs, seed=8)
    assert not torch.equal(next(make_adapter(store_dirs, seed=7)), next(c))


def test_codec_store_aligned_with_master(store_dirs):
    """rot-int4 store yields the same rows in the same order (codec error only)."""
    m = next(make_adapter(store_dirs, "master", seed=3))
    q = next(make_adapter(store_dirs, "rot-int4", seed=3))
    assert q.shape == m.shape
    rel = (q - m).norm() / m.norm()
    assert rel < 0.2  # same rows; int4 quantization error only


def test_cycles_indefinitely(store_dirs):
    """Cache is 8192 tokens; drawing more than one epoch's worth must work."""
    store = make_adapter(store_dirs, prefetch=0)
    n_batches_per_epoch = 8192 // BATCH
    batches = [next(store) for _ in range(n_batches_per_epoch + 3)]
    assert all(b.shape == (BATCH, D) for b in batches)
    assert store.batches_yielded == n_batches_per_epoch + 3
    # epoch 2 is reshuffled, not a replay of epoch 1
    assert not torch.equal(batches[0], batches[n_batches_per_epoch])


def test_rejects_store_smaller_than_batch(store_dirs):
    with pytest.raises(ValueError, match="fewer than one"):
        CompressedActivationsStore(store_dirs / "master", batch_size=10_000)


def test_checkpoint_save_resume(store_dirs, tmp_path):
    a = make_adapter(store_dirs, seed=5)
    for _ in range(5):
        next(a)
    a.save_to_checkpoint(tmp_path / "ckpt")
    expected = [next(a) for _ in range(3)]

    b = make_adapter(store_dirs, seed=5)
    b.load_from_checkpoint(tmp_path / "ckpt")
    assert b.batches_yielded == 5
    for e in expected:
        torch.testing.assert_close(next(b), e, rtol=0, atol=0)

    mismatched = make_adapter(store_dirs, seed=6)
    with pytest.raises(ValueError, match="would not line up"):
        mismatched.load_from_checkpoint(tmp_path / "ckpt")


@pytest.mark.parametrize("name", ["master", "rot-int4"])
def test_real_saelens_training_run(store_dirs, name):
    """A real SAELens SAETrainer run fed entirely from the compressed cache."""
    torch.manual_seed(0)
    sae = StandardTrainingSAE(StandardTrainingSAEConfig(d_in=D, d_sae=128))
    w_enc0 = sae.W_enc.detach().clone()
    store = make_adapter(store_dirs, name, seed=0)
    cfg = SAETrainerConfig(
        total_training_samples=BATCH * 20,
        train_batch_size_samples=BATCH,
        lr=3e-4,
        lr_end=3e-5,  # SAETrainerConfig (unlike the LLM runner cfg) has no
        # post-init defaulting lr_end to lr/10; the trainer asserts it is set
        device="cpu",
        logger=LoggingConfig(log_to_wandb=False),
    )
    trainer = SAETrainer(cfg=cfg, sae=sae, data_provider=store)
    trained = trainer.fit()

    assert trainer.n_training_steps == 20
    assert trainer.n_training_samples == BATCH * 20
    assert store.batches_yielded == 20
    assert not torch.equal(trained.W_enc.detach(), w_enc0)  # weights moved
    for p in trained.parameters():
        assert torch.isfinite(p).all()

    # one more manual step: loss is finite and every component is finite
    out = trainer.step(next(store))
    assert torch.isfinite(out.loss).all()
    for loss_name, v in out.losses.items():
        v = v() if callable(v) else v
        v = v.item() if isinstance(v, torch.Tensor) else v
        assert np.isfinite(v), loss_name


def test_norm_estimation_path(store_dirs):
    """normalize_activations='expected_average_only_in' pulls extra batches
    through the data provider for norm estimation before training."""
    torch.manual_seed(0)
    sae = StandardTrainingSAE(
        StandardTrainingSAEConfig(
            d_in=D, d_sae=128, normalize_activations="expected_average_only_in"
        )
    )
    store = make_adapter(store_dirs, seed=1)
    cfg = SAETrainerConfig(
        total_training_samples=BATCH * 4,
        train_batch_size_samples=BATCH,
        lr_end=3e-5,
        device="cpu",
        n_batches_for_norm_estimate=8,
        logger=LoggingConfig(log_to_wandb=False),
    )
    trainer = SAETrainer(cfg=cfg, sae=sae, data_provider=store)
    trainer.fit()
    assert store.batches_yielded == 8 + 4  # 8 for norm estimate + 4 train
    for p in sae.parameters():
        assert torch.isfinite(p).all()


def test_adapter_imports_without_sae_lens():
    """acc.saelens_adapter must not import sae_lens (zero hard dependency)."""
    import subprocess
    import sys
    from pathlib import Path

    import acc

    code = (
        "import sys\n"
        "sys.modules['sae_lens'] = None  # any import attempt would raise\n"
        "from acc.saelens_adapter import CompressedActivationsStore\n"
        "assert 'sae_lens' not in CompressedActivationsStore.__module__\n"
    )
    env_path = str(Path(acc.__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
