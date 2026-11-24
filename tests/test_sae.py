import numpy as np
import torch

from acc.codecs import get_codec
from acc.sae import SAEConfig, TopKSAE, load_sae, train_sae
from acc.store import BatchLoader, Store, StoreWriter

RNG = np.random.default_rng(3)


def test_topk_sparsity_and_shapes():
    sae = TopKSAE(d_in=32, d_sae=128, k=4, seed=0)
    x = torch.randn(16, 32)
    recon, z, pre = sae(x)
    assert recon.shape == x.shape and z.shape == (16, 128)
    assert (z.gt(0).sum(dim=1) <= 4).all()
    assert (z >= 0).all()


def test_decoder_rows_unit_norm_after_init_and_renorm():
    sae = TopKSAE(32, 128, 4, seed=0)
    norms = sae.W_dec.detach().norm(dim=1)
    torch.testing.assert_close(norms, torch.ones(128), atol=1e-5, rtol=0)
    with torch.no_grad():
        sae.W_dec.mul_(3.0)
    sae.renorm_decoder()
    torch.testing.assert_close(sae.W_dec.detach().norm(dim=1),
                               torch.ones(128), atol=1e-5, rtol=0)


def test_same_seed_same_init():
    a = TopKSAE(32, 128, 4, seed=5)
    b = TopKSAE(32, 128, 4, seed=5)
    for (n1, p1), (_, p2) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(p1, p2, atol=0, rtol=0)
    c = TopKSAE(32, 128, 4, seed=6)
    assert not torch.equal(a.W_dec, c.W_dec)


def _synthetic_store(dir, n=6000, d=32, n_true=64):
    """Sparse dictionary data an SAE should crack easily."""
    true_dirs = RNG.standard_normal((n_true, d)).astype(np.float32)
    true_dirs /= np.linalg.norm(true_dirs, axis=1, keepdims=True)
    codes = np.zeros((n, n_true), dtype=np.float32)
    for i in range(n):
        on = RNG.choice(n_true, size=3, replace=False)
        codes[i, on] = RNG.uniform(0.5, 2.0, size=3)
    x = codes @ true_dirs
    w = StoreWriter(dir, get_codec("fp32"), d)
    for c0 in range(0, n, 2000):
        w.write_shard(x[c0 : c0 + 2000])
    w.close()


def test_training_reduces_fvu_and_saves(tmp_path):
    _synthetic_store(tmp_path / "store")
    store = Store(tmp_path / "store")
    cfg = SAEConfig(d_in=32, d_sae=128, k=4, aux_k=16, dead_tokens=10_000,
                    lr=3e-3, batch_size=512, steps=300, log_every=50)
    loader = BatchLoader(store, cfg.batch_size, seed=0)
    train_sae(loader, cfg, seed=0, device="cpu", out_dir=tmp_path / "sae",
              codec_name="fp32")
    import json
    logs = [json.loads(l) for l in open(tmp_path / "sae" / "train_log.jsonl")]
    assert logs[-1]["fvu"] < 0.5 * logs[0]["fvu"]
    assert logs[-1]["fvu"] < 0.15
    sae, ckpt = load_sae(tmp_path / "sae")
    assert ckpt["codec"] == "fp32"
    x = torch.randn(8, 32)
    recon, _, _ = sae(x)
    assert recon.shape == x.shape


def test_relu_arch_forward_and_eval_interface():
    sae = TopKSAE(32, 128, 4, seed=0, arch="relu", eval_k=16)
    x = torch.randn(8, 32)
    recon, z, pre = sae(x)
    assert recon.shape == x.shape and z.shape == (8, 128)
    assert (z >= 0).all()
    vals, idx = sae.feature_acts(x)
    assert vals.shape == (8, 16) and idx.shape == (8, 16)
    torch.testing.assert_close(vals, torch.topk(z, 16, dim=-1).values)


def test_relu_l1_reduces_l0(tmp_path):
    _synthetic_store(tmp_path / "store", n=4000)
    store = Store(tmp_path / "store")
    l0 = {}
    for name, coef in (("free", 0.0), ("pen", 1.0)):
        cfg = SAEConfig(d_in=32, d_sae=128, k=4, arch="relu", l1_coef=coef,
                        aux_coef=0.0, dead_tokens=100_000, lr=3e-3,
                        batch_size=512, steps=200, log_every=100)
        loader = BatchLoader(store, cfg.batch_size, seed=0)
        sae = train_sae(loader, cfg, seed=0, device="cpu",
                        out_dir=tmp_path / name, codec_name="fp32")
        x = torch.from_numpy(
            np.concatenate([xs for xs, _ in store.iter_shards()])[:1000])
        _, z, _ = sae(x)
        l0[name] = float(z.gt(0).sum(dim=1).float().mean())
    assert l0["pen"] < 0.6 * l0["free"]
    assert l0["free"] - l0["pen"] > 10


def test_relu_arch_checkpoint_roundtrip(tmp_path):
    _synthetic_store(tmp_path / "store", n=2000)
    store = Store(tmp_path / "store")
    cfg = SAEConfig(d_in=32, d_sae=64, k=4, arch="relu", l1_coef=0.1,
                    eval_k=24, dead_tokens=5000, lr=1e-3,
                    batch_size=256, steps=10, log_every=10)
    loader = BatchLoader(store, cfg.batch_size, seed=0)
    train_sae(loader, cfg, seed=0, device="cpu", out_dir=tmp_path / "r",
              codec_name="fp32")
    sae, ckpt = load_sae(tmp_path / "r")
    assert sae.arch == "relu" and sae.eval_k == 24
    vals, idx = sae.feature_acts(torch.randn(4, 32))
    assert vals.shape == (4, 24)


def test_jumprelu_forward_gates_below_threshold():
    sae = TopKSAE(32, 128, 4, seed=0, arch="jumprelu", eval_k=16)
    with torch.no_grad():
        sae.jr_bw_abs.fill_(0.1)
        sae.log_theta.fill_(0.0)  # theta = 1.0
    x = torch.randn(8, 32)
    recon, z, pre = sae(x)
    assert torch.equal(z, pre * (pre > 1.0))
    vals, idx = sae.feature_acts(x)
    assert vals.shape == (8, 16)


def test_jumprelu_theta_receives_gradient():
    sae = TopKSAE(16, 64, 4, seed=0, arch="jumprelu")
    with torch.no_grad():
        sae.jr_bw_abs.fill_(10.0)  # wide kernel: most z near threshold
        sae.log_theta.fill_(0.0)
    x = torch.randn(64, 16)
    recon, z, pre = sae(x)
    loss = (recon - x).pow(2).sum(dim=1).mean() + 0.1 * sae.l0_estimate(pre)
    loss.backward()
    assert sae.log_theta.grad is not None
    assert sae.log_theta.grad.abs().sum() > 0


def test_jumprelu_l0_penalty_raises_thresholds(tmp_path):
    _synthetic_store(tmp_path / "store", n=4000)
    store = Store(tmp_path / "store")
    l0 = {}
    for name, coef in (("free", 0.0), ("pen", 1.0)):
        cfg = SAEConfig(d_in=32, d_sae=128, k=4, arch="jumprelu",
                        l1_coef=coef, aux_coef=0.0, dead_tokens=100_000,
                        lr=3e-3, batch_size=512, steps=400, log_every=200)
        loader = BatchLoader(store, cfg.batch_size, seed=0)
        sae = train_sae(loader, cfg, seed=0, device="cpu",
                        out_dir=tmp_path / name, codec_name="fp32")
        x = torch.from_numpy(
            np.concatenate([xs for xs, _ in store.iter_shards()])[:1000])
        _, z, _ = sae(x)
        l0[name] = float(z.gt(0).sum(dim=1).float().mean())
    assert l0["pen"] < l0["free"] - 5
    assert l0["pen"] > 0.5  # sparser, but NOT collapsed


def test_jumprelu_quantile_init_starts_at_target_l0():
    sae = TopKSAE(32, 256, 8, seed=0, arch="jumprelu")
    pre = torch.randn(2048, 256) * 3.0
    sae.init_jumprelu_scale(pre)
    l0_init = float((pre > sae.theta).sum(dim=1).float().mean())
    assert 5.0 < l0_init < 12.0  # ~k=8 by construction
    assert float(sae.jr_bw_abs) > 0


def test_jumprelu_checkpoint_roundtrip(tmp_path):
    _synthetic_store(tmp_path / "store", n=2000)
    store = Store(tmp_path / "store")
    cfg = SAEConfig(d_in=32, d_sae=64, k=4, arch="jumprelu", l1_coef=0.01,
                    eval_k=24, dead_tokens=5000, lr=1e-3,
                    batch_size=256, steps=10, log_every=10)
    loader = BatchLoader(store, cfg.batch_size, seed=0)
    trained = train_sae(loader, cfg, seed=0, device="cpu",
                        out_dir=tmp_path / "j", codec_name="fp32")
    sae, _ = load_sae(tmp_path / "j")
    assert sae.arch == "jumprelu"
    assert float(sae.jr_bw_abs) == float(trained.jr_bw_abs) and float(sae.jr_bw_abs) > 0
    torch.testing.assert_close(sae.log_theta, trained.log_theta.cpu())
    x = torch.randn(4, 32)
    torch.testing.assert_close(sae(x)[0], trained(x)[0])


def test_training_deterministic_on_cpu(tmp_path):
    _synthetic_store(tmp_path / "store", n=2000)
    store = Store(tmp_path / "store")
    cfg = SAEConfig(d_in=32, d_sae=64, k=4, aux_k=8, dead_tokens=5000,
                    lr=1e-3, batch_size=256, steps=20, log_every=10)
    outs = []
    for run in ("a", "b"):
        loader = BatchLoader(store, cfg.batch_size, seed=0)
        sae = train_sae(loader, cfg, seed=0, device="cpu",
                        out_dir=tmp_path / run, codec_name="fp32")
        outs.append(sae.W_dec.detach().clone())
    torch.testing.assert_close(outs[0], outs[1], atol=0, rtol=0)
