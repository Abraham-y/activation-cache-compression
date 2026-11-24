"""TopK sparse autoencoder (Gao et al. 2024 style) and trainer.

Determinism contract: parameters are initialized on CPU from a seeded
generator before moving to the device, and the data order is fixed by the
loader seed. Two runs with the same (codec, seed) differ only by device
kernel nondeterminism, which the run-noise control quantifies.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .store import BatchLoader
from .utils import append_jsonl, write_json


@dataclass
class SAEConfig:
    d_in: int = 768
    d_sae: int = 12288
    k: int = 32
    aux_k: int = 512
    aux_coef: float = 1 / 32
    dead_tokens: int = 1_000_000
    lr: float = 3e-4
    batch_size: int = 4096
    steps: int = 8000
    log_every: int = 100
    arch: str = "topk"        # "topk" | "relu" (L1) | "jumprelu" (L0, STE)
    l1_coef: float = 0.0      # relu: L1 coef; jumprelu: L0 coef (lambda)
    eval_k: int = 192         # non-topk: cap on feature_acts sparsity
    jr_bandwidth: float = 0.05  # jumprelu STE kernel width, fraction of pre-act std
    jr_theta_lr_mult: float = 30.0  # jumprelu: log_theta learning-rate multiplier


class _JumpReLUFn(torch.autograd.Function):
    """z * (z > theta), straight-through gradient for theta via a
    rectangular kernel of width `bandwidth` around the threshold
    (Rajamanoharan et al. 2024, Gemma Scope)."""

    @staticmethod
    def forward(ctx, z, theta, bandwidth):
        gate = (z > theta).to(z.dtype)
        ctx.save_for_backward(z, theta, gate)
        ctx.bandwidth = bandwidth
        return z * gate

    @staticmethod
    def backward(ctx, grad_out):
        z, theta, gate = ctx.saved_tensors
        bw = ctx.bandwidth
        grad_z = grad_out * gate
        near = ((z - theta).abs() < bw / 2).to(z.dtype)
        grad_theta = -(theta / bw) * near * grad_out
        return grad_z, grad_theta.sum(dim=0), None


class _StepL0Fn(torch.autograd.Function):
    """(z > theta) with the same kernel STE for theta — the L0 penalty."""

    @staticmethod
    def forward(ctx, z, theta, bandwidth):
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        return (z > theta).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        z, theta = ctx.saved_tensors
        bw = ctx.bandwidth
        near = ((z - theta).abs() < bw / 2).to(z.dtype)
        return None, (-(1.0 / bw) * near * grad_out).sum(dim=0), None


class TopKSAE(nn.Module):
    """SAE with TopK, ReLU+L1, or JumpReLU activation. One class so the
    whole eval stack (sparse (vals, idx) interface) works for all; `arch`
    is stored in the checkpoint config."""

    def __init__(self, d_in: int, d_sae: int, k: int, seed: int = 0,
                 arch: str = "topk", eval_k: int = 192,
                 jr_bandwidth: float = 0.05):
        super().__init__()
        assert arch in ("topk", "relu", "jumprelu")
        self.d_in, self.d_sae, self.k = d_in, d_sae, k
        self.arch = arch
        self.eval_k = min(eval_k, d_sae) if arch != "topk" else k
        self.jr_bandwidth = jr_bandwidth
        gen = torch.Generator().manual_seed(seed)
        w = torch.randn(d_sae, d_in, generator=gen)
        w = w / w.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(w)                    # [d_sae, d_in] rows unit-norm
        self.W_enc = nn.Parameter(w.t().clone())        # tied init
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if arch == "jumprelu":
            # log-threshold parameterization keeps theta positive
            self.log_theta = nn.Parameter(torch.full((d_sae,), -2.0))
            # absolute STE kernel width, set from pre-act scale at step 0
            # (jr_bandwidth is a FRACTION of pre-act std; Gemma Scope's
            # eps=0.001 assumes normalized activations, ours are raw)
            self.register_buffer("jr_bw_abs", torch.tensor(0.0))

    @property
    def theta(self):
        return self.log_theta.exp()

    @torch.no_grad()
    def init_jumprelu_scale(self, pre_sample: torch.Tensor):
        """Set kernel width from pre-act scale and initialize each
        feature's threshold at the quantile that puts expected L0 at k
        immediately. Adam-normalized theta steps move ~lr log-units per
        step regardless of lambda, so starting orders of magnitude below
        the sparse regime (as a tiny-constant init does) is unreachable in
        our training budget — start AT the regime and let training refine."""
        s = pre_sample.std()
        self.jr_bw_abs.fill_(float(self.jr_bandwidth * s))
        q = 1.0 - self.k / self.d_sae
        theta0 = torch.quantile(pre_sample.float(), q, dim=0)
        theta0 = theta0.clamp(min=float(0.001 * s))
        self.log_theta.copy_(theta0.log())

    def pre_acts(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def topk(self, pre):
        vals, idx = torch.topk(pre, self.k, dim=-1)
        return vals.clamp(min=0.0), idx

    def decode(self, vals, idx):
        z = torch.zeros(vals.shape[0], self.d_sae, device=vals.device,
                        dtype=vals.dtype)
        z.scatter_(1, idx, vals)
        return z @ self.W_dec + self.b_dec, z

    def forward(self, x):
        pre = self.pre_acts(x)
        if self.arch == "topk":
            vals, idx = self.topk(pre)
            recon, z = self.decode(vals, idx)
        elif self.arch == "jumprelu":
            z = _JumpReLUFn.apply(pre, self.theta, float(self.jr_bw_abs))
            recon = z @ self.W_dec + self.b_dec
        else:
            z = pre.clamp(min=0.0)
            recon = z @ self.W_dec + self.b_dec
        return recon, z, pre

    def l0_estimate(self, pre):
        """Differentiable (via theta-STE) per-token L0 — jumprelu only."""
        gates = _StepL0Fn.apply(pre, self.theta, float(self.jr_bw_abs))
        return gates.sum(dim=1).mean()

    @torch.no_grad()
    def feature_acts(self, x):
        """Sparse feature activations for eval: (vals [n,k], idx [n,k]).
        For non-topk archs, the top eval_k of the (typically sparser)
        output; zero entries are dropped downstream."""
        pre = self.pre_acts(x)
        if self.arch == "topk":
            return self.topk(pre)
        if self.arch == "jumprelu":
            z = pre * (pre > self.theta)
        else:
            z = pre.clamp(min=0.0)
        return torch.topk(z, self.eval_k, dim=-1)

    @torch.no_grad()
    def project_decoder_grads(self):
        if self.W_dec.grad is None:
            return
        parallel = (self.W_dec.grad * self.W_dec).sum(dim=1, keepdim=True)
        self.W_dec.grad -= parallel * self.W_dec

    @torch.no_grad()
    def renorm_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)


def train_sae(loader: BatchLoader, cfg: SAEConfig, seed: int, device: str,
              out_dir: str | Path, codec_name: str = "?",
              init_seed: int | None = None) -> TopKSAE:
    # init_seed, when set, decouples parameter initialization from the
    # loader's data order (which stays on `seed`): the init-only floor.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    sae = TopKSAE(cfg.d_in, cfg.d_sae, cfg.k,
                  seed=seed if init_seed is None else init_seed,
                  arch=cfg.arch,
                  eval_k=cfg.eval_k, jr_bandwidth=cfg.jr_bandwidth).to(device)
    if cfg.arch == "jumprelu":
        theta_params = [sae.log_theta]
        others = [p for n, p in sae.named_parameters() if n != "log_theta"]
        opt = torch.optim.Adam([
            {"params": others, "lr": cfg.lr},
            {"params": theta_params, "lr": cfg.lr * cfg.jr_theta_lr_mult},
        ])
    else:
        opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    tokens_since_fired = torch.zeros(cfg.d_sae, dtype=torch.long, device=device)
    dead_frac = 0.0

    batches = loader.batches(epochs=None)
    t_start = time.perf_counter()
    for step in range(cfg.steps):
        x = next(batches)
        if step == 0:
            with torch.no_grad():
                sae.b_dec.copy_(x.mean(dim=0))
                if cfg.arch == "jumprelu":
                    sae.init_jumprelu_scale(sae.pre_acts(x))

        recon, z, pre = sae(x)
        mse = (recon - x).pow(2).sum(dim=1).mean()

        fired = z.detach().gt(0).any(dim=0)
        tokens_since_fired += cfg.batch_size
        tokens_since_fired[fired] = 0
        dead = tokens_since_fired > cfg.dead_tokens
        n_dead = int(dead.sum())

        loss = mse
        if cfg.arch == "relu" and cfg.l1_coef > 0:
            loss = loss + cfg.l1_coef * z.sum(dim=1).mean()
        if cfg.arch == "jumprelu" and cfg.l1_coef > 0:
            loss = loss + cfg.l1_coef * sae.l0_estimate(pre)
        if cfg.arch == "topk" and n_dead > 0 and cfg.aux_coef > 0:
            err = x - recon.detach()
            k_aux = min(cfg.aux_k, n_dead)
            pre_dead = pre.masked_fill(~dead.unsqueeze(0), float("-inf"))
            aux_vals, aux_idx = torch.topk(pre_dead, k_aux, dim=-1)
            aux_vals = aux_vals.clamp(min=0.0)
            z_aux = torch.zeros_like(z)
            z_aux.scatter_(1, aux_idx, aux_vals)
            recon_aux = z_aux @ sae.W_dec
            aux_mse = (recon_aux - err).pow(2).sum(dim=1).mean()
            if torch.isfinite(aux_mse):
                loss = loss + cfg.aux_coef * aux_mse

        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.project_decoder_grads()
        opt.step()
        sae.renorm_decoder()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                var = x.var(dim=0).sum()
                fvu = float(mse / var.clamp(min=1e-9))
                l0 = float(z.gt(0).sum(dim=1).float().mean())
            dead_frac = n_dead / cfg.d_sae
            elapsed = time.perf_counter() - t_start
            append_jsonl(log_path, {
                "step": step,
                "mse": float(mse.detach()),
                "fvu": fvu,
                "l0": l0,
                "dead_frac": dead_frac,
                "tokens_seen": (step + 1) * cfg.batch_size,
                "tokens_per_s": (step + 1) * cfg.batch_size / max(elapsed, 1e-9),
                **loader.stats(),
            })

    ckpt = {
        "state_dict": {k: v.cpu() for k, v in sae.state_dict().items()},
        "cfg": asdict(cfg),
        "seed": seed,
        "init_seed": init_seed,
        "codec": codec_name,
    }
    torch.save(ckpt, out_dir / "sae.pt")
    write_json(out_dir / "summary.json", {
        "codec": codec_name, "seed": seed, "init_seed": init_seed,
        "cfg": asdict(cfg),
        "final_dead_frac": dead_frac,
        "train_seconds": time.perf_counter() - t_start,
        "loader": loader.stats(),
    })
    return sae


def load_sae(path: str | Path, device: str = "cpu") -> tuple[TopKSAE, dict]:
    ckpt = torch.load(Path(path) / "sae.pt", map_location="cpu", weights_only=True)
    cfg = ckpt["cfg"]
    sae = TopKSAE(cfg["d_in"], cfg["d_sae"], cfg["k"],
                  arch=cfg.get("arch", "topk"),
                  eval_k=cfg.get("eval_k", 192),
                  jr_bandwidth=cfg.get("jr_bandwidth", 0.05))
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(device)
    return sae, ckpt
