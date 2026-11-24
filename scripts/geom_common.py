"""Shared helpers for the geometry follow-up (scripts/geom_*.py): load an
SAE, stream eval-cache tokens by global index range, and the seed-stability
labels the paper uses (best match r > 0.9 in both other seeds)."""
from pathlib import Path

import numpy as np
import torch

from acc.sae import load_sae
from acc.store import Store
from acc.utils import get_device

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "runs_m1" / "eval"
SAES = ROOT / "runs_m1" / "saes"
CACHE = ROOT / "runs_m1" / "cache"
SCORE = (0, 500_000)            # the paper's eval tokens (pair and frozen tests)
HELDOUT = (500_000, 1_000_125)  # the rest of the eval cache


def device():
    return get_device()


def sae(name, dev=None):
    s, ck = load_sae(SAES / name, dev or device())
    return s, ck


def iter_range(store_name, lo, hi, chunk=8192):
    """Yield (global_start, x[n, d] float32) over eval tokens [lo, hi)."""
    st = Store(CACHE / store_name)
    off = 0
    for i, sh in enumerate(st.shards):
        n = sh["n_tokens"]
        a, b = max(lo, off), min(hi, off + n)
        if a < b:
            x, _ = st.read_shard(i)
            x = x[a - off:b - off]
            for c0 in range(0, x.shape[0], chunk):
                yield a + c0, np.ascontiguousarray(x[c0:c0 + chunk])
        off += n
        if off >= hi:
            break


def pair(ref, cand):
    return np.load(EVAL / f"pair_{ref}__{cand}.npz")


def stability(ref, others, thr=0.9):
    """(alive mask over d_sae, stable mask over d_sae), as cmd_stability."""
    stable, alive = None, None
    for o in others:
        z = pair(ref, o)
        ok = np.nan_to_num(z["best_corr"]) > thr
        stable = ok if stable is None else stable & ok
        alive = z["alive"]
    return alive, stable


def survived(ref, cand, thr=0.9):
    z = pair(ref, cand)
    return z["alive"], np.nan_to_num(z["best_corr"]) > thr


@torch.no_grad()
def firing_alive(s, store="master-eval", lo=SCORE[0], hi=SCORE[1], dev=None):
    """Firing rate > 1e-3 on [lo, hi): the paper's alive definition."""
    dev = dev or device()
    fire = torch.zeros(s.d_sae, dtype=torch.long, device=dev)
    n = 0
    for _, x in iter_range(store, lo, hi):
        vals, idx = s.feature_acts(torch.from_numpy(x).to(dev))
        f = torch.zeros(x.shape[0], s.d_sae, device=dev, dtype=torch.bool)
        f.scatter_(1, idx, vals > 0)
        fire += f.sum(0)
        n += x.shape[0]
    return (fire.float() / n).cpu().numpy() > 1e-3
