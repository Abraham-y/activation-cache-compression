"""Fragility predictors: can fp32 feature statistics predict which features
survive a codec? (Duan found AUC 0.92-0.97 for weight quantization, peak
activation strongest — this is the activation-side analogue, and the basis
for mixed-precision bit allocation in Milestone 3.)

Usage: PYTHONPATH=src python3 scripts/fragility.py --root runs_m1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.cli import cache_dir, paths  # noqa: E402
from acc.sae import load_sae  # noqa: E402
from acc.store import Store  # noqa: E402
from acc.utils import get_device, write_json  # noqa: E402


def feature_stats(root, sae_name="fp32_s0", max_tokens=500_000):
    """Per-feature stats of the frozen fp32 SAE on master-eval tokens."""
    device = get_device()
    sae, _ = load_sae(paths(root)["saes"] / sae_name, device)
    m = sae.d_sae
    n_fired = np.zeros(m)
    act_sum = np.zeros(m)
    act_max = np.zeros(m)
    seen = 0
    store = Store(cache_dir(root, "fp32", "eval"))
    for i in range(len(store.shards)):
        if seen >= max_tokens:
            break
        x, _ = store.read_shard(i)
        x = x[: max_tokens - seen]
        seen += x.shape[0]
        for c0 in range(0, x.shape[0], 8192):
            xb = torch.from_numpy(x[c0 : c0 + 8192]).to(device)
            vals, idx = sae.feature_acts(xb)
            v = vals.cpu().numpy().ravel()
            j = idx.cpu().numpy().ravel()
            fired = v > 0
            np.add.at(n_fired, j[fired], 1)
            np.add.at(act_sum, j[fired], v[fired])
            np.maximum.at(act_max, j[fired], v[fired])
    rate = n_fired / seen
    mean_act = np.divide(act_sum, n_fired, out=np.zeros_like(act_sum),
                         where=n_fired > 0)
    return {"rate": rate, "mean_act": mean_act, "peak_act": act_max}


def auc(score, y):
    """Rank-based AUC (Mann-Whitney)."""
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def logistic_auc(X, y, seed=0, folds=5):
    """Cross-validated AUC of a logistic model on standardized features."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    scores = np.zeros(len(y))
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
        Xtr = torch.from_numpy((X[train] - mu) / sd)
        ytr = torch.from_numpy(y[train].astype(np.float64))
        w = torch.zeros(X.shape[1] + 1, dtype=torch.float64,
                        requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = Xtr @ w[:-1] + w[-1]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, ytr)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            scores[test] = (
                torch.from_numpy((X[test] - mu) / sd) @ w[:-1] + w[-1]
            ).numpy()
    return auc(scores, y.astype(bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--codecs", nargs="+",
                    default=["rot-int6", "rot-int4", "int4-o16", "int6"])
    args = ap.parse_args()

    stats = feature_stats(args.root)
    eps = 1e-12
    feats = {
        "log_rate": np.log(stats["rate"] + eps),
        "log_mean_act": np.log(stats["mean_act"] + eps),
        "log_peak_act": np.log(stats["peak_act"] + eps),
    }
    X = np.stack(list(feats.values()), axis=1)

    results = {}
    for codec in args.codecs:
        z = np.load(Path(args.root) / "eval" /
                    f"pair_fp32_s0__{codec}_s0.npz")
        alive = z["alive"]
        y = (np.nan_to_num(z["best_corr"]) > 0.9)[alive]
        Xa = X[alive]
        marginals = {k: auc(v[alive], y) for k, v in feats.items()}
        cv = logistic_auc(Xa, y)
        results[codec] = {
            "n_alive": int(alive.sum()),
            "survival_rate": float(y.mean()),
            "cv_auc_all_feats": cv,
            "marginal_auc": marginals,
        }
        marg = ", ".join(f"{k} {v:.3f}" for k, v in marginals.items())
        print(f"{codec:>10}: cv-AUC {cv:.3f}  ({marg})")
    write_json(Path(args.root) / "eval" / "fragility.json", results)


if __name__ == "__main__":
    main()
