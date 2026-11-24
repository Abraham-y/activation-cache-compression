"""Stratify on a SINGLE-RUN statistic, not a run-to-run comparison.

The strongest objection from an internal audit: our stable set is selected by the outcome of
a seed comparison and its floor is another seed comparison, so selector
and denominator share the treatment type while the codec numerator does
not. Conditioning on codec survivors instead returns rot-int4 from
1.18x to ~1.60x, i.e. the stratification arm was doing the work.

The clean test needs a conditioning variable computed from ONE run, with
no comparison in it at all. The fragility analysis (scripts/fragility.py)
already showed per-feature survival is predictable from single-run fp32
statistics at CV AUC ~0.84, with mean activation the strongest marginal.
So we stratify directly on those statistics --- no fitted model, no
labels, nothing that touches a second run --- and recompute the floor and
every codec on that population.

Population SIZE is matched to the seed-defined set so the comparison is
at equal selection depth (a lesson from the budget-tier control).

Usage: PYTHONPATH=src python scripts/stratify_singlerun.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.sae import load_sae  # noqa: E402
from acc.store import Store  # noqa: E402
from acc.utils import get_device, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
THRESH = 0.9
CODECS = ["fp16", "rot-int8", "rot-int4", "int4"]


def feature_stats(root, sae_name, max_tokens, device):
    """Per-feature mean activation and peak, from ONE run's own forward.
    Nothing here compares two runs."""
    sae, _ = load_sae(ROOT / root / "saes" / sae_name, device)
    master = Store(ROOT / root / "cache" / "master-eval")
    m = sae.d_sae
    tot = torch.zeros(m, dtype=torch.float32, device=device)  # MPS: no f64
    peak = torch.zeros(m, dtype=torch.float32, device=device)
    n = 0
    for x, _ in master.iter_shards():
        if n >= max_tokens:
            break
        x = x[: max_tokens - n]
        n += x.shape[0]
        for c0 in range(0, x.shape[0], 8192):
            xb = torch.from_numpy(x[c0:c0 + 8192]).to(device)
            vals, idx = sae.feature_acts(xb)
            z = torch.zeros(xb.shape[0], m, device=device)
            z.scatter_(1, idx, vals.clamp(min=0))
            tot += z.sum(0)
            peak = torch.maximum(peak, z.max(0).values)
    # promote on the host: MPS has no float64
    return ((tot / max(n, 1)).cpu().numpy().astype("float64"),
            peak.cpu().numpy().astype("float64"), n)


def main():
    device = get_device()
    root = "runs_m1"
    ev = ROOT / root / "eval"
    z1 = np.load(ev / "pair_fp32_s0__fp32_s1.npz")
    z2 = np.load(ev / "pair_fp32_s0__fp32_s2.npz")
    alive = z1["alive"]
    c1, c2 = np.nan_to_num(z1["best_corr"]), np.nan_to_num(z2["best_corr"])
    seed_set = (c1 > THRESH) & (c2 > THRESH) & alive
    n_target = int(seed_set.sum())

    mean_act, peak_act, n_tok = feature_stats(root, "fp32_s0", 200_000, device)
    codec = {c: np.nan_to_num(np.load(ev / f"pair_fp32_s0__{c}_s0.npz")["best_corr"])
             for c in CODECS}

    out = {"n_eval_tokens": n_tok, "threshold": THRESH,
           "n_target": n_target,
           "note": ("Strata built from single-run fp32 feature statistics "
                    "(no run-to-run comparison anywhere in the selector). "
                    "Floor is the s0-vs-s1 seed pair evaluated on the same "
                    "population. Sizes matched to the seed-defined set."),
           "arms": []}
    arms = [("seed-defined (for reference)", seed_set),
            ("top mean activation", None), ("top peak activation", None)]
    for name, stat in (("top mean activation", mean_act),
                       ("top peak activation", peak_act)):
        s = np.where(alive, stat, -np.inf)
        cut = np.argsort(s)[::-1][:n_target]
        mask = np.zeros_like(alive)
        mask[cut] = True
        arms[[a[0] for a in arms].index(name)] = (name, mask)
    for name, mask in arms:
        floor = float((c1[mask] > THRESH).mean())
        row = {"arm": name, "n": int(mask.sum()), "floor": floor,
               "codecs": {c: {"survived": float((codec[c][mask] > THRESH).mean())}
                          for c in CODECS}}
        for c in CODECS:
            row["codecs"][c]["ratio_to_floor"] = (
                row["codecs"][c]["survived"] / floor if floor else None)
        out["arms"].append(row)
    o = ev / "stratify_singlerun.json"
    write_json(o, out)
    print(f"{'arm':<30}{'n':>6}{'floor':>8}" +
          "".join(f"{c:>18}" for c in CODECS))
    for r in out["arms"]:
        print(f"{r['arm']:<30}{r['n']:>6}{100*r['floor']:>7.1f}%" +
              "".join(f"{100*r['codecs'][c]['survived']:>10.1f}%"
                      f"={r['codecs'][c]['ratio_to_floor']:>5.2f}x"
                      for c in CODECS))
    print(f"wrote {o}")




def injectivity():
    """Does many-to-one matching inflate survival?

    Best-match survival lets several reference features claim the same
    candidate. If that happened often, survival would be inflated. It
    does not: matching is 96-99.9% injective for the floor and the viable
    codecs, and what inflation exists is LARGER for the worse codecs, so
    the metric is conservative for every claim we make with it.
    """
    ev = ROOT / "runs_m1" / "eval"
    rows = {}
    for label, f in (("fp32_s1", "pair_fp32_s0__fp32_s1.npz"),
                     ("fp16", "pair_fp32_s0__fp16_s0.npz"),
                     ("rot-int8", "pair_fp32_s0__rot-int8_s0.npz"),
                     ("rot-int4", "pair_fp32_s0__rot-int4_s0.npz"),
                     ("int4", "pair_fp32_s0__int4_s0.npz"),
                     ("pca256", "pair_fp32_s0__pca256_s0.npz")):
        p = ev / f
        if not p.exists():
            continue
        z = np.load(p)
        surv = (np.nan_to_num(z["best_corr"]) > THRESH) & z["alive"]
        claimed = z["best_idx"][surv]
        u, counts = np.unique(claimed, return_counts=True)
        rows[label] = {"survivors": int(surv.sum()), "distinct": int(len(u)),
                       "injective_frac": float(len(u) / max(surv.sum(), 1)),
                       "max_multiplicity": int(counts.max())}
    write_json(ev / "matching_injectivity.json", rows)
    print("\ninjectivity of best-match survival:")
    for k, v in rows.items():
        print(f"  {k:<10} {v['survivors']:>5} survivors, "
              f"{100*v['injective_frac']:.1f}% injective, "
              f"max multiplicity {v['max_multiplicity']}")


if __name__ == "__main__":
    main()
    injectivity()
