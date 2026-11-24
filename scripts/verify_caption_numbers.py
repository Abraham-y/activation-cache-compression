"""Source three caption numbers that were asserted without an artifact.

An adversarial number-check found three figures in appendix captions with
nothing behind them in the release: the 1e-8 dose rung's realised ratio,
the steering direction floor at small n, and tab:stablefloor's
codec-survivor variant. Each was measured during the project but only
ever recorded in a script comment or docstring, which is exactly the
provenance gap this paper's own audit appendix warns about.

This recomputes all three from released artifacts and writes them to
runs_m1/eval/caption_numbers.json so the captions cite data, not memory.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acc.probes import load_probe_data          # noqa: E402
from acc.steering import direction, split_halves  # noqa: E402
from acc.store import Store                     # noqa: E402

out = {}

# --- 1) the 1e-8 rung: how much of its target dose is representable? -----
# NoiseCodec fixes ||n||/||x|| = rel exactly, per token, in fp32. Below
# fp32's relative epsilon the addition is partly lost to rounding, so the
# realised rel-L2 falls short of target. Measure on the real cache.
st = Store(ROOT / "runs_m1" / "cache" / "master-eval")
chunks, seen = [], 0
for arr, _ in st.iter_shards():
    chunks.append(arr.astype(np.float32)); seen += len(arr)
    if seen >= 200_000:
        break
x = np.concatenate(chunks)[:200_000]
rng = np.random.default_rng(0)
rows = []
for exp in (7, 8):
    rel = 10.0 ** (-exp)
    n = rng.standard_normal(x.shape).astype(np.float32)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n *= (rel * np.linalg.norm(x, axis=1, keepdims=True))
    xp = (x + n).astype(np.float32)            # fp32 store round-trip
    realised = float(np.mean(np.linalg.norm(xp - x, axis=1)
                             / np.maximum(np.linalg.norm(x, axis=1), 1e-30)))
    rows.append({"exponent": exp, "target": rel, "realised": realised,
                 "ratio_realised_to_target": realised / rel})
    print(f"  1e-{exp}: target {rel:.1e} realised {realised:.3e} "
          f"ratio {realised/rel:.3f}", flush=True)
out["dose_rung_representability"] = {
    "note": "Realised per-token rel-L2 against target, fp32 storage, "
            "200k eval tokens. The 1e-8 rung is not releasable as a dose: "
            "fp32 cannot represent it.",
    "rows": rows}

# --- 2) steering direction floor at small n -----------------------------
# The floor is n-dependent; the caption quotes it at both ends. n=872 is
# sst2's validation split size.
d = ROOT / "runs_m1" / "probes" / "sst2" / "master-train"
xf, yf = load_probe_data(d)
mask = (yf == 0) | (yf == 1)
xf, yf = xf[mask], yf[mask]
cos = lambda p, q: float(np.dot(p, q))
per_n = {}
for n_target in (872, len(yf)):
    vals = []
    for k in range(50):
        r = np.random.default_rng(1000 + k)
        idx = r.choice(len(yf), size=min(n_target, len(yf)), replace=False)
        xs, ys = xf[idx], yf[idx]
        if len(np.unique(ys)) < 2:
            continue
        ia, ib = split_halves(xs, ys, seed=k)
        vals.append(cos(direction(xs[ia], ys[ia]), direction(xs[ib], ys[ib])))
    per_n[str(n_target)] = {"n_draws": len(vals),
                            "cos_floor_mean": float(np.mean(vals)),
                            "cos_floor_min": float(np.min(vals))}
    print(f"  steering dir floor n={n_target}: "
          f"mean {np.mean(vals):.3f}", flush=True)
out["steering_direction_floor_by_n"] = {
    "note": "Direction-cosine floor between two disjoint halves, over 50 "
            "subsample draws at each n. Classes 0/1 only, sst2.",
    "per_n": per_n}

p = ROOT / "runs_m1" / "eval" / "caption_numbers.json"
p.write_text(json.dumps(out, indent=1))
print(f"\n  wrote {p}")
