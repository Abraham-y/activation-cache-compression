"""Data-resample floor for the linear-probe instrument.

Why this exists. The probe trainer is convex and deterministic
(probes.py), so unlike the SAE benchmark there is no seed noise, and the
paper has therefore reported probe deltas with NO same-process control.
That is the one leg of the three-instrument comparison without a floor,
in a paper whose thesis is that a reading without a control is not
interpretable. This supplies it.

Construction mirrors the steering floor: two DISJOINT halves of the same
fp32 training data, a probe trained on each, compared to each other on
the same clean fp32 validation set. That is the variation the estimator
produces from nothing but which examples it saw.

READ THE CAVEAT. Each floor probe trains on n/2 while every codec probe
trains on n. Probe direction cosine rises with training n, so a
half-sample floor is DEPRESSED relative to a like-for-like full-n
referent -- the floor looks wider than it should, and codecs therefore
clear it too easily. The floor reported here is permissive, not
conservative. This is the same half-vs-full mismatch the steering floor
has, in the opposite direction, and it is why both numbers are reported
with their n rather than as a bare ratio.

Pre-registered reading, fixed before running:
  - |acc delta| for viable codecs INSIDE the floor's acc delta
    => probe accuracy cannot separate them; "certifies" is unsupported
       and the paper must say the instrument lacks the resolution.
  - PCA-256's direction cosine BELOW the floor cosine
    => the destroyed-directions claim is licensed against a control.
  - PCA-256's direction cosine ABOVE (inside) the floor
    => the claim is not licensed and must be withdrawn.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from acc.probes import (TASKS, direction_cosine, eval_probe,  # noqa: E402
                        load_probe_data, train_probe)


def probe_floor(task_dir, n_classes, seed=0):
    task_dir = Path(task_dir)
    x, y = load_probe_data(task_dir / "master-train")
    x_val, y_val = load_probe_data(task_dir / "master-val")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(x.shape[0])
    h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]

    out = {}
    ws, stats = [], []
    for half in (h1, h2):
        xa, ya = x[half], y[half]
        st = (xa.mean(axis=0), xa.std(axis=0) + 1e-6)
        ws.append(train_probe(xa, ya, n_classes, *st))
        stats.append(st)

    out["n_full"] = int(x.shape[0])
    out["n_half"] = int(len(h1))
    out["acc_half_1"] = eval_probe(ws[0], x_val, y_val, *stats[0])
    out["acc_half_2"] = eval_probe(ws[1], x_val, y_val, *stats[1])
    out["floor_acc_delta"] = abs(out["acc_half_1"] - out["acc_half_2"])
    out["floor_direction_cos"] = direction_cosine(
        ws[0], stats[0][1], ws[1], stats[1][1])
    return out


N_SPLITS = 7  # the SAE floor gets 3 pairs and a range; a single split
              # here would leave two of three instruments uncontrolled


def main():
    root = Path("runs_m1/probes")
    results = {}
    for task, spec in TASKS.items():
        d = root / task
        if not d.exists():
            print(f"  {task}: no store, skipped")
            continue
        draws = [probe_floor(d, spec["n_classes"], seed=k)
                 for k in range(N_SPLITS)]
        accs = [x["floor_acc_delta"] for x in draws]
        coss = [x["floor_direction_cos"] for x in draws]
        r = dict(draws[0])
        r["n_splits"] = N_SPLITS
        r["floor_acc_delta"] = float(np.mean(accs))
        r["floor_acc_delta_min"] = float(np.min(accs))
        r["floor_acc_delta_max"] = float(np.max(accs))
        r["floor_direction_cos"] = float(np.mean(coss))
        r["floor_direction_cos_min"] = float(np.min(coss))
        r["floor_direction_cos_max"] = float(np.max(coss))
        print(f"  {task}: {N_SPLITS} splits — acc delta "
              f"{np.mean(accs):.4f} [{np.min(accs):.4f},{np.max(accs):.4f}]  "
              f"dir cos {np.mean(coss):.4f} "
              f"[{np.min(coss):.4f},{np.max(coss):.4f}]")
        results[task] = r
        print(f"  {task}: floor acc delta {r['floor_acc_delta']:.4f}  "
              f"floor direction cos {r['floor_direction_cos']:.4f}  "
              f"(n_half={r['n_half']:,} vs codec probes at n={r['n_full']:,})")

        # score every codec comparison already on disk against this floor
        for f in sorted(Path("runs_m1/eval").glob(f"probe_{task}__*.json")):
            c = json.load(open(f))
            delta = abs(c["acc_codec"] - c["acc_fp32"])
            print(f"      {c['codec']:9} |dacc| {delta:.4f} "
                  f"{'INSIDE floor' if delta <= r['floor_acc_delta'] else 'outside'}"
                  f"   dir cos {c['direction_cos']:.4f} "
                  f"{'BELOW floor' if c['direction_cos'] < r['floor_direction_cos'] else 'inside floor'}")

    out = Path("runs_m1/eval/probe_floor.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
