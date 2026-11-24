"""Report the two 2026-08-15 controls against their pre-registered readings.

Reads only eval JSONs, so it runs after the caches are reclaimed.
"""
import json
import sys
from pathlib import Path

EV = Path("runs_m1/eval")


def load(name):
    p = EV / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(d, key="survived"):
    return 100 * d[key] if d else float("nan")


def main():
    print("\n" + "=" * 72)
    print("A. FORWARD PRECISION (bf16 forward, fp32 storage) AT THREE SEEDS")
    print("=" * 72)
    fwd, rot = [], []
    for s in (0, 1, 2):
        f = load(f"pair_fp32_s{s}__fwdbf16_s{s}")
        # rot-int8's own three-seed range, built the same seed-matched way
        r = load(f"pair_fp32_s{s}__rot-int8_s{s}") or \
            load(f"pair_fp32_s{s}__rot-int8_k32_s{s}")
        if f:
            fwd.append(pct(f))
        if r:
            rot.append(pct(r))
        print(f"  seed {s}:  bf16-forward {pct(f):5.1f}%    "
              f"rot-int8 storage {pct(r):5.1f}%")
    if len(fwd) == 3 and len(rot) == 3:
        print(f"\n  bf16-forward range {min(fwd):.1f}-{max(fwd):.1f}% "
              f"(mean {sum(fwd)/3:.1f}%)")
        print(f"  rot-int8     range {min(rot):.1f}-{max(rot):.1f}% "
              f"(mean {sum(rot)/3:.1f}%)")
        overlap = min(fwd) <= max(rot) and min(rot) <= max(fwd)
        print(f"  ranges {'OVERLAP' if overlap else 'ARE DISJOINT'} -> "
              f"{'parity claim holds at three seeds' if overlap else 'ordered; state the direction'}")
    else:
        print("\n  (incomplete: still running)")

    print("\n" + "=" * 72)
    print("B. DOSE-RESPONSE BELOW fp16")
    print("=" * 72)
    print(f"  {'dose':>10} {'rel-L2':>10} {'frozen r':>11} {'RETRAINED':>11}")
    ceil = load("pair_fp32_s0__fp32_s0_r1")
    print(f"  {'zero':>10} {0.0:>10.1e} {1.0:>11.6f} {pct(ceil):>10.1f}%"
          "   <- bit-exact run-noise ceiling")
    rows = []
    for e in (7, 6, 5, 4):
        p = load(f"pair_fp32_s0__noise{e}_s0")
        fr = load(f"frozen_noise{e}__fp32_s0")
        if not p:
            continue
        rel = fr["direct"]["rel_l2_mean"] if fr else float("nan")
        r = fr["mean_corr"] if fr else float("nan")
        rows.append((e, rel, pct(p)))
        print(f"  {'1e-' + str(e):>10} {rel:>10.2e} {r:>11.6f} {pct(p):>10.1f}%")
    f16 = load("pair_fp32_s0__fp16_s0")
    ff16 = load("frozen_fp16__fp32_s0")
    print(f"  {'fp16':>10} {ff16['direct']['rel_l2_mean'] if ff16 else float('nan'):>10.2e} "
          f"{ff16['mean_corr'] if ff16 else float('nan'):>11.6f} {pct(f16):>10.1f}%"
          "   <- smallest dose previously tested")
    floor = [load(f"pair_fp32_s0__fp32_s{s}") for s in (1, 2)]
    floor = [pct(d) for d in floor if d]
    if floor:
        print(f"  {'seed floor':>10} {'-':>10} {'-':>11} "
              f"{sum(floor)/len(floor):>10.1f}%   <- codec-free lower bound")

    if rows:
        print("\n  PRE-REGISTERED READING:")
        deepest = min(rows, key=lambda r: r[1])
        s = deepest[2]
        if s >= 97.0:
            v = ("survival returns to the ceiling -> fp16 sits on a REAL "
                 "shoulder.\n  Damage is information loss, not training "
                 "chaos. Chaos objection dead.")
        elif s < 85.0:
            v = ("survival stays far below ceiling at a negligible dose ->\n"
                 "  target is CHAOS-DOMINATED. Survival is only meaningful "
                 "against a\n  same-process control; the bits framing must "
                 "be retired.")
        else:
            v = ("intermediate -> the shoulder is real but soft; report the\n"
                 "  curve and sharpen the 'damage tracks magnitude, not "
                 "source' claim.")
        print(f"  at rel-L2 {deepest[1]:.1e}, retrained survival {s:.1f}%:\n  {v}")
    print()


if __name__ == "__main__":
    sys.exit(main())
