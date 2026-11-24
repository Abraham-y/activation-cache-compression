#!/bin/bash
# Is ratio-to-floor invariant along a THIRD axis (sparsity)?
#
# The paper reports two axes on which the calibrated ratio behaves
# differently. Across SITES it holds (rot-int4 at 1.4-1.6x floor at all
# three). Across BUDGET it does not (rot-int4 1.63x -> 1.09x), which the
# paper discloses. Sparsity is untested, and the k-sweep already showed
# the floor itself moves a lot with k: 29.2% / 23.1% / 15.0% at
# k = 16/32/64. So sparsity is the cleanest available test of whether
# calibration is doing real work.
#
# PRE-REGISTERED PREDICTIONS (anchor: at k=32, rot-int8 = 53.8% against a
# 23.0% floor = 2.34x):
#   H1  ratio-to-floor is the invariant
#       -> rot-int8 should land near 68% at k=16 and near 35% at k=64
#   H2  survival is instead roughly floor-independent
#       -> both land near 54%, and the ratio itself moves about 2x
# These are far enough apart that one run separates them. H2 would be a
# real correction to the paper's framing and belongs in Limitations; H1
# strengthens the central claim. Either way it is one sentence plus an
# appendix row, which is all the page budget allows.
#
# Design: the codec arms (rot-int8_k{16,64}_s0) already exist from the M6
# sweep. What is missing is the fp32 REFERENCE at each k, and its own
# cross-seed floor — measured on the fp32 cache with three seeds, exactly
# as every other floor in this paper is, rather than borrowing the
# rot-int8-cache floor. Borrowing it would assume the very thing an
# experiment about floors should measure.
#
# CONFOUND, carried from the M6 sweep and not fixed here: n_alive
# co-varies with k (2,878 / 4,841 / 8,240 features at k = 16/32/64), so
# the per-feature data budget differs across the three points. Report it
# with the result.
#
# Ordering: seed 0 at both k first, so the codec ratio is estimable after
# ~2h against the existing sweep floors; the extra seeds that make the
# floors properly triangulated follow. All stages guarded/restartable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"
stamp() { echo "[$(date '+%F %T')] $*"; }

R=runs_m1
CLI="$PY -m acc.cli --root $R --preset m1"

for other in run_m1.sh run_m4.sh run_arch.sh run_m6.sh run_budget_scaling.sh run_fwdprec.sh; do
  while pgrep -f "$other" > /dev/null 2>&1; do
    stamp "waiting: $other still active"; sleep 300
  done
done

train_k () {  # $1 = k, $2 = seed
  if [ ! -f "$R/saes/fp32_k$1_s$2/sae.pt" ]; then
    stamp "stage: train fp32 k=$1 seed=$2"
    $CLI train --codec fp32 --seed "$2" --k "$1" --tag "k$1"
  fi
}
pair () {     # $1 = ref, $2 = cand
  if [ ! -f "$R/eval/pair_$1__$2.json" ]; then
    stamp "stage: eval-pair $1 vs $2"
    $CLI eval-pair --ref "$1" --cand "$2"
  fi
}

# --- 1. seed 0 at both k: enough for an early codec ratio ---
for k in 16 64; do train_k "$k" 0; done
for k in 16 64; do pair "fp32_k${k}_s0" "rot-int8_k${k}_s0"; done
stamp "=== early signal available (codec arms measured) ==="
"$PY" - <<'EOF'
import json
import numpy as np
def sv(p):
    try: return json.load(open(p))["survived"]
    except Exception: return None
fl32 = np.mean([sv(f"runs_m1/eval/pair_fp32_s{a}__fp32_s{b}.json")
                for a, b in ((0, 1), (0, 2), (1, 2))])
r32 = sv("runs_m1/eval/pair_fp32_s0__rot-int8_s0.json") / fl32
print(f"\n  k=32 anchor: {r32:.2f}x floor")
for k, prov in ((16, 0.2921), (64, 0.1495)):
    s = sv(f"runs_m1/eval/pair_fp32_k{k}_s0__rot-int8_k{k}_s0.json")
    if s is None:
        continue
    print(f"  k={k:2d}: rot-int8 {100*s:.1f}%   H1 predicted ~{100*r32*prov:.0f}%   "
          f"H2 predicted ~{100*r32*fl32:.0f}%   (provisional ratio {s/prov:.2f}x "
          f"vs the sweep floor)")
print()
EOF

# --- 2. the extra seeds that make each floor properly triangulated ---
for k in 16 64; do for s in 1 2; do train_k "$k" "$s"; done; done
for k in 16 64; do
  pair "fp32_k${k}_s0" "fp32_k${k}_s1"
  pair "fp32_k${k}_s0" "fp32_k${k}_s2"
  pair "fp32_k${k}_s1" "fp32_k${k}_s2"
done

# --- 3. stable/unstable stratification at each k (nearly free once the
# three seeds exist, and it is the stratification the paper leans on) ---
for k in 16 64; do
  stamp "stage: stability stratification at k=$k"
  $CLI stability --ref "fp32_k${k}_s0" \
    --seeds "fp32_k${k}_s1" "fp32_k${k}_s2" --threshold 0.9
done

stamp "=== sparsity-invariance test complete ==="
"$PY" - <<'EOF'
import json
import numpy as np
def sv(p):
    try: return json.load(open(p))["survived"]
    except Exception: return None
def floor(tag):
    v = [sv(f"runs_m1/eval/pair_fp32{tag}_s{a}__fp32{tag}_s{b}.json")
         for a, b in ((0, 1), (0, 2), (1, 2))]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), (max(v) - min(v)) / 2, len(v)) if v else (None, None, 0)

print("\nIS RATIO-TO-FLOOR INVARIANT ACROSS SPARSITY?  (GPT-2 L6, rot-int8)")
print(f"  {'k':>4} {'floor (fp32 cache)':>22} {'rot-int8':>10} {'ratio':>8} {'n_alive':>9}")
rows = []
for k, tag, cand in ((16, "_k16", "rot-int8_k16_s0"),
                     (32, "", "rot-int8_s0"),
                     (64, "_k64", "rot-int8_k64_s0")):
    f, half, n = floor(tag)
    ref = f"fp32{tag}_s0"
    p = f"runs_m1/eval/pair_{ref}__{cand}.json"
    s = sv(p)
    if f is None or s is None:
        print(f"  {k:>4}  (incomplete)")
        continue
    d = json.load(open(p))
    print(f"  {k:>4} {100*f:>15.1f} +-{100*half:.1f} {100*s:>9.1f}% "
          f"{s/f:>7.2f}x {d['n_alive']:>9}")
    rows.append(s / f)
if len(rows) == 3:
    spread = max(rows) - min(rows)
    print(f"\n  ratio spread across a {max(rows)/min(rows):.1f}x range of k: "
          f"{spread:.2f}x ({min(rows):.2f}-{max(rows):.2f})")
    print("  VERDICT:", "H1 — ratio-to-floor survives the sparsity axis"
          if spread < 0.35 else
          "H2 — the ratio moves with sparsity; calibration is site- AND "
          "budget- AND sparsity-conditional")
    print("  (n_alive co-varies with k — a partial confound, report it)")
print()
EOF
