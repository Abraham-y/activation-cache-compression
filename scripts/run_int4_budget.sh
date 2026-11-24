#!/bin/bash
# Is the below-floor verdict budget-invariant?
#
# The sharpest structural point from an internal audit: the below-floor verdict is the
# ONE thing the floor uniquely buys (no codec-relative normalizer can
# express "this codec does more damage than changing the seed does"),
# and it is established only at the 1x budget — which §6 then shows is
# the tier where the floor is largely an under-training artifact. If the
# verdict does not survive to 10x, the protocol's unique deliverable is
# a threshold that moves with a budget nobody would train at.
#
# The floors and the fp32/rot-int8/rot-int4 arms already exist at both
# tiers from run_budget_scaling.sh; only the naive int4 arms are
# missing. The int4 cache also already exists, so this is training only.
#
# PRE-REGISTERED: at 1x, int4 sits at 16.0% against a 23.0% floor =
# 0.70x. The floor rises to 33.3% (3x) and 42.3% (10x). If the verdict
# is budget-invariant, int4 should stay below ~1.0x at both. If int4's
# survival rises faster than the floor, the verdict is a 1x artifact and
# §5's claim about what the floor uniquely buys needs scoping.
#
# ~2.5h at 3x + ~8h at 10x. Guarded/restartable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"
stamp() { echo "[$(date '+%F %T')] $*"; }

R=runs_m1
CLI="$PY -m acc.cli --root $R --preset m1"

for other in run_m1.sh run_m4.sh run_arch.sh run_m6.sh run_fwdprec.sh \
             run_ksweep_codec.sh run_pythia_frozen.sh run_budget_scaling.sh; do
  while pgrep -f "$other" > /dev/null 2>&1; do
    stamp "waiting: $other still active"; sleep 300
  done
done

[ -f "$R/cache/int4-train/manifest.json" ] || { stamp "FATAL: int4 cache missing"; exit 1; }

for spec in "b3x 24000" "b10x 80000"; do
  set -- $spec
  TAG=$1; STEPS=$2
  if [ ! -f "$R/saes/int4_${TAG}_s0/sae.pt" ]; then
    stamp "stage: train int4 at $TAG ($STEPS steps)"
    $CLI train --codec int4 --seed 0 --steps "$STEPS" --tag "$TAG"
  fi
  if [ ! -f "$R/eval/pair_fp32_${TAG}_s0__int4_${TAG}_s0.json" ]; then
    stamp "stage: eval-pair fp32_${TAG}_s0 vs int4_${TAG}_s0"
    $CLI eval-pair --ref "fp32_${TAG}_s0" --cand "int4_${TAG}_s0"
  fi
done

stamp "===== is the below-floor verdict budget-invariant? ====="
"$PY" - <<'EOF'
import json
from pathlib import Path
import numpy as np
def sv(p):
    q = Path("runs_m1/eval") / p
    return json.loads(q.read_text())["survived"] if q.exists() else None
print(f"\n  {'tier':<6}{'floor':>8}{'int4':>9}{'x floor':>10}{'verdict':>14}")
for tag, label in (("", "1x"), ("_b3x", "3x"), ("_b10x", "10x")):
    fl = [sv(f"pair_fp32{tag}_s{a}__fp32{tag}_s{b}.json")
          for a, b in ((0, 1), (0, 2), (1, 2))]
    fl = [x for x in fl if x is not None]
    s = sv(f"pair_fp32{tag}_s0__int4{tag}_s0.json")
    if not fl or s is None:
        print(f"  {label:<6}  (incomplete)")
        continue
    f = float(np.mean(fl))
    print(f"  {label:<6}{100*f:>7.1f}%{100*s:>8.1f}%{s/f:>9.2f}x"
          f"{'BELOW floor' if s < f else 'ABOVE floor':>14}")
print()
EOF
stamp "===== complete ====="
