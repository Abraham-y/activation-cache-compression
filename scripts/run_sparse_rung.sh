#!/bin/bash
# THE STEP-VS-KNEE EXPERIMENT.
#
# sec:dose reports that a dense perturbation 2000x below fp16 still costs
# ~34pp of retrained agreement, and explicitly declines to say whether the
# approach to the bit-exact ceiling is CONTINUOUS (a knee somewhere below
# 1e-7) or a STEP at exactly zero. A dense ladder cannot tell them apart,
# because fp32's own resolution bounds a dense perturbation below at about
# one ulp (~6e-8) and our deepest rung already sits there.
#
# A SPARSE perturbation escapes that bound: touch a fraction f of scalars
# and mean relative L2 falls as sqrt(f). sparseulp6 perturbs 1e-6 of
# entries by exactly one ulp each, landing at rel-L2 ~1.4e-12 -- five
# decades below the deepest dense rung and about as close to zero as fp32
# permits while still being nonzero.
#
# PRE-REGISTERED READING (fixed before running -- honor it):
#   >= 90% retrained agreement  -> CONTINUOUS. A knee exists between
#     ~1e-12 and 1e-7, the dense bound genuinely prevents any storage
#     codec from reaching it, and "saturates before losslessness" is
#     established rather than asserted. Report the knee as bracketed.
#   <= 75%  -> STEP AT ZERO. Any perturbation at all costs the full
#     penalty; this is the SAE-side Summers & Dinneen phenomenon. Frame
#     the result as "SAE training is chaotic with respect to its training
#     data" rather than as saturation.
#   75-90%  -> INTERMEDIATE. The knee is inside our reach; report the
#     bracket and say the shape is partially resolved.
#
# This is also the mixed-precision case sec:dose names as untested: a
# codec keeping most channels exact and quantizing a few has this shape.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
R=runs_m1
CLI="$PY -m acc.cli --root $R --preset m1"
stamp(){ echo "[$(date '+%F %T')] $*"; }
free_gb(){ df -g . | awk 'NR==2{print $4}'; }

N=sparseulp6
if [ ! -f "$R/saes/${N}_s0/sae.pt" ]; then
  if [ ! -f "$R/cache/${N}-train/manifest.json" ]; then
    [ "$(free_gb)" -lt 90 ] && { stamp "ABORT: $(free_gb) GB free, need ~90"; exit 1; }
    stamp "stage: transcode ${N} (fp32 + one ulp on 1e-6 of entries)"
    $CLI transcode --codec "$N"
  fi
  if [ ! -f "$R/eval/null_${N}_frozen.json" ]; then
    stamp "stage: null + canary on ${N}"
    $CLI null --codec "$N" --canary
  fi
  if [ ! -f "$R/eval/frozen_${N}__fp32_s0.json" ]; then
    stamp "stage: eval-frozen ${N}"
    $CLI eval-frozen --codec "$N"
  fi
  stamp "stage: train SAE on ${N} (~55min)"
  $CLI train --codec "$N" --seed 0
fi
if [ ! -f "$R/eval/pair_fp32_s0__${N}_s0.json" ]; then
  stamp "stage: eval-pair fp32_s0 vs ${N}_s0"
  $CLI eval-pair --ref fp32_s0 --cand "${N}_s0"
fi
if [ -d "$R/cache/${N}-train" ] && [ -f "$R/saes/${N}_s0/sae.pt" ] \
   && [ -f "$R/eval/pair_fp32_s0__${N}_s0.json" ]; then
  stamp "reclaim: rm ${N} stores ($(du -sh $R/cache/${N}-train | cut -f1))"
  rm -rf "$R/cache/${N}-train" "$R/cache/${N}-eval"
fi
stamp "=== sparse rung complete ==="
