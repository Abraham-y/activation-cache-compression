#!/bin/bash
# THE ROTATION ABLATION: is it the rotation, or the standardization?
#
# The paper's most actionable recommendation is "never quantize per-token
# at 4 bits without a rotation; with one, 4 bits is the codec to take".
# But RotQuantCodec does per-channel STANDARDIZATION and THEN the
# rotation, so every "rotation repairs int4" number (16.0% -> 37.7%)
# measures both operations at once. main.tex never mentions
# standardization. Per-channel scaling is the textbook first-line fix for
# activation outliers and is far cheaper than a dense 768x768 decode
# matmul -- rot-int4 decodes at 86 MB/s against naive int4's 185 -- so
# which operation carries the repair is the difference between two very
# different recommendations, and one of them misprices the decode cost.
#
# std-int4 is byte-identical to rot-int4 except the rotation is the
# identity, so the arms differ by exactly the rotation.
#
# PRE-REGISTERED READING, fixed before this ran. Reference points:
# naive int4 16.0%, rot-int4 37.7%, seed floor 23.0%.
#   std-int4 >= 35%  -> STANDARDIZATION CARRIES IT. The rotation is
#     near-free to drop, the recommendation must name standardization,
#     and the decode cost of the matmul is not buying the repair.
#   std-int4 <= 25%  -> THE ROTATION CARRIES IT. The current claim stands
#     as written and the confound is ruled out.
#   25-35%           -> BOTH CONTRIBUTE. Report the split and rewrite the
#     recommendation to name both operations.
# In every branch the paper must start saying "standardization and a
# rotation", because that is what the codec does.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
R=runs_m1
CLI="$PY -m acc.cli --root $R --preset m1"
stamp(){ echo "[$(date '+%F %T')] $*"; }
free_gb(){ df -g . | awk 'NR==2{print $4}'; }

N=std-int4
if [ ! -f "$R/saes/${N}_s0/sae.pt" ]; then
  if [ ! -f "$R/cache/${N}-train/manifest.json" ]; then
    [ "$(free_gb)" -lt 90 ] && { stamp "ABORT: $(free_gb) GB free"; exit 1; }
    stamp "stage: transcode ${N} (standardize + per-token int4, NO rotation)"
    $CLI transcode --codec "$N"
  fi
  if [ ! -f "$R/eval/null_${N}_frozen.json" ]; then
    stamp "stage: null + canary on ${N}"; $CLI null --codec "$N" --canary
  fi
  if [ ! -f "$R/eval/frozen_${N}__fp32_s0.json" ]; then
    stamp "stage: eval-frozen ${N}"; $CLI eval-frozen --codec "$N"
  fi
  stamp "stage: train SAE on ${N} (~75min)"
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
stamp "=== rotation ablation complete ==="
