#!/bin/bash
# THE MINIMAL PERTURBATION: one scalar, one ulp, in the entire corpus.
#
# sparseulp6 moves ~15,000 scalars across a 20M-token store -- roughly
# three tokens per training batch carry a difference, continuously, for
# all 8000 steps. So its 70.5% leaves one thing open: does the 30-point
# drop require a perturbation PRESENT THROUGHOUT training, or is a single
# altered bit anywhere sufficient? This is the SAE-side analogue of
# Summers & Dinneen's single-weight-bit result, on the data instead.
#
# PRE-REGISTERED READING (fixed before running -- honor it):
#   <= 75%  -> THE STEP IS AT ONE BIT. A single altered scalar in 14
#     billion costs the full penalty. The paper's claim reaches its
#     strongest and simplest form and the title is exactly right.
#   >= 90%  -> DOSE-DEPENDENCE INSIDE THE SPARSE REGIME. "Discontinuous
#     at bit-identity" is then the WRONG shape: there is a real curve
#     between one scalar and a millionth of them, sparseulp6 is on its
#     slope rather than at a step, and sec:dose must be rewritten again
#     around a knee we have now bracketed.
#   75-90%  -> INTERMEDIATE. Report the bracket; the step exists but has
#     structure inside the sparse regime.
#
# Whatever it returns, report it. This is the one remaining run that can
# change the conclusion rather than re-confirm it.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
R=runs_m1
CLI="$PY -m acc.cli --root $R --preset m1"
stamp(){ echo "[$(date '+%F %T')] $*"; }
free_gb(){ df -g . | awk 'NR==2{print $4}'; }

N=oneulp
if [ ! -f "$R/saes/${N}_s0/sae.pt" ]; then
  if [ ! -f "$R/cache/${N}-train/manifest.json" ]; then
    [ "$(free_gb)" -lt 90 ] && { stamp "ABORT: $(free_gb) GB free"; exit 1; }
    stamp "stage: transcode ${N} (exactly one scalar, one ulp)"
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
stamp "=== one-ulp rung complete ==="
