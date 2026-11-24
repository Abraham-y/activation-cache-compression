#!/bin/bash
# THE LOCALITY CONTROL: same count, concentrated instead of spread.
#
# The paper currently claims the response is governed by how MANY stored
# values change. But sparseulp6's 15,360 changes are spread over 20M
# tokens -- one perturbed token per ~1,300, so essentially EVERY optimizer
# step meets a perturbation -- while oneulp's single change touches one
# step. Count and steps-affected move together, so the claim is
# confounded, and the alternative ("a data perturbation matters in
# proportion to how often the optimizer meets it") is the one the
# nondeterminism literature would predict.
#
# This holds count fixed at 15,360 and confines it to a single shard.
#
# PRE-REGISTERED READING (fixed before running -- honor it):
#   >= 90%  -> FREQUENCY, NOT COUNT. Concentrating the same changes
#     recovers the loss, so the governing variable is the fraction of
#     training steps perturbed. "How many" is then wrong and "how
#     often" is right, and the mixed-precision conclusion drawn from
#     count must be withdrawn.
#   <= 78%  -> COUNT SURVIVES. Concentration does not rescue it, the
#     confound is ruled out, and the count framing stands.
#   78-90%  -> BOTH MATTER. Report as partially confounded and weaken the
#     title to a joint claim.
#
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
R=runs_m1
CLI="$PY -m acc.cli --root $R --preset m1"
stamp(){ echo "[$(date '+%F %T')] $*"; }
free_gb(){ df -g . | awk 'NR==2{print $4}'; }

N=clumpulp
if [ ! -f "$R/saes/${N}_s0/sae.pt" ]; then
  if [ ! -f "$R/cache/${N}-train/manifest.json" ]; then
    [ "$(free_gb)" -lt 90 ] && { stamp "ABORT: $(free_gb) GB free"; exit 1; }
    stamp "stage: transcode ${N} (exactly 15,360 ulps confined to one shard)"
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
