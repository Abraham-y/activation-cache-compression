#!/bin/bash
# DOES THE TITLE SURVIVE THE PAPER'S OWN BUDGET CONTROL?
#
# The title rests on a 2.7 pp gap (seed 0: bf16 forward 51.2% against
# rot-int8 storage 53.8%) measured ONLY at the 1x budget -- the budget
# this paper itself says is the one where seed noise is largest and the
# floor is "largely an under-training artifact". sec:payoff shows codec
# gaps compress about 4x from 1x to 10x (rot-int4 1.63 -> 1.18 -> 1.09x
# floor). So the paper's strongest self-control has never been applied to
# the claim on its cover.
#
# Everything else at 3x already exists: the fp32 floor (three seeds,
# 33.3/33.3/33.4%) and rot-int8_b3x_s0, the matched comparator. Only the
# treated arm is missing, so this is one training run at 24,000 steps.
#
# Single seed, matching rot-int8's single seed at this tier. That cannot
# establish that a gap SURVIVES -- it can only show whether one is still
# visible above the resolution the paper already quotes.
#
# PRE-REGISTERED READING, fixed before this ran. At 1x the seed-0 gap is
# +2.7 pp (rot-int8 minus bf16 forward). The paper's own tie band is
# +-1.1 pp.
#   gap >= +1.1 pp   -> SURVIVES. The title holds at a second budget and
#     the sharpest available objection is pre-empted. Report both tiers.
#   gap <= -1.1 pp   -> REVERSES. The title is false at 3x and must go.
#   -1.1 < gap < 1.1 -> DISSOLVES into the tie band. The effect is
#     budget-dependent and cannot be stated as a general fact about
#     pipelines; the frozen-proxy result becomes the main finding. Report the collapse explicitly; it is the same
#     lesson sec:payoff already teaches about rot-int4.
# In every branch the 3x number goes in the paper. This is run to be
# reported, not to be believed only if it is convenient.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
R=runs_m1
CLI="$PY -m acc.cli --root $R --preset m1"
stamp(){ echo "[$(date '+%F %T')] $*"; }

N=fwdbf16
TAG=b3x
STEPS=24000

if [ ! -f "$R/cache/${N}-train/manifest.json" ]; then
  stamp "ABORT: ${N}-train cache is required and was reclaimed"; exit 1
fi
if [ ! -f "$R/saes/${N}_${TAG}_s0/sae.pt" ]; then
  stamp "stage: train ${N} seed=0 steps=${STEPS} (${TAG}) ~3h"
  $CLI train --codec "$N" --seed 0 --steps "$STEPS" --tag "$TAG"
fi
if [ ! -f "$R/eval/pair_fp32_${TAG}_s0__${N}_${TAG}_s0.json" ]; then
  stamp "stage: eval-pair fp32_${TAG}_s0 vs ${N}_${TAG}_s0"
  $CLI eval-pair --ref "fp32_${TAG}_s0" --cand "${N}_${TAG}_s0"
fi
stamp "=== forward-precision budget control complete ==="
