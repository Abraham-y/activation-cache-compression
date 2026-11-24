#!/bin/bash
# FOLLOW-UP CONTROLS, 31 Aug: a stricter floor and repeated draws of the
# two headline perturbations.
#
# PHASE 1 — INIT-ONLY FLOOR. Our floor reseeds init AND
# data order together, which sits low and makes ratios generous. Train
# two fp32 arms with the data order held at seed 0 and only the
# parameter init reseeded (--init-seed, new), giving three same-data
# different-init pairs: (s0, i1), (s0, i2), (i1, i2).
#
# PRE-REGISTERED READING (fixed before running -- honor it):
#   init-only floor mean:
#     > 38.0 (std-int4's agreement)  -> std-int4 does NOT clear the
#       init-only floor; the positive repair verdict is conditional on
#       the wide-nuisance floor and the paper must say so.
#     23.0-38.0 -> repair verdict survives the stricter floor; report
#       both floors and quote the ratio against each.
#     ~= 23.0 (inside +-1.1) -> data order contributes ~nothing at this
#       budget; the "generous" caveat can be retired.
#
# PHASE 2 — LADDER REDRAWS.
# Two independent draws of the two headline rungs (noise4 = the dense
# rung at fp16's distortion; sparseulp6 = the one-ulp-on-a-millionth
# rung), giving n=3 draws per rung with the originals. Report mean and
# half min-max range; the abstract's 70.5 and the 5.4pp flatness claim
# inherit that range.
#
# PRE-REGISTERED READING:
#   half-range <= 1.5pp per rung -> quote rung means with the range;
#     bracket language stays but "one draw" caveat retires.
#   1.5-4pp -> keep bracket language, quote the range prominently, and
#     soften any sentence that leans on a single-draw level.
#   > 4pp -> the dose ladder's levels are draw-dominated; rewrite
#     sec:dose around the count bracket only (70.5 becomes "70 +- draw
#     noise") and say so in Limitations.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
R=runs_m1
CLI="$PY -m acc.cli --root $R --preset m1"
stamp(){ echo "[$(date '+%F %T')] $*"; }
free_gb(){ df -g . | awk 'NR==2{print $4}'; }

stamp "preflight"
[ -f "$R/cache/master-train/manifest.json" ] || { stamp "ABORT: no master-train"; exit 1; }
[ -f "$R/saes/fp32_s0/sae.pt" ] || { stamp "ABORT: no fp32_s0 reference"; exit 1; }
$PY - <<'EOF'
import sys; sys.path.insert(0, "src")
import torch
from acc.sae import TopKSAE
sd = lambda m: torch.cat([p.flatten() for p in m.state_dict().values()])
a, b, c = (TopKSAE(8, 16, 2, seed=s) for s in (0, 1, 0))
assert not torch.equal(sd(a), sd(b)), "init seeds 0/1 identical?!"
assert torch.equal(sd(a), sd(c)), "init seed 0 not reproducible?!"
print("preflight: init-seed decoupling OK")
EOF

stamp "=== PHASE 1: init-only floor (2 runs + 3 pairs) ==="
for I in 1 2; do
  N="fp32_i${I}_s0"
  if [ ! -f "$R/saes/$N/sae.pt" ]; then
    stamp "stage: train $N (data order seed 0, init seed $I, ~55min)"
    $CLI train --codec fp32 --seed 0 --init-seed "$I" --tag "i${I}"
  fi
done
for PAIR in "fp32_s0 fp32_i1_s0" "fp32_s0 fp32_i2_s0" "fp32_i1_s0 fp32_i2_s0"; do
  set -- $PAIR
  if [ ! -f "$R/eval/pair_${1}__${2}.json" ]; then
    stamp "stage: eval-pair $1 vs $2"
    $CLI eval-pair --ref "$1" --cand "$2"
  fi
done
stamp "init-only floor pairs:"
for F in $R/eval/pair_fp32_s0__fp32_i1_s0.json $R/eval/pair_fp32_s0__fp32_i2_s0.json $R/eval/pair_fp32_i1_s0__fp32_i2_s0.json; do
  $PY -c "import json;d=json.load(open('$F'));print('  $F'.split('/')[-1], round(100*d['survived'],2))"
done

stamp "=== PHASE 2: ladder redraws (4 arms, store reclaimed after each) ==="
for N in noise4d1 noise4d2 sparseulp6d1 sparseulp6d2; do
  if [ ! -f "$R/eval/pair_fp32_s0__${N}_s0.json" ]; then
    if [ ! -f "$R/cache/${N}-train/manifest.json" ]; then
      [ "$(free_gb)" -lt 90 ] && { stamp "ABORT: $(free_gb) GB free, need ~90"; exit 1; }
      stamp "stage: transcode ${N}"
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
    if [ ! -f "$R/saes/${N}_s0/sae.pt" ]; then
      stamp "stage: train SAE on ${N} (~55min)"
      $CLI train --codec "$N" --seed 0
    fi
    stamp "stage: eval-pair fp32_s0 vs ${N}_s0"
    $CLI eval-pair --ref fp32_s0 --cand "${N}_s0"
  fi
  if [ -d "$R/cache/${N}-train" ] && [ -f "$R/eval/pair_fp32_s0__${N}_s0.json" ]; then
    stamp "reclaim: rm ${N} stores ($(du -sh $R/cache/${N}-train | cut -f1))"
    rm -rf "$R/cache/${N}-train" "$R/cache/${N}-eval"
  fi
done
stamp "redraw results (with originals):"
for N in noise4 noise4d1 noise4d2 sparseulp6 sparseulp6d1 sparseulp6d2; do
  F="$R/eval/pair_fp32_s0__${N}_s0.json"
  [ -f "$F" ] && $PY -c "import json;d=json.load(open('$F'));print('  ${N}:', round(100*d['survived'],2))"
done
stamp "=== run_redraws complete ==="
