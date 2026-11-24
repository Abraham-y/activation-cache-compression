#!/usr/bin/env bash
# Noise rungs at 1e-3 and 1e-2 -- INSIDE the magnitude range real codecs
# occupy (fp16 2.03e-4 .. naive int4 4.55e-1).
#
# WHY: the paper licenses control-relative claims on a resolution
# asymmetry -- ~2.5 pp/decade near lossless (isotropic noise) against
# ~8.5 pp/decade across the quantizer family. Those two rates are
# currently measured on DISJOINT x-ranges and in DIFFERENT perturbation
# families, and the paper's own text says a per-decade rate is well
# defined only within a family. So the asymmetry may be nothing but
# concavity of a single damage curve.
#
# PRE-REGISTERED READING (fixed before running -- honor it):
#   If isotropic noise at 1e-3/1e-2 falls at 6 pp/decade or steeper, the
#   two families agree where they overlap: the asymmetry is a property of
#   the DISTORTION LEVEL, the licensing argument stands, and it should be
#   restated as "resolution depends on regime, not on family".
#   If it falls at under 4 pp/decade, the quantizer rate is NOT reproduced
#   by matched-magnitude isotropic noise: the asymmetry is confounded with
#   perturbation structure, and the licensing claim must be withdrawn and
#   replaced by the below-floor verdict alone.
#   Between 4 and 6 pp/decade: indeterminate, report as such.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
R=runs_m1
CLI="$PY -m acc.cli --root $R --preset m1"
stamp(){ echo "[$(date '+%F %T')] $*"; }
free_gb(){ df -g . | awk 'NR==2{print $4}'; }

for e in 3 2; do
  N="noise${e}"
  if [ ! -f "$R/saes/${N}_s0/sae.pt" ]; then
    if [ ! -f "$R/cache/${N}-train/manifest.json" ]; then
      if [ "$(free_gb)" -lt 90 ]; then
        stamp "ABORT: $(free_gb) GB free, need ~90 GB for ${N}"; exit 1
      fi
      stamp "stage: transcode ${N} (fp32 + exact rel-L2 1e-${e})"
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
done
stamp "=== noise overlap rungs complete ==="
