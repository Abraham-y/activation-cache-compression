#!/bin/bash
# Post-M1 additions: rotation codecs (QuaRot-style incoherence), Hungarian
# weight-space alignment for every pair, and the stability-stratified
# analysis. Run AFTER run_m1.sh completes; same guards, same restartability.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"

ROOT=${ROOT:-runs_m1}
PRESET=${PRESET:-m1}
ROT_CODECS="rot-int8 rot-int6 rot-int4"
acc() { "$PY" -m acc.cli --root "$ROOT" --preset "$PRESET" "$@"; }
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "=== M1 follow-up into $ROOT ==="

for c in $ROT_CODECS; do
  if [ ! -f "$ROOT/cache/$c-eval/manifest.json" ]; then
    stamp "stage: transcode $c"; acc transcode --codec "$c"
  else stamp "skip: transcode $c"; fi
done

for c in $ROT_CODECS; do
  if [ ! -f "$ROOT/saes/${c}_s0/sae.pt" ]; then
    stamp "stage: train $c seed=0"; acc train --codec "$c" --seed 0
  else stamp "skip: train ${c}_s0"; fi
done

for c in $ROT_CODECS; do
  if [ ! -f "$ROOT/eval/pair_fp32_s0__${c}_s0.json" ]; then
    stamp "stage: eval-pair fp32_s0 vs ${c}_s0"
    acc eval-pair --ref fp32_s0 --cand "${c}_s0"
  else stamp "skip: eval-pair ${c}_s0"; fi
  if [ ! -f "$ROOT/eval/frozen_${c}__fp32_s0.json" ]; then
    stamp "stage: eval-frozen $c"; acc eval-frozen --codec "$c"
  else stamp "skip: eval-frozen $c"; fi
done

stamp "stage: hungarian alignment for all trained pairs"
for d in "$ROOT"/saes/*/; do
  cand=$(basename "$d")
  [ "$cand" = "fp32_s0" ] && continue
  if [ ! -f "$ROOT/eval/hungarian_fp32_s0__${cand}.json" ]; then
    acc eval-hungarian --ref fp32_s0 --cand "$cand"
  fi
done

stamp "stage: stability stratification"
acc stability --ref fp32_s0 --seeds fp32_s1 fp32_s2

stamp "stage: plot"; acc plot
stamp "=== follow-up complete ==="
