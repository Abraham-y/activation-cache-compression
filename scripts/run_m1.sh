#!/bin/bash
# Milestone 1 full run. Restartable: every stage is skipped if its output
# artifact already exists, so kill + rerun resumes where it left off.
# Expected: ~45 min caching, ~1h transcoding, ~11h SAE training (14 SAEs),
# ~4h evals. ~225 GB under $ROOT. Run under caffeinate to prevent sleep.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"

ROOT=${ROOT:-runs_m1}
PRESET=${PRESET:-m1}
CODECS="fp16 bf16 int8 int8-o8 int6 int6-o8 int4 int4-o16 pca256 rp256"
TRAIN_CODECS="$CODECS"
acc() { "$PY" -m acc.cli --root "$ROOT" --preset "$PRESET" "$@"; }
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "=== M1 run into $ROOT ==="

if [ ! -f "$ROOT/tokens/meta.json" ]; then
  stamp "stage: tokens"; acc tokens
else stamp "skip: tokens"; fi

if [ ! -f "$ROOT/cache/master-eval/manifest.json" ]; then
  stamp "stage: cache (fp32 master, both splits)"; acc cache
else stamp "skip: cache"; fi

for c in $CODECS; do
  if [ ! -f "$ROOT/cache/$c-eval/manifest.json" ]; then
    stamp "stage: transcode $c"; acc transcode --codec "$c"
  else stamp "skip: transcode $c"; fi
done

stamp "stage: null (raw channels) + canary"
acc null --codec fp16 --canary

# fp32 reference set: 3 seeds for the noise floor + 1 replicate for the ceiling
for spec in "fp32 0 0" "fp32 1 0" "fp32 2 0" "fp32 0 1"; do
  set -- $spec; c=$1; s=$2; r=$3
  dir="$ROOT/saes/${c}_s${s}"; [ "$r" != "0" ] && dir="${dir}_r${r}"
  if [ ! -f "$dir/sae.pt" ]; then
    stamp "stage: train $c seed=$s rep=$r"
    acc train --codec "$c" --seed "$s" --rep "$r"
  else stamp "skip: train $(basename "$dir")"; fi
done

for c in $TRAIN_CODECS; do
  if [ ! -f "$ROOT/saes/${c}_s0/sae.pt" ]; then
    stamp "stage: train $c seed=0"; acc train --codec "$c" --seed 0
  else stamp "skip: train ${c}_s0"; fi
done

stamp "stage: null through frozen SAE + canary"
acc null --codec fp16 --canary

pair() {
  if [ ! -f "$ROOT/eval/pair_$1__$2.json" ]; then
    stamp "stage: eval-pair $1 vs $2"; acc eval-pair --ref "$1" --cand "$2"
  else stamp "skip: eval-pair $1 vs $2"; fi
}
pair fp32_s0 fp32_s0_r1   # run-noise ceiling
pair fp32_s0 fp32_s1      # seed-noise floor
pair fp32_s0 fp32_s2
pair fp32_s1 fp32_s2
for c in $TRAIN_CODECS; do pair fp32_s0 "${c}_s0"; done

for c in $CODECS; do
  if [ ! -f "$ROOT/eval/frozen_${c}__fp32_s0.json" ]; then
    stamp "stage: eval-frozen $c"; acc eval-frozen --codec "$c"
  else stamp "skip: eval-frozen $c"; fi
done

stamp "stage: plot"; acc plot
stamp "=== M1 run complete ==="
