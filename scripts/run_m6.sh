#!/bin/bash
# M6 protocol demo: the sparsity-sweep x multi-seed grid the paper argues
# compression makes affordable. k in {16,32,64} x seeds {0,1,2}, all trained
# from the single rot-int8 cache (15.4 GB vs 61.4 GB fp32). Restartable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"

ROOT=${ROOT:-runs_m1}
PRESET=${PRESET:-m1}
CODEC=rot-int8
acc() { "$PY" -m acc.cli --root "$ROOT" --preset "$PRESET" "$@"; }
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "=== M6 grid: $CODEC, k x seed ==="

# rot-int8_k32_s0 would be bit-identical to the existing rot-int8_s0
# (same cfg/seed/data, training is deterministic — verified by the
# fp32_s0_r1 ceiling), so copy instead of retraining 55 min.
if [ ! -f "$ROOT/saes/${CODEC}_k32_s0/sae.pt" ]; then
  if [ -f "$ROOT/saes/${CODEC}_s0/sae.pt" ]; then
    stamp "copy: ${CODEC}_s0 -> ${CODEC}_k32_s0 (identical config)"
    rm -rf "$ROOT/saes/${CODEC}_k32_s0"   # partial dir would make cp -R nest
    cp -R "$ROOT/saes/${CODEC}_s0" "$ROOT/saes/${CODEC}_k32_s0"
  fi
fi

for k in 16 32 64; do
  for s in 0 1 2; do
    dir="$ROOT/saes/${CODEC}_k${k}_s${s}"
    if [ ! -f "$dir/sae.pt" ]; then
      stamp "stage: train k=$k seed=$s"
      acc train --codec "$CODEC" --k "$k" --tag "k$k" --seed "$s"
    else stamp "skip: train k=$k seed=$s"; fi
  done
done

for k in 16 32 64; do
  for pair in "0 1" "0 2" "1 2"; do
    set -- $pair; a="${CODEC}_k${k}_s$1"; b="${CODEC}_k${k}_s$2"
    if [ ! -f "$ROOT/eval/pair_${a}__${b}.json" ]; then
      stamp "stage: eval-pair $a vs $b"; acc eval-pair --ref "$a" --cand "$b"
    else stamp "skip: eval-pair $a $b"; fi
    if [ ! -f "$ROOT/eval/hungarian_${a}__${b}.json" ]; then
      acc eval-hungarian --ref "$a" --cand "$b"
    fi
  done
done

stamp "=== M6 grid complete ==="
