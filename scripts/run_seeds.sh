#!/bin/bash
# Error-bar seeds: 2 extra seeds for the key codecs so the paper's codec
# numbers carry intervals. rot-int8 needs no training — the M6 grid's
# k32_s1/s2 ARE rot-int8 at the default config — only its same-seed pairs
# against fp32_s1/s2 are missing. Restartable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"

ROOT=${ROOT:-runs_m1}
PRESET=${PRESET:-m1}
CODECS="fp16 rot-int4 int4"
acc() { "$PY" -m acc.cli --root "$ROOT" --preset "$PRESET" "$@"; }
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "=== error-bar seeds ==="
for s in 1 2; do
  for c in $CODECS; do
    if [ ! -f "$ROOT/saes/${c}_s${s}/sae.pt" ]; then
      stamp "stage: train $c seed=$s"; acc train --codec "$c" --seed "$s"
    else stamp "skip: train ${c}_s${s}"; fi
  done
done

pair() {
  if [ ! -f "$ROOT/eval/pair_$1__$2.json" ]; then
    stamp "stage: eval-pair $1 vs $2"; acc eval-pair --ref "$1" --cand "$2"
  else stamp "skip: eval-pair $1 $2"; fi
}
for s in 1 2; do
  for c in $CODECS; do pair "fp32_s${s}" "${c}_s${s}"; done
  pair "fp32_s${s}" "rot-int8_k32_s${s}"
done
stamp "=== error-bar seeds complete ==="
