#!/bin/bash
# Milestone 2: linear-probe fidelity across codecs. Restartable via guards.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"

ROOT=${ROOT:-runs_m1}
PRESET=${PRESET:-m1}
TASKS="sst2 ag_news"
CODECS="fp16 int8 rot-int8 rot-int6 rot-int4 int4 int4-o16 pca256"
acc() { "$PY" -m acc.cli --root "$ROOT" --preset "$PRESET" "$@"; }
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "=== M2 probes into $ROOT ==="
for t in $TASKS; do
  if [ ! -f "$ROOT/probes/$t/master-val/manifest.json" ]; then
    stamp "stage: probe-cache $t"; acc probe-cache --task "$t"
  else stamp "skip: probe-cache $t"; fi
  for c in $CODECS; do
    if [ ! -f "$ROOT/probes/$t/$c-val/manifest.json" ]; then
      stamp "stage: probe-transcode $t $c"; acc probe-transcode --task "$t" --codec "$c"
    else stamp "skip: probe-transcode $t $c"; fi
    if [ ! -f "$ROOT/eval/probe_${t}__${c}.json" ]; then
      stamp "stage: probe-run $t $c"; acc probe-run --task "$t" --codec "$c"
    else stamp "skip: probe-run $t $c"; fi
  done
done
stamp "=== M2 complete ==="
