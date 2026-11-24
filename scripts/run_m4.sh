#!/bin/bash
# M4 scale check: Pythia-1.4B, layers 8 and 16, 10M train tokens each,
# bf16 forward, TopK SAEs (d_sae 16384, k 32, 6000 steps).
# Question: does the M1 frame (floor, graded curve, rotation dominance,
# stable-feature concentration, int4-below-floor) hold at 11x scale on a
# different model family, dataset, and two sites?
#
# Disk-aware sequencing: each layer's fp32 master-train (~82 GB) is
# DELETED once its fp32 reference SAEs are trained (codecs train from
# their own stores; evals use master-eval only). Peak ~280 GB.
# Restartable; a deleted master-train is only needed again if fp32 SAEs
# must be retrained — re-run `acc cache --layer N` in that root first.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"

PRESET=m4-pythia
LAYERS="8 16"
CODECS="fp16 rot-int8 rot-int4 int4"
stamp() { echo "[$(date '+%F %T')] $*"; }

for L in $LAYERS; do
  ROOT="runs_m4_L${L}"
  acc() { "$PY" -m acc.cli --root "$ROOT" --preset "$PRESET" "$@"; }
  stamp "=== M4 layer $L into $ROOT ==="

  if [ ! -f "$ROOT/tokens/meta.json" ]; then
    if [ -f "runs_m4_L8/tokens/meta.json" ] && [ "$ROOT" != "runs_m4_L8" ]; then
      stamp "stage: tokens (copy from runs_m4_L8 — deterministic build)"
      mkdir -p "$ROOT"; cp -R runs_m4_L8/tokens "$ROOT/tokens"
    else
      stamp "stage: tokens"; acc tokens
    fi
  else stamp "skip: tokens"; fi

  if [ ! -f "$ROOT/cache/master-eval/manifest.json" ]; then
    stamp "stage: cache layer $L"; acc cache --layer "$L"
  else stamp "skip: cache"; fi

  # int8 is transcoded here too, while master-train still exists; its
  # SAE and pair are trained by run_controls_a.sh (the "M4 int8 points")
  for c in $CODECS int8; do
    if [ ! -f "$ROOT/cache/$c-eval/manifest.json" ]; then
      stamp "stage: transcode $c"; acc transcode --codec "$c"
    else stamp "skip: transcode $c"; fi
  done

  stamp "stage: null + canary"
  acc null --codec fp16 --canary

  for s in 0 1 2; do
    if [ ! -f "$ROOT/saes/fp32_s${s}/sae.pt" ]; then
      stamp "stage: train fp32 seed=$s"; acc train --codec fp32 --seed "$s"
    else stamp "skip: train fp32_s${s}"; fi
  done

  # fail fast on the first layer's first SAE regime
  "$PY" - <<EOF
import json
logs = [json.loads(l) for l in open("$ROOT/saes/fp32_s0/train_log.jsonl")]
final = logs[-1]
assert final["fvu"] < 0.5, f"fp32_s0 unhealthy: fvu={final['fvu']:.3f}"
print(f"fp32_s0 healthy: l0={final['l0']:.1f} fvu={final['fvu']:.3f}")
EOF

  if [ -d "$ROOT/cache/master-train" ]; then
    stamp "disk: deleting $ROOT/cache/master-train (fp32 refs trained)"
    rm -rf "$ROOT/cache/master-train"
  fi

  for c in $CODECS; do
    if [ ! -f "$ROOT/saes/${c}_s0/sae.pt" ]; then
      stamp "stage: train $c"; acc train --codec "$c" --seed 0
    else stamp "skip: train ${c}_s0"; fi
  done

  pair() {
    if [ ! -f "$ROOT/eval/pair_$1__$2.json" ]; then
      stamp "stage: eval-pair $1 vs $2"; acc eval-pair --ref "$1" --cand "$2"
    else stamp "skip: pair $1 $2"; fi
    if [ ! -f "$ROOT/eval/hungarian_$1__$2.json" ]; then
      acc eval-hungarian --ref "$1" --cand "$2"
    fi
  }
  pair fp32_s0 fp32_s1
  pair fp32_s0 fp32_s2
  pair fp32_s1 fp32_s2
  for c in $CODECS; do pair fp32_s0 "${c}_s0"; done

  stamp "stage: stability layer $L"
  acc stability --ref fp32_s0 --seeds fp32_s1 fp32_s2

  stamp "=== M4 layer $L complete ==="
done
stamp "=== M4 COMPLETE ==="
