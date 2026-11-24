#!/bin/bash
# Follow-up compute queue (audit-requested), runs after M4 completes:
# 1. int8_s0 SAEs for both M4 layers (the preserved real 1-byte point)
# 2. Error-bar seeds for rot-int6 / int6-o8 (shores the "rotation >=
#    outlier channels" ordering that currently rests on n=1 cells)
# 3. Budget sensitivity: fp32 + rot-int8 at 3x steps from the same caches
#    (the "does the floor move with training budget" question)
# All guarded/restartable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"
stamp() { echo "[$(date '+%F %T')] $*"; }

# never overlap the M4 driver on the same GPU
while pgrep -f "run_m4.sh" > /dev/null 2>&1; do
  stamp "waiting: run_m4.sh still active"; sleep 600
done

# m1 stages FIRST (cheap, self-contained); M4 int8 stages last and
# non-fatal, so an M4-side problem cannot hold the m1 error bars hostage

# --- 2. rot-int6 / int6-o8 error-bar seeds (m1) ---
for c in rot-int6 int6-o8; do
  for s in 1 2; do
    if [ ! -f "runs_m1/saes/${c}_s${s}/sae.pt" ]; then
      stamp "stage: m1 $c seed=$s"
      "$PY" -m acc.cli --root runs_m1 --preset m1 train --codec "$c" --seed "$s"
    fi
    if [ ! -f "runs_m1/eval/pair_fp32_s${s}__${c}_s${s}.json" ]; then
      "$PY" -m acc.cli --root runs_m1 --preset m1 eval-pair --ref "fp32_s${s}" --cand "${c}_s${s}"
    fi
  done
done

# --- 3. Budget sensitivity: 3x steps (24000) on m1 ---
for c in fp32 rot-int8; do
  if [ ! -f "runs_m1/saes/${c}_b3x_s0/sae.pt" ]; then
    stamp "stage: m1 $c 3x budget"
    "$PY" -m acc.cli --root runs_m1 --preset m1 train --codec "$c" --seed 0 \
      --steps 24000 --tag b3x
  fi
done
if [ ! -f "runs_m1/saes/fp32_b3x_s1/sae.pt" ]; then
  stamp "stage: m1 fp32 3x seed 1 (floor-at-budget)"
  "$PY" -m acc.cli --root runs_m1 --preset m1 train --codec fp32 --seed 1 \
    --steps 24000 --tag b3x
fi
for pair in "fp32_b3x_s0 fp32_b3x_s1" "fp32_b3x_s0 rot-int8_b3x_s0"; do
  set -- $pair
  if [ ! -f "runs_m1/eval/pair_$1__$2.json" ]; then
    stamp "stage: eval-pair $1 vs $2"
    "$PY" -m acc.cli --root runs_m1 --preset m1 eval-pair --ref "$1" --cand "$2"
  fi
done

# --- M4 int8 points (non-fatal) ---
for L in 8 16; do
  R="runs_m4_L${L}"
  if [ -f "$R/cache/int8-train/manifest.json" ]; then
    if [ ! -f "$R/saes/int8_s0/sae.pt" ]; then
      stamp "stage: M4 L$L int8_s0"
      "$PY" -m acc.cli --root "$R" --preset m4-pythia train --codec int8 --seed 0 \
        || { stamp "WARN: L$L int8 train failed, continuing"; continue; }
    fi
    if [ ! -f "$R/eval/pair_fp32_s0__int8_s0.json" ]; then
      "$PY" -m acc.cli --root "$R" --preset m4-pythia eval-pair --ref fp32_s0 --cand int8_s0 \
        || stamp "WARN: L$L int8 eval failed, continuing"
    fi
  else stamp "skip: L$L int8 (no store)"; fi
done

# --- regenerate the throughput benchmark AS AN ARTIFACT (audit: the
# split-timer table had no on-disk provenance); prefetch=0 for clean
# disk/decode semantics ---
"$PY" scripts/bench_throughput.py --root runs_m1 || stamp "WARN: bench failed"
stamp "=== controls queue A complete ==="
