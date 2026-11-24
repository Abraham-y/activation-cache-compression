#!/usr/bin/env bash
# Subspace recovery (camera-ready addition). Pre-registered readings
# in scripts/subspace_recovery.py. Restartable: skips finished pairs.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_env.sh
ROOT=runs_m1
run() {
  local ref=$1 cand=$2
  if [ -f "$ROOT/eval/subspace_${ref}__${cand}.json" ]; then
    echo "[skip] $ref vs $cand"; return; fi
  echo "[$(date '+%F %T')] $ref vs $cand"
  PYTHONPATH=src "$PY" -W ignore scripts/subspace_recovery.py \
    --root "$ROOT" --ref "$ref" --cand "$cand"
}
for c in fp32_s0_r1 sparseulp6_s0 sparseulp6d1_s0 sparseulp6d2_s0 fp16_s0 \
         std-int4_s0 int4_s0 fp32_s1 fp32_s2 rot-int4_s0; do
  run fp32_s0 "$c"
done
run fp32_s1 std-int4_s1
run fp32_s2 std-int4_s2
run fp32_s1 fp32_s2
echo "[$(date '+%F %T')] subspace recovery complete"
