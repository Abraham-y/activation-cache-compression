#!/bin/bash
# Budget scaling of the headline quantity.
# Stable-only survival of fp16 / rot-int8 / rot-int4 at 1x/3x/10x
# training budgets on GPT-2, with the seed floor and stable set measured
# AT EACH BUDGET (3 fp32 seeds per tier). 1x exists; this adds:
#   3x  (24000 steps): fp32_s2 + fp16 + rot-int4 (+pairs, stability)
#   10x (80000 steps): fp32 s0/s1/s2 + fp16 + rot-int8 + rot-int4
# Disclosure: budget scales by epoch-recycling the same 20M-token cache
# (16 epochs at 10x); production budgets also scale unique data.
# ~62h total. Restartable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"
acc() { "$PY" -m acc.cli --root runs_m1 --preset m1 "$@"; }
stamp() { echo "[$(date '+%F %T')] $*"; }

train_tag() {  # codec seed steps tag
  dir="runs_m1/saes/$1_$4_s$2"
  if [ ! -f "$dir/sae.pt" ]; then
    stamp "stage: train $1 seed=$2 steps=$3 ($4)"
    acc train --codec "$1" --seed "$2" --steps "$3" --tag "$4"
  else stamp "skip: $1_$4_s$2"; fi
}
pair() {
  if [ ! -f "runs_m1/eval/pair_$1__$2.json" ]; then
    stamp "stage: eval-pair $1 vs $2"; acc eval-pair --ref "$1" --cand "$2"
  else stamp "skip: pair $1 $2"; fi
}

stamp "=== budget scaling: 3x completions ==="
train_tag fp32 2 24000 b3x
train_tag fp16 0 24000 b3x
train_tag rot-int4 0 24000 b3x
pair fp32_b3x_s0 fp32_b3x_s2
pair fp32_b3x_s1 fp32_b3x_s2
pair fp32_b3x_s0 fp16_b3x_s0
pair fp32_b3x_s0 rot-int4_b3x_s0
stamp "stage: stability at 3x"
acc stability --ref fp32_b3x_s0 --seeds fp32_b3x_s1 fp32_b3x_s2

stamp "=== budget scaling: 10x tier ==="
for s in 0 1 2; do train_tag fp32 "$s" 80000 b10x; done
# fail fast on regime before the codec runs
"$PY" - <<'EOF'
import json
logs = [json.loads(l) for l in open("runs_m1/saes/fp32_b10x_s0/train_log.jsonl")]
f = logs[-1]
assert f["fvu"] < 0.5 and f["l0"] == 32, f"b10x unhealthy: {f}"
print(f"fp32_b10x_s0 healthy: fvu={f['fvu']:.3f}")
EOF
train_tag fp16 0 80000 b10x
train_tag rot-int8 0 80000 b10x
train_tag rot-int4 0 80000 b10x
pair fp32_b10x_s0 fp32_b10x_s1
pair fp32_b10x_s0 fp32_b10x_s2
pair fp32_b10x_s1 fp32_b10x_s2
pair fp32_b10x_s0 fp16_b10x_s0
pair fp32_b10x_s0 rot-int8_b10x_s0
pair fp32_b10x_s0 rot-int4_b10x_s0
stamp "stage: stability at 10x"
acc stability --ref fp32_b10x_s0 --seeds fp32_b10x_s1 fp32_b10x_s2
stamp "=== budget scaling complete ==="
