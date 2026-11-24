#!/bin/bash
# POST-HOC (not pre-registered) resolution check for R3 of run_fresh_b3x.sh,
# 7 Oct 2026: score naive int4 and fp32, both on the replayed 20M cache, with
# the same cross-seed measure against fresh fp32. Eval only.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
acc() { "$PY" -m acc.cli --root runs_m1 --preset m1 "$@"; }
pair() { [ -f "runs_m1/eval/pair_$1__$2.json" ] || acc eval-pair --ref "$1" --cand "$2"; }
for a in 1 2; do pair "fp32_freshb3x_s$a" int4_b3x_s0; done
for a in 0 1 2; do for b in 0 1 2; do [ "$a" = "$b" ] && continue; pair "fp32_freshb3x_s$a" "fp32_b3x_s$b"; done; done
echo "r3 resolution pairs done $(date)"
