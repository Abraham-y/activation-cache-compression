#!/bin/bash
# Timing on the M5 laptop (the compute-bound regime): the seven loader
# configurations of scripts/modal_timing.py, 2,000 steps each, seed 0,
# from the local SSD. Every configuration feeds bit-identical batches, so
# throughput differences are pure loader cost.
#
# PRE-REGISTERED READING, fixed 7 Oct 2026 before any run:
#  T1  every configuration's end-to-end tokens/s within +-10% of fp32/cpu
#      -> training is compute-bound here and the cache format does not change
#      wall-clock on this hardware. Otherwise report which paths are slower.
#  V   within a codec, every decode path gives bit-identical weights.
# STATUS: stopped after the first configuration (fp32/cpu) because the
# laptop was needed for other work. T1 was never evaluated
# (timing_summary.json: m5.complete = false); the paper's laptop timing
# comes from the 3x runs instead (Appendix H says so).
# Restartable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"
acc() { "$PY" -m acc.cli --root runs_m1 --preset m1 "$@"; }
stamp() { echo "[$(date '+%F %T')] $*"; }
for cfg in fp32:cpu fp32:gpu fp16:cpu fp16:gpu std-int4:cpu std-int4:cpu-nomatmul std-int4:gpu; do
  codec=${cfg%%:*}; decode=${cfg#*:}
  tag="tm5${decode//-/}"
  if [ -f "runs_m1/saes/${codec}_${tag}_s0/sae.pt" ]; then stamp "skip ${codec}_${tag}"; continue; fi
  stamp "train $codec decode=$decode"
  acc train --codec "$codec" --seed 0 --steps 2000 --tag "$tag" --decode "$decode"
done
stamp "M5 timing complete"
