#!/bin/bash
# std-int4 at seeds 1 and 2 (the recommended codec was validated at one
# seed only). Same-seed pairs against fp32_s1 / fp32_s2 isolate the codec.
# Also entropy + loader-throughput rows for std-int4 while its store exists.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
R=runs_m1; CLI="$PY -m acc.cli --root $R --preset m1"
stamp(){ echo "[$(date '+%F %T')] $*"; }
N=std-int4
[ -f "$R/cache/${N}-train/manifest.json" ] || { stamp "transcode $N"; $CLI transcode --codec $N; }
for S in 1 2; do
  [ -f "$R/saes/${N}_s${S}/sae.pt" ] || { stamp "train ${N}_s${S} (~55min)"; $CLI train --codec $N --seed $S; }
  [ -f "$R/eval/pair_fp32_s${S}__${N}_s${S}.json" ] || { stamp "eval-pair fp32_s${S} vs ${N}_s${S}"; $CLI eval-pair --ref fp32_s${S} --cand ${N}_s${S}; }
done
stamp "std-int4 same-seed agreement:"; for S in 0 1 2; do F=$R/eval/pair_fp32_s${S}__${N}_s${S}.json; [ -f $F ] && $PY -c "import json;print('  seed $S:', round(100*json.load(open('$F'))['survived'],2))"; done
stamp "=== std-int4 seeds complete ==="
