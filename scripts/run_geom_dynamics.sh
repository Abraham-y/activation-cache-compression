#!/bin/bash
# Training-dynamics driver (Appendix J; pre-registered readings, the
# deviation and the amended plan are in scripts/geom_dynamics.py). Waits for the
# rebuilt fp32 master-train cache, checks it against the snapshot of the
# original manifest, then runs phase 1, the amended late and early
# passes, and the paper's eval-pair
# on every full-length fork. Restartable per stage.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
R=runs_m1
stamp(){ echo "[$(date '+%F %T')] $*"; }

until [ -f "$R/cache/master-train/manifest.json" ]; do sleep 30; done
# serialize with scripts/geom_margins.py: both together exceed RAM
while pgrep -f "scripts/geom_margins.py" > /dev/null; do sleep 60; done
stamp "master-train present; checking against the snapshot manifest"
"$PY" - <<'EOF'
import json, sys
a = json.load(open("runs_m1/cache/master-train/manifest.json"))
b = json.load(open("cache_manifests_snapshot/runs_m1_cache_master-train_manifest.json"))
keys = ("file", "n_tokens", "payload_bytes")
same = (a["n_tokens"] == b["n_tokens"] and len(a["shards"]) == len(b["shards"]) and
        all(all(x[k] == y[k] for k in keys) for x, y in zip(a["shards"], b["shards"])))
print("manifest matches snapshot:", same)
sys.exit(0 if same else 1)
EOF

if [ ! -f "$R/eval/geom_dynamics_phase1.json" ] && ! pgrep -f "geom_dynamics.py phase1" > /dev/null; then
  stamp "phase 1: A + ulp_t0 + doses + null clone (~2.7 h)"
  "$PY" -u scripts/geom_dynamics.py phase1
fi
# phase 2 as first written was cancelled (its one-ulp impulse was absorbed;
# see the DEVIATION in scripts/geom_dynamics.py); the amended passes:
while pgrep -f "geom_dynamics.py phase1" > /dev/null; do sleep 30; done
[ -f "$R/eval/geom_dynamics_phase1.json" ] || { stamp "phase 1 did not finish"; exit 1; }
if [ ! -f "$R/eval/geom_dynamics_late.json" ]; then
  stamp "late pass: dense 1e-7 impulses at t0 >= 4000, absorption scan at 4000"
  "$PY" -u scripts/geom_dynamics.py late
fi
if [ ! -f "$R/eval/geom_dynamics_early.json" ]; then
  stamp "early pass: d7 at t0 = 0, oneulp replay, absorption scan at 0"
  "$PY" -u scripts/geom_dynamics.py early
fi
for d in "$R"/saes/geomdyn_d7_t*_s0 "$R"/saes/geomdyn_oneulp_replay_s0; do
  c=$(basename "$d")
  [ -f "$R/eval/pair_geomdyn_A_s0__${c}.json" ] && continue
  stamp "eval-pair geomdyn_A_s0 vs $c"
  "$PY" -m acc.cli --root "$R" --preset m1 eval-pair --ref geomdyn_A_s0 --cand "$c"
done
stamp "=== training dynamics complete ==="
