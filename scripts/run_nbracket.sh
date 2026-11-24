#!/bin/bash
# THE n-BRACKET: three sparse rungs inside the open
# interval 1 < n < 15,335 touched values, at the existing codec's deeper
# exponents. Per-shard k = max(1, round(n_shard*d*1e-E)) gives, over the
# 77 train shards, approximately:
#   sparseulp9 -> n ~ 77     sparseulp8 -> n ~ 154    sparseulp7 -> n ~ 1,540
# Exact realized counts are written to eval/sparse_counts_bracket.json.
# Zero new codec code: the d-seed regex family already parses these names.
#
# Anchors: n=1 -> 99.98, n=15,335 -> 70.5 (+-1.0 over three draws).
# A log-linear count response between the anchors predicts about
# 7pp per decade of n: ~86.7% at n~77, ~84.6% at n~154, ~77.6% at n~1,540.
#
# PRE-REGISTERED READING (fixed before running -- honor it):
#   All three within ~3pp of the log-linear interpolation -> GRADED:
#     the response is consistent with a smooth log-count dependence,
#     no cliff; report the points and the implied ~7pp/decade, fit no law.
#   sparseulp9 >= 97% -> LATE CLIFF: the transition sits above n~77;
#     report the tightened bracket.
#   sparseulp9 <= 80% -> EARLY CLIFF: the transition sits below n~77;
#     the bracket tightens 200x (15,335 -> 77) and 1 < n < 77 stays open.
#   Anything else -> INTERMEDIATE: report the points as a bracket,
#     no shape claim.
# All rungs are one draw each; last night's redraw spread (+-1.0pp at
# n=15,335) is the scale against which point differences are read.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
R=runs_m1
CLI="$PY -m acc.cli --root $R --preset m1"
stamp(){ echo "[$(date '+%F %T')] $*"; }
free_gb(){ df -g . | awk 'NR==2{print $4}'; }

stamp "preflight"
[ -f "$R/cache/master-train/manifest.json" ] || { stamp "ABORT: no master-train"; exit 1; }
[ -f "$R/saes/fp32_s0/sae.pt" ] || { stamp "ABORT: no fp32_s0 reference"; exit 1; }

# realized per-corpus counts from the train manifest's shard sizes,
# mirroring SparseUlpCodec's per-shard k
$PY - <<'EOF'
import json, sys
sys.path.insert(0, "src")
man = json.load(open("runs_m1/cache/master-train/manifest.json"))
out = {}
for E in (9, 8, 7):
    frac = 10.0 ** (-E)
    ks = [max(1, int(round(s["n_tokens"] * man["d"] * frac)))
          for s in man["shards"]]
    out[f"sparseulp{E}"] = {"total_touched": sum(ks), "n_shards": len(ks),
                            "per_shard_min": min(ks), "per_shard_max": max(ks)}
    print(f"  sparseulp{E}: n = {sum(ks)}")
json.dump(out, open("runs_m1/eval/sparse_counts_bracket.json", "w"), indent=1)
EOF

for N in sparseulp9 sparseulp8 sparseulp7; do
  if [ ! -f "$R/eval/pair_fp32_s0__${N}_s0.json" ]; then
    if [ ! -f "$R/cache/${N}-train/manifest.json" ]; then
      [ "$(free_gb)" -lt 90 ] && { stamp "ABORT: $(free_gb) GB free, need ~90"; exit 1; }
      stamp "stage: transcode ${N}"
      $CLI transcode --codec "$N"
    fi
    if [ ! -f "$R/eval/null_${N}_frozen.json" ]; then
      stamp "stage: null + canary on ${N}"
      $CLI null --codec "$N" --canary
    fi
    if [ ! -f "$R/eval/frozen_${N}__fp32_s0.json" ]; then
      stamp "stage: eval-frozen ${N}"
      $CLI eval-frozen --codec "$N"
    fi
    if [ ! -f "$R/saes/${N}_s0/sae.pt" ]; then
      stamp "stage: train SAE on ${N} (~55min)"
      $CLI train --codec "$N" --seed 0
    fi
    stamp "stage: eval-pair fp32_s0 vs ${N}_s0"
    $CLI eval-pair --ref fp32_s0 --cand "${N}_s0"
  fi
  if [ -d "$R/cache/${N}-train" ] && [ -f "$R/eval/pair_fp32_s0__${N}_s0.json" ]; then
    stamp "reclaim: rm ${N} stores ($(du -sh $R/cache/${N}-train | cut -f1))"
    rm -rf "$R/cache/${N}-train" "$R/cache/${N}-eval"
  fi
done
stamp "n-bracket results (with anchors):"
for N in oneulp sparseulp9 sparseulp8 sparseulp7 sparseulp6; do
  F="$R/eval/pair_fp32_s0__${N}_s0.json"
  [ -f "$F" ] && $PY -c "import json;d=json.load(open('$F'));print('  ${N}:', round(100*d['survived'],2))"
done
stamp "=== n-bracket complete ==="
