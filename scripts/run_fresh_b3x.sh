#!/bin/bash
# Camera-ready budget experiment: fresh tokens vs replay, and matched bytes,
# at the 3x budget (24,000 steps x 4,096 = 98.3M token-visits), GPT-2 L6,
# TopK k=32, every hyperparameter as in the paper. Camera-ready addition.
# Restartable: every stage skips if its output exists.
#
# STORES (scripts/fresh_store.py; the first 77 shards of every fresh store
# ARE the existing 20M-token cache, referenced in place):
#   replay  cache/{master,fp16,std-int4}-train   20.0M tokens, ~4.9 passes
#   fresh   cache/{master,std-int4}-fresh        98.5M tokens, one pass
#   mb      cache/std-int4-mb                    first 302 fresh shards:
#           78.98M tokens in 30.64 GB, vs fp16-train's 30.72 GB (matched bytes)
#
# PRE-REGISTERED READINGS, fixed 29 Sept 2026 before any run below started.
# Agreement = the paper's retraining count (alive ref features whose best
# match reaches r > 0.9 on the first 500k eval tokens). Floors are
# three-pair means. "Seed range" = min-max over the three runs.
#
#  R1  Is the 3x floor a fixed-corpus artifact? Fresh floor F_fr vs replay
#      floor F_rep (33.35, exists). |F_fr - F_rep| <= 2 pp: replay does not
#      inflate the 3x floor, the budget effect is optimization. F_fr <
#      F_rep - 2: part of the rise was convergence on one corpus; the
#      paper's budget paragraph must say so. F_fr > F_rep + 2: fresh data
#      makes seeds agree more.
#  R2  Does the codec verdict survive fresh data at 3x? Same-seed pairs
#      fp32_freshb3x_sK vs std-int4_freshb3x_sK, K = 0,1,2 (same data, same
#      order: the paper's design). Mean above F_fr by > 2 pp and every seed
#      above the floor's max pair: the verdict holds. Mean within 2 pp:
#      std-int4 is indistinguishable from reseeding at 3x with fresh data,
#      and "viable" must be scoped to 1x wherever the paper states it.
#      Below: fails at 3x. The replay counterpart (fp32_b3x vs
#      std-int4_b3x, same seeds) is reported beside it.
#  R3  Matched bytes: fp16 on 20M replayed vs std-int4 on 79M,
#      same stored bytes. Same-seed agreement with the fresh reference is
#      CONFOUNDED here (std-int4-mb shares the reference's data order for
#      ~19k steps, fp16 replay for ~4.9k), so the primary metric is
#      CROSS-SEED: for each candidate, mean agreement over the six pairs
#      (fp32_freshb3x_sA, cand_sB), A != B, where neither init nor order is
#      shared. The ideal is F_fr itself (fresh fp32 against fresh fp32
#      across seeds). X4 - X16 > +1 pp with disjoint six-pair ranges:
#      4-bit with 3.95x the text is closer to the fresh fp32 SAE, so at a
#      fixed storage budget store more text in 4 bits. X16 - X4 > 1 pp,
#      disjoint: precision wins, store fp16. Otherwise a tie on agreement;
#      held-out FVU then decides only if the candidates differ by >= 1%
#      relative (secondary, declared now).
#  P   Pipeline checks, must pass before anything is read:
#      P1 forward re-cache of master-train shard 0 is bit-identical (done:
#         eval/fresh_forward_check.json); P2 the tail's std-int4 codec state
#         equals std-int4-train's; P3 fp32_freshb3x_s0's training log equals
#         fp32_b3x_s0's at every logged step up to 4,800 (shared shards,
#         shared shuffle), and std-int4_mbb3x_s0's equals
#         std-int4_freshb3x_s0's up to step 19,100.
#
# DISK: the tail needs ~253 GiB (fp32 224.6 + std-int4 28.4). The cache
# stage refuses to start below 260 GiB free.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"
acc() { "$PY" -m acc.cli --root runs_m1 --preset m1 "$@"; }
stamp() { echo "[$(date '+%F %T')] $*"; }
STEPS=24000

train() {  # codec seed tag split
  dir="runs_m1/saes/$1_$3_s$2"
  if [ ! -f "$dir/sae.pt" ]; then
    stamp "stage: train $1 seed=$2 tag=$3 split=$4"
    acc train --codec "$1" --seed "$2" --steps "$STEPS" --tag "$3" --split "$4"
  else stamp "skip: $1_$3_s$2"; fi
}
pair() {
  if [ ! -f "runs_m1/eval/pair_$1__$2.json" ]; then
    stamp "stage: eval-pair $1 vs $2"; acc eval-pair --ref "$1" --cand "$2"
  else stamp "skip: pair $1 $2"; fi
}

stamp "=== A. replay arms (existing 20M caches) ==="
for s in 1 2; do train fp16 "$s" b3x train; done
for s in 0 1 2; do train std-int4 "$s" b3x train; done

stamp "=== B. fresh stores ==="
[ -f runs_m1/eval/fresh_forward_check.json ] || \
  "$PY" -W ignore scripts/fresh_store.py check-forward
if [ ! -f runs_m1/cache/master-freshtail-train/manifest.json ]; then
  free_gib=$(df -g runs_m1 | awk 'NR==2 {print $4}')
  if [ "$free_gib" -lt 260 ]; then
    stamp "STOP: ${free_gib} GiB free, the fresh tail needs 260. Free disk, then rerun."
    exit 2
  fi
  stamp "stage: cache fresh tail (78.5M tokens)"
  "$PY" -W ignore scripts/fresh_store.py cache-tail
fi
[ -f runs_m1/cache/std-int4-freshtail-train/manifest.json ] || \
  { stamp "stage: transcode tail"; "$PY" -W ignore scripts/fresh_store.py transcode-tail; }
[ -f runs_m1/cache/master-fresh/manifest.json ] || \
  { stamp "stage: merge manifests"; "$PY" -W ignore scripts/fresh_store.py merge; }

stamp "=== C. fresh and matched-bytes arms ==="
train fp32 0 freshb3x fresh
"$PY" - <<'EOF'
import json
a = {r["step"]: r["mse"] for r in map(json.loads, open("runs_m1/saes/fp32_b3x_s0/train_log.jsonl"))}
b = {r["step"]: r["mse"] for r in map(json.loads, open("runs_m1/saes/fp32_freshb3x_s0/train_log.jsonl"))}
shared = [s for s in sorted(a) if s <= 4800 and s in b]
bad = [s for s in shared if a[s] != b[s]]
print(f"P3 fresh vs replay, {len(shared)} logged steps <= 4800: {'PASS' if shared and not bad else 'FAIL ' + str(bad[:5])}")
assert shared and not bad
EOF
for s in 1 2; do train fp32 "$s" freshb3x fresh; done
for s in 0 1 2; do train std-int4 "$s" freshb3x fresh; done
for s in 0 1 2; do train std-int4 "$s" mbb3x mb; done
"$PY" - <<'EOF'
import json
a = {r["step"]: r["mse"] for r in map(json.loads, open("runs_m1/saes/std-int4_freshb3x_s0/train_log.jsonl"))}
b = {r["step"]: r["mse"] for r in map(json.loads, open("runs_m1/saes/std-int4_mbb3x_s0/train_log.jsonl"))}
shared = [s for s in sorted(a) if s <= 19100 and s in b]
bad = [s for s in shared if a[s] != b[s]]
print(f"P3 mb vs fresh, {len(shared)} logged steps <= 19100: {'PASS' if shared and not bad else 'FAIL ' + str(bad[:5])}")
assert shared and not bad
EOF

stamp "=== D. evaluation ==="
# R1: fresh floor
pair fp32_freshb3x_s0 fp32_freshb3x_s1
pair fp32_freshb3x_s0 fp32_freshb3x_s2
pair fp32_freshb3x_s1 fp32_freshb3x_s2
# R2: same-seed codec pairs, fresh and replay
for s in 0 1 2; do pair "fp32_freshb3x_s$s" "std-int4_freshb3x_s$s"; done
for s in 0 1 2; do pair "fp32_b3x_s$s" "std-int4_b3x_s$s"; done
for s in 1 2; do pair "fp32_b3x_s$s" "fp16_b3x_s$s"; done
# R3: cross-seed against the fresh reference, both matched-bytes candidates
for a in 0 1 2; do for b in 0 1 2; do
  [ "$a" = "$b" ] && continue
  pair "fp32_freshb3x_s$a" "fp16_b3x_s$b"
  pair "fp32_freshb3x_s$a" "std-int4_mbb3x_s$b"
done; done
# descriptive: how far replay moves an fp32 SAE from its fresh twin
for s in 0 1 2; do pair "fp32_freshb3x_s$s" "fp32_b3x_s$s"; done
stamp "stage: stability at 3x fresh"
[ -f runs_m1/eval/stability_fp32_freshb3x_s0_t0p9.json ] || \
  acc stability --ref fp32_freshb3x_s0 --seeds fp32_freshb3x_s1 fp32_freshb3x_s2
stamp "=== fresh/matched-bytes experiment complete ==="
