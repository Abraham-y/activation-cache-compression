#!/bin/bash
# Post-experiment disk cleanup. Tiered: each tier is safe once the
# experiments above it are complete. NOTHING here deletes a result —
# only regenerable activation caches. SAE checkpoints, eval JSONs/NPZs,
# figures, logs and the paper are never touched.
#
# Usage:  bash scripts/cleanup.sh           # dry run, shows what/how much
#         bash scripts/cleanup.sh --apply   # actually delete
#
# Regeneration cost if you ever need a cache back:
#   runs_m4_L*/cache : ~2.5h caching + ~15min transcodes per layer
#   runs_m1/cache codecs : ~1h transcodes (needs master-train, tier 3)
#   runs_m1/cache/master-train : ~1h GPT-2 forward pass
set -uo pipefail
cd "$(dirname "$0")/.."
APPLY=${1:-}

guard_running() {
  if pgrep -f "run_.*\.sh|acc\.cli" > /dev/null 2>&1; then
    echo "REFUSING: experiments still running (run_*.sh or acc.cli alive)."
    echo "Wait for them to finish, then re-run."
    exit 1
  fi
}

size() { du -sh "$1" 2>/dev/null | awk '{print $1}'; }

drop() {  # path reason
  [ -e "$1" ] || return 0
  echo "  $(printf '%6s' "$(size "$1")")  $1   ($2)"
  [ "$APPLY" = "--apply" ] && rm -rf "$1"
}

guard_running

# snapshot every store manifest before anything is deleted: analysis
# scripts read bytes/token and d from these, and a reclaimed cache must
# not break figure or table regeneration
snap="cache_manifests_snapshot"
if [ "$APPLY" = "--apply" ]; then
  mkdir -p "$snap"
  for m in runs_m*/cache/*/manifest.json; do
    [ -e "$m" ] || continue
    key=$(echo "$m" | tr '/' '_')
    cp "$m" "$snap/$key" 2>/dev/null || true
  done
  echo "snapshotted $(ls "$snap" 2>/dev/null | wc -l | tr -d ' ') manifests to $snap/"
else
  echo "(manifests would be snapshotted to $snap/ before deleting)"
fi
echo
echo "=== Tier 1: superseded / retracted (always safe) ==="
drop runs "old smoke-test run, superseded by runs_m1"
drop runs_m1/l1_calib "L1/JumpReLU calibration probe SAEs (results in eval/l1_calibration*.json)"
drop runs_m1/archive_jr_dense/cache "retracted dense pilot's caches (evals kept for the record)"

echo
echo "=== Tier 2: Pythia caches (safe: M4 complete, all evals banked) ==="
drop runs_m4_L8/cache "Pythia L8 activation caches"
drop runs_m4_L16/cache "Pythia L16 activation caches"

echo
echo "=== Tier 3: GPT-2 codec caches (safe, but these are what make"
echo "    follow-up experiments cheap — keep unless you need the space) ==="
for c in bf16 int8 int8-o8 int6 int6-o8 int4 int4-o16 pca256 rp256 \
         rot-int6 rot-int4 fp16; do
  drop "runs_m1/cache/${c}-train" "codec cache (transcode from master, ~5min)"
done

echo
echo "=== Tier 4: last resort (breaks all cheap follow-ups) ==="
echo "  NOT listed by default. If you truly need ~80GB more:"
echo "    rm -rf runs_m1/cache/master-train runs_m1/cache/rot-int8-train"
echo "  (master-train costs a ~1h GPT-2 forward to rebuild; rot-int8 is"
echo "   the paper's headline codec and the M6 sweep substrate.)"

echo
if [ "$APPLY" = "--apply" ]; then
  echo "APPLIED. Free space now:"
else
  echo "DRY RUN — nothing deleted. Re-run with --apply to delete tiers 1-3."
  echo "Current free space:"
fi
df -h /System/Volumes/Data | tail -1 | awk '{print "  " $4 " free of " $2}'
