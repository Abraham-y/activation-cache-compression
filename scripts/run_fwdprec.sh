#!/bin/bash
# Forward-precision control: the SECOND treatment through the protocol,
# and the first that is not a storage codec.
#
# Treatment: compute the activation cache with a bf16 model forward
# instead of fp32. Storage is fp32 on both sides, so no quantization is
# involved anywhere; the only difference is the dtype of the forward that
# produced the numbers. This is the choice Gemma Scope actually made, so
# a large fraction of real caches are already on the treated side of it.
#
# What it tests (paper §7): the construct-validity failure is a property
# of the measurement, not of compression. If the frozen judge certifies a
# bf16 forward while retraining diverges, the lesson generalizes past
# codecs to how the cache is produced. If retraining does NOT diverge,
# that is equally informative — the protocol discriminates rather than
# flagging every data-side choice.
#
# Ceiling (1.0000) and floor (23.0%) are already measured for this site,
# so this run only pays for the treated arm.
# All stages guarded/restartable.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
stamp() { echo "[$(date '+%F %T')] $*"; }

R=runs_m1
N=fwdbf16

source "$(dirname "$0")/_env.sh"
CLI="$PY -m acc.cli --root $R --preset m1"

# never overlap another driver on the same GPU
for other in run_m1.sh run_m4.sh run_arch.sh run_m6.sh run_budget_scaling.sh; do
  while pgrep -f "$other" > /dev/null 2>&1; do
    stamp "waiting: $other still active"; sleep 300
  done
done

# --- 1. recompute the cache with a bf16 forward (~1h train + ~3min eval) ---
if [ ! -f "$R/cache/$N-train/manifest.json" ] || \
   [ ! -f "$R/cache/$N-eval/manifest.json" ]; then
  stamp "stage: bf16-forward cache (asserts shard alignment with master)"
  $PY scripts/forward_precision_cache.py --root "$R" --dtype bf16 --name "$N"
fi

# --- 2. null + canary on the NEW store, BEFORE trusting any number from
# it (project rule: every pipeline change reruns the null) ---
if [ ! -f "$R/eval/null_${N}_frozen.json" ]; then
  stamp "stage: null + misalignment canary on the $N store"
  $CLI null --codec "$N" --canary
fi

# --- 3. what the frozen judge says about the treatment ---
if [ ! -f "$R/eval/frozen_${N}__fp32_s0.json" ]; then
  stamp "stage: eval-frozen (the proxy's verdict)"
  $CLI eval-frozen --codec "$N"
fi

# --- 4. the treated arm: retrain at seed 0 on the bf16-forward cache ---
if [ ! -f "$R/saes/${N}_s0/sae.pt" ]; then
  stamp "stage: train SAE on the bf16-forward cache (~55min)"
  $CLI train --codec "$N" --seed 0
fi

# --- 5. retrained reality, against the existing ceiling and floor ---
if [ ! -f "$R/eval/pair_fp32_s0__${N}_s0.json" ]; then
  stamp "stage: eval-pair fp32_s0 vs ${N}_s0"
  $CLI eval-pair --ref fp32_s0 --cand "${N}_s0"
fi
if [ ! -f "$R/eval/hungarian_fp32_s0__${N}_s0.json" ]; then
  stamp "stage: eval-hungarian (independent weight-space definition)"
  $CLI eval-hungarian --ref fp32_s0 --cand "${N}_s0"
fi

# --- 6. re-stratify: adds a ${N}_s0 row, leaves every existing row
# untouched (each cand is computed independently from its own npz) ---
stamp "stage: stability stratification (refresh, adds $N row)"
$CLI stability --ref fp32_s0 --seeds fp32_s1 fp32_s2 --threshold 0.9

stamp "=== forward-precision control complete ==="
$PY - <<'EOF'
import json
from pathlib import Path
ev = Path("runs_m1/eval")
fl = [json.loads((ev / f"pair_fp32_s{a}__fp32_s{b}.json").read_text())["survived"]
      for a, b in ((0, 1), (0, 2), (1, 2))]
floor = sum(fl) / 3
p = json.loads((ev / "pair_fp32_s0__fwdbf16_s0.json").read_text())
f = json.loads((ev / "frozen_fwdbf16__fp32_s0.json").read_text())
st = {r["cand"]: r for r in
      json.loads((ev / "stability_fp32_s0_t0p9.json").read_text())["rows"]}
row = st.get("fwdbf16_s0", {})
print(f"""
FORWARD PRECISION (bf16 forward, fp32 storage, GPT-2 L6)
  frozen judge   mean r = {f['mean_corr']:.6f}   survived = {100*f['survived']:.1f}%
  raw distortion rel-L2 = {f['direct']['rel_l2_mean']:.5f}  cos = {f['direct']['cos_mean']:.6f}
  RETRAINED      survived = {100*p['survived']:.1f}%   mean r = {p['mean_corr']:.4f}
  seed floor     {100*floor:.1f}%    ->  treatment sits at {p['survived']/floor:.2f}x floor
  stable-only    {100*row.get('survived_stable', float('nan')):.1f}%   unstable {100*row.get('survived_unstable', float('nan')):.1f}%
  FVU ref/cand   {p['fvu_ref']:.4f} / {p['fvu_cand']:.4f}
""")
EOF
