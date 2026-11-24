#!/bin/bash
# Second-architecture check: JumpReLU SAEs (Gemma Scope's production
# architecture; stable sparsity equilibria — plain L1 collapsed in an
# earlier attempt) at calibrated lambda across key codecs, with their
# own seed floor. Answers "is the stable-feature result TopK-specific?"
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"

ROOT=${ROOT:-runs_m1}
PRESET=${PRESET:-m1}
ARCH=${ARCH:-jumprelu}
CODECS="fp16 rot-int8 rot-int4 int4"
CALIB="$ROOT/eval/l1_calibration_${ARCH}.json"
acc() { "$PY" -m acc.cli --root "$ROOT" --preset "$PRESET" "$@"; }
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "=== $ARCH architecture check ==="

# guard on VALIDITY, not existence — a failed calibration writes the json
# (valid: false) before exiting nonzero, and must not be silently reused
calib_ok() {
  "$PY" -c "import json,sys; d=json.load(open('$CALIB')); sys.exit(0 if d.get('valid') else 1)" 2>/dev/null
}
if ! calib_ok; then
  stamp "stage: lambda calibration ($ARCH)"
  "$PY" scripts/calibrate_l1.py --root "$ROOT" --preset "$PRESET" --arch "$ARCH"
  calib_ok || { stamp "ABORT: calibration invalid"; exit 1; }
else stamp "skip: calibration (valid)"; fi
COEF=$("$PY" -c "import json; print(json.load(open('$CALIB'))['chosen_coef'])")
stamp "using coef=$COEF"
"$PY" -c "import json; d=json.load(open('$CALIB')); \
  print('NOTE: sparsity target NOT achieved — panel runs at L0~%.0f, disclose in analysis' % d['intended_l0']) \
  if not d.get('target_achieved') else None"

train_arch() {  # codec seed
  dir="$ROOT/saes/$1_${ARCH}_s$2"
  if [ ! -f "$dir/sae.pt" ]; then
    stamp "stage: train $ARCH $1 seed=$2"
    acc train --codec "$1" --seed "$2" --arch "$ARCH" --l1 "$COEF"
  else stamp "skip: $ARCH $1 s$2"; fi
}
train_arch fp32 0

# fail fast: the first SAE must be alive before we burn 5 more hours
# (fixed-coef L1 collapsed to L0~0/FVU~1 exactly here in an earlier attempt)
"$PY" - <<EOF
import json
logs = [json.loads(l) for l in open("$ROOT/saes/fp32_${ARCH}_s0/train_log.jsonl")]
final = logs[-1]
assert 4 <= final["l0"] <= 320 and final["fvu"] < 0.9, \
    f"fp32_${ARCH}_s0 out of regime: l0={final['l0']:.1f} fvu={final['fvu']:.3f}"
print(f"fp32_${ARCH}_s0 healthy: l0={final['l0']:.1f} fvu={final['fvu']:.3f}")
EOF

train_arch fp32 1
for c in $CODECS; do train_arch "$c" 0; done

pair() {
  if [ ! -f "$ROOT/eval/pair_$1__$2.json" ]; then
    stamp "stage: eval-pair $1 vs $2"; acc eval-pair --ref "$1" --cand "$2"
  else stamp "skip: pair $1 $2"; fi
  if [ ! -f "$ROOT/eval/hungarian_$1__$2.json" ]; then
    acc eval-hungarian --ref "$1" --cand "$2"
  fi
}
pair "fp32_${ARCH}_s0" "fp32_${ARCH}_s1"          # arch's own seed floor
for c in $CODECS; do pair "fp32_${ARCH}_s0" "${c}_${ARCH}_s0"; done

stamp "=== $ARCH architecture check complete ==="
