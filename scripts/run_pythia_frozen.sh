#!/bin/bash
# Frozen-basis eval at the two Pythia sites — the missing arm of the
# construct-validity claim.
#
# WHY: an internal audit found that the frozen judge's CONTINUOUS statistic
# (mean_corr) ranks the GPT-2 codecs against retrained survival at
# Spearman rho = 0.99. That makes the proxy UNCALIBRATED rather than
# INVALID — unless the frozen -> retrained mapping fails to TRANSFER
# across sites, in which case calibrating it once is useless and the
# strong claim is restored. We have never measured it anywhere but
# GPT-2: `ls runs_m4_*/eval/frozen_*` is empty at both Pythia sites.
#
# THE TEST: at GPT-2, rot-int4 sits at frozen r = 0.9854 -> 37.7%
# retrained. If the mapping is site-invariant, the same frozen r at
# Pythia should predict the same retrained survival. It will not, if the
# site's own floor (15.7% / 18.4% vs 23.0%) governs the level. Either
# answer decides how §4 should be framed.
#
# COST: the Pythia caches were reclaimed by cleanup.sh, so the masters
# must be regenerated. The TRAIN master is needed even though we only
# evaluate on eval, because the rotated codecs calibrate mu/sigma and the
# rotation from the first 262k train tokens — calibrating on eval instead
# would produce a different codec from the one the SAEs were trained on.
# ~100 GB and ~80 min per site. Sites run SEQUENTIALLY and each site's
# caches are deleted before the next starts, so peak disk stays ~100 GB.
#
# The SAE checkpoints and token sets survived cleanup, so nothing is
# retrained and the token set is identical to the original runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source "$(dirname "$0")/_env.sh"
stamp() { echo "[$(date '+%F %T')] $*"; }

CODECS="fp16 int8 rot-int8 rot-int4 int4"
KEEP_EVAL_MASTER=${KEEP_EVAL_MASTER:-1}   # 8.2 GB/site of cheap insurance

for other in run_m1.sh run_m4.sh run_arch.sh run_m6.sh run_fwdprec.sh run_ksweep_codec.sh; do
  while pgrep -f "$other" > /dev/null 2>&1; do
    stamp "waiting: $other still active"; sleep 300
  done
done

for spec in "runs_m4_L8 8" "runs_m4_L16 16"; do
  set -- $spec
  R=$1; L=$2
  CLI="$PY -m acc.cli --root $R --preset m4-pythia"
  stamp "===== $R (layer $L) ====="

  df -h . | tail -1 | awk -v r="$R" '{print "  disk free before "r": "$4}'

  # 1. regenerate the masters (bf16 forward, same as the original runs,
  #    which is what makes fp16 storage bit-lossless at these sites)
  if [ ! -f "$R/cache/master-eval/manifest.json" ] \
     || [ ! -f "$R/cache/master-train/manifest.json" ]; then
    stamp "stage: regenerate masters (~80 min)"
    $CLI cache --layer "$L"
  fi

  # 2. eval-split codec stores, calibrated from the train master
  for c in $CODECS; do
    stamp "stage: transcode $c (eval split only)"
    "$PY" scripts/transcode_eval_only.py --root "$R" --preset m4-pythia --codec "$c"
  done

  # 3. the measurement
  for c in $CODECS; do
    if [ ! -f "$R/eval/frozen_${c}__fp32_s0.json" ]; then
      stamp "stage: eval-frozen $c"
      $CLI eval-frozen --codec "$c"
    fi
  done

  # 4. reclaim before the next site; optionally keep the eval master,
  #    which is the split every future eval at this site would need
  stamp "stage: reclaim $R caches"
  if [ "$KEEP_EVAL_MASTER" = "1" ]; then
    find "$R/cache" -mindepth 1 -maxdepth 1 -not -name "master-eval" -exec rm -rf {} +
    stamp "  kept $R/cache/master-eval ($(du -sh "$R/cache/master-eval" | cut -f1))"
  else
    rm -rf "$R/cache"
  fi
done

stamp "===== does the frozen -> retrained mapping transfer? ====="
"$PY" - <<'EOF'
import json
from pathlib import Path
import numpy as np

def load(root):
    out = {}
    for f in sorted(Path(root, "eval").glob("frozen_*__fp32_s0.json")):
        c = f.name[len("frozen_"):-len("__fp32_s0.json")]
        fr = json.loads(f.read_text())
        p = Path(root, "eval", f"pair_fp32_s0__{c}_s0.json")
        if not p.exists():
            continue
        out[c] = (fr["mean_corr"], json.loads(p.read_text())["survived"])
    return out

sites = {"GPT-2 L6": load("runs_m1"), "Pythia L8": load("runs_m4_L8"),
         "Pythia L16": load("runs_m4_L16")}
floors = {}
for name, root in (("GPT-2 L6", "runs_m1"), ("Pythia L8", "runs_m4_L8"),
                   ("Pythia L16", "runs_m4_L16")):
    v = [json.loads(Path(root, "eval",
         f"pair_fp32_s{a}__fp32_s{b}.json").read_text())["survived"]
         for a, b in ((0, 1), (0, 2), (1, 2))]
    floors[name] = float(np.mean(v))

print(f"\n{'codec':<10}" + "".join(f"{s:>26}" for s in sites))
print(f"{'':<10}" + "".join(f"{'frozen r -> retrained':>26}" for _ in sites))
common = set.intersection(*(set(v) for v in sites.values()))
for c in sorted(common, key=lambda c: -sites["GPT-2 L6"][c][0]):
    row = f"{c:<10}"
    for s in sites:
        fr, rt = sites[s][c]
        row += f"{fr:>16.5f} ->{100*rt:>6.1f}%"
    print(row)
print(f"\n{'floor':<10}" + "".join(f"{100*floors[s]:>24.1f}%" for s in sites))
print("\nSAME frozen r, DIFFERENT retrained survival across sites => the")
print("mapping does NOT transfer, and a once-calibrated proxy is useless.")
for c in sorted(common):
    vals = [(s, *sites[s][c]) for s in sites]
    fr = [v[1] for v in vals]; rt = [v[2] for v in vals]
    if max(fr) - min(fr) < 0.01:   # matched frozen reading
        print(f"  {c}: frozen r within {max(fr)-min(fr):.4f} across sites, "
              f"retrained spans {100*min(rt):.1f}-{100*max(rt):.1f}%")
EOF
stamp "===== complete ====="
