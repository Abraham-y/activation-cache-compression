#!/bin/bash
# NON-SAE, NON-PROBE ARM: causal steering vectors.
#
# Every fidelity measure in the paper reads the cache through an SAE, or
# through a linear probe fit to the same activations. Audit passes across all
# five review rounds named this as the ceiling on the paper's scope. This
# arm is neither: a difference-in-means steering vector extracted from the
# cache, injected back into the model's forward pass, scored by what it
# DOES to the output rather than by where it points.
#
# The protocol substitution is the contribution. A steering vector is
# deterministic given the data, so there is no seed to vary and the
# same-process control cannot be a seed floor. The analogue is the
# estimator's own nuisance variation: two disjoint halves of the same fp32
# data. That is the claim the paper wants -- the protocol is not "compare
# SAEs across seeds", it is "compare against the estimator's own nuisance
# variation", and reseeding vs resampling is an implementation detail.
#
# PRE-REGISTERED READING (fixed before running -- honor it):
#   A codec PASSES on this instrument if its ratio-to-floor is <= 1 on
#   BOTH readouts, i.e. it perturbs the direction, and the direction's
#   causal effect, no more than a data resample does.
#   If the codec ORDERING here matches the SAE ordering, the protocol
#   generalises beyond SAEs and that is the headline.
#   If the ordering DIFFERS, that is a more interesting result and must be
#   reported as such: it would mean SAE feature survival is not a proxy
#   for causal fidelity, which would bound the paper's own claims.
#   If the geometric and causal readouts disagree, report both and do not
#   pick the flattering one.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
source scripts/_env.sh
R=runs_m1
stamp(){ echo "[$(date '+%F %T')] $*"; }

TASK=${TASK:-sst2}
SPLIT=${SPLIT:-train}   # floor is n-dependent; use the LARGEST split
LAYER=${LAYER:-6}
CODECS=${CODECS:-"fp16 rot-int8 rot-int4 int4 pca256"}

# label-aligned caches already exist from the probe arm, under
# runs_m1/probes/<task>/<codec>-<split>. Reuse them; transcode only what
# is missing so this arm costs no new caching.
D="$R/probes/$TASK"
[ -d "$D/master-${SPLIT}" ] || { stamp "ABORT: no probe cache for $TASK; run probe-cache first"; exit 1; }
for c in $CODECS; do
  if [ ! -f "$D/${c}-${SPLIT}/manifest.json" ]; then
    stamp "stage: transcode probe cache -> ${c}"
    $PY -m acc.cli --root $R --preset m1 probe-transcode --task "$TASK" --codec "$c" \
      || { stamp "transcode failed for ${c}; skipping"; continue; }
  fi
done

for c in $CODECS; do
  [ -f "$D/${c}-${SPLIT}/manifest.json" ] || { stamp "no cache for ${c}; skip"; continue; }
  o="$R/eval/steering_${TASK}_${c}.json"
  [ -f "$o" ] && { stamp "have $o"; continue; }
  stamp "stage: steering comparison ${c}"
  $PY - "$c" "$TASK" "$LAYER" "$o" "$D" "$SPLIT" <<'PY'
import sys
from acc.steering import run_steering_comparison
codec, task, layer, out, d, sp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
PROMPTS = [
    "The movie was", "I thought the film", "Overall, this was",
    "My honest opinion is that it", "The acting felt",
    "Would I recommend it? It", "The script was", "In short, the picture",
]
r = run_steering_comparison(f"{d}/master-{sp}", f"{d}/{codec}-{sp}",
                            codec, layer, out, prompts=PROMPTS)
g = r["geometric"]
print(f"  {codec}: cos(fp32,codec)={g['cos_fp32_vs_codec']:.4f} "
      f"floor(half,half)={g['cos_floor_half_vs_half']:.4f}")
if "causal" in r:
    c = r["causal"]
    print(f"    causal cos={c['cos_effect_fp32_vs_codec']:.4f} "
          f"floor={c['cos_effect_floor_half_vs_half']:.4f} "
          f"ratio_to_floor_L2={c['ratio_to_floor_l2']:.3f}")
PY
done
stamp "=== steering arm complete ==="
