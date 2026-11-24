#!/bin/bash
# Two controls demanded by the 2026-08-15 internal audit panel.
#
# A. FORWARD-PRECISION AT THREE SEEDS. §7's generality claim ("the
#    protocol is not specific to compression") rests on a single seed,
#    in a paper whose entire thesis is that single-seed comparisons are
#    uninterpretable. Two internal audit passes ranked fixing this first. The seed-0
#    arm (51.2%) is already measured against rot-int8's three-seed range
#    (53.8/51.6/52.7); this adds the matching s1 and s2 arms so the
#    comparison is range-vs-range instead of point-vs-range.
#    Cache already exists (57 GB), so this costs training only.
#
# B. DOSE-RESPONSE BELOW fp16. The design has a hole: zero perturbation
#    gives a bit-exact 1.0000 ceiling, and the smallest perturbation
#    ever tested (fp16, rel-L2 2.07e-4) already flips ~40% of features,
#    with NOTHING in between. Survival is near log-linear in rel-L2 over
#    the 2.1 decades we do cover (~6.6pp/decade), and extrapolating that
#    fit three decades below fp16 predicts ~80% survival from a
#    perturbation nobody would call a codec. If that extrapolation is
#    right, retrained survival is a property of SGD's sensitivity and
#    not of information loss, and the paper's question is ill-posed.
#
#    PRE-REGISTERED, before running (this matters -- the result is
#    genuinely two-sided and we do not get to pick after seeing it):
#      - If survival at 1e-6 and 1e-7 returns to ~100% (near the
#        bit-exact ceiling), fp16 sits on a real shoulder, damage is
#        information loss, and the chaos objection is dead. Best case.
#      - If survival at 1e-7 is still far below ceiling (say <85%), the
#        target is chaos-dominated at any nonzero dose. That REFRAMES
#        the paper -- survival numbers are only ever meaningful relative
#        to a same-process control, which is what the protocol already
#        insists on, but the framing "how many bits does an activation
#        need" would have to go.
#      - Anything in between traces the shoulder, which is publishable
#        as-is and sharpens §7's "damage tracks perturbation magnitude,
#        not its source".
#    Either way it gets reported. Four doses, not one, because the
#    open question is whether a CURVE exists, which one point
#    cannot answer.
#
#    noise{E} = fp32 storage + isotropic perturbation at exact per-token
#    rel-L2 of 1e-E. NOT a codec and not a rate-distortion point: it
#    stores 4d bytes/token and buys nothing. 1e-8 is deliberately absent
#    -- it is below fp32's representable relative precision (measured
#    ratio 0.69 of target), so the dose would be a lie.
#
# Disk: each noise store is full fp32 size (~60 GB train+eval) and is
# DELETED after its SAE is trained. Peak extra usage is one store.
# All stages guarded/restartable; re-running skips completed work.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
stamp() { echo "[$(date '+%F %T')] $*"; }

R=runs_m1
source "$(dirname "$0")/_env.sh"
CLI="$PY -m acc.cli --root $R --preset m1"

# never overlap another driver on the same GPU
for other in run_m1.sh run_m4.sh run_arch.sh run_m6.sh run_budget_scaling.sh \
             run_fwdprec.sh run_int4_budget.sh; do
  while pgrep -f "$other" > /dev/null 2>&1; do
    stamp "waiting: $other still active"; sleep 300
  done
done

free_gb() { df -g . | awk 'NR==2 {print $4}'; }

# ---------------------------------------------------------------- A ---
stamp "=== A. forward precision at seeds 1 and 2 ==="
for s in 1 2; do
  if [ ! -f "$R/saes/fwdbf16_s${s}/sae.pt" ]; then
    stamp "stage: train fwdbf16 seed $s (~55min)"
    $CLI train --codec fwdbf16 --seed "$s"
  fi
  # seed-matched pair, exactly as rot-int8's three-seed range is built
  if [ ! -f "$R/eval/pair_fp32_s${s}__fwdbf16_s${s}.json" ]; then
    stamp "stage: eval-pair fp32_s${s} vs fwdbf16_s${s}"
    $CLI eval-pair --ref "fp32_s${s}" --cand "fwdbf16_s${s}"
  fi
done

# ---------------------------------------------------------------- B ---
stamp "=== B. dose-response ladder below fp16 ==="
for e in 4 5 6 7; do
  N="noise${e}"
  if [ ! -f "$R/saes/${N}_s0/sae.pt" ]; then
    if [ ! -f "$R/cache/${N}-train/manifest.json" ]; then
      if [ "$(free_gb)" -lt 90 ]; then
        stamp "ABORT: $(free_gb) GB free, need ~90 GB headroom for ${N}"
        exit 1
      fi
      stamp "stage: transcode ${N} (fp32 + exact rel-L2 1e-${e})"
      $CLI transcode --codec "$N"
    fi
    # project rule: every new pipeline path reruns the null before any
    # number from it is trusted
    if [ ! -f "$R/eval/null_${N}_frozen.json" ]; then
      stamp "stage: null + misalignment canary on the ${N} store"
      $CLI null --codec "$N" --canary
    fi
    if [ ! -f "$R/eval/frozen_${N}__fp32_s0.json" ]; then
      stamp "stage: eval-frozen ${N} (what the proxy says about this dose)"
      $CLI eval-frozen --codec "$N"
    fi
    stamp "stage: train SAE on ${N} (~55min)"
    $CLI train --codec "$N" --seed 0
  fi
  if [ ! -f "$R/eval/pair_fp32_s0__${N}_s0.json" ]; then
    stamp "stage: eval-pair fp32_s0 vs ${N}_s0"
    $CLI eval-pair --ref fp32_s0 --cand "${N}_s0"
  fi
  # store has served its purpose (train + frozen + null); reclaim it
  if [ -d "$R/cache/${N}-train" ] && [ -f "$R/saes/${N}_s0/sae.pt" ] \
     && [ -f "$R/eval/pair_fp32_s0__${N}_s0.json" ]; then
    stamp "reclaim: rm ${N}-train ${N}-eval ($(du -sh $R/cache/${N}-train | cut -f1))"
    rm -rf "$R/cache/${N}-train" "$R/cache/${N}-eval"
  fi
done

# refresh stratification so the new arms get stable/unstable rows
stamp "stage: stability stratification (refresh)"
$CLI stability --ref fp32_s0 --seeds fp32_s1 fp32_s2 --threshold 0.9

stamp "=== controls complete ==="
$PY scripts/summarize_controls.py
