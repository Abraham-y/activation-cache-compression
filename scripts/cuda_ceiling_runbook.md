# CUDA ceiling mini-replication runbook (~$10-15, ~4-6h on one A100/4090)

> EXECUTED 2026-08-10 via scripts/modal_cuda_ceiling.py (Modal A100,
> ~40 min GPU-time thanks to 99k tok/s training). Outcome: strongest
> case — ceiling EXACTLY 1.0 at m1 scale (bit-identical checkpoints),
> floor 22.8% (MPS matching pair: 23.2%), rot-int4 37.1% = 1.63x floor
> (MPS 1.64x). Results in runs_cuda_modal/{smoke,m1}/eval/.

Purpose: measure the run-noise ceiling, one floor pair, and one codec
point on CUDA, converting the paper's "MPS-specific exact ceiling"
limitation into a measured cross-hardware result.

## On the cloud box (Ubuntu + CUDA torch):

```bash
git clone <REPO_URL> && cd activation_cache_compression   # or rsync the repo (~50 MB, no runs/)
pip install torch transformers datasets numpy scipy matplotlib tqdm pytest zstandard
python3 -m pytest -q                       # SAELens tests skip if it is not installed
export PYTHONPATH=src

# smoke-scale first (~30 min total): tokens+cache+codec
python3 -m acc.cli --root runs_cuda --preset smoke tokens
python3 -m acc.cli --root runs_cuda --preset smoke cache
python3 -m acc.cli --root runs_cuda --preset smoke transcode --codec fp16   # null reads this
python3 -m acc.cli --root runs_cuda --preset smoke transcode --codec rot-int4
python3 -m acc.cli --root runs_cuda --preset smoke null --codec fp16 --canary

# CEILING: same seed twice (deterministic algorithms NOT forced — we want
# the stack's real behavior)
python3 -m acc.cli --root runs_cuda --preset smoke train --codec fp32 --seed 0
python3 -m acc.cli --root runs_cuda --preset smoke train --codec fp32 --seed 0 --rep 1
python3 -m acc.cli --root runs_cuda --preset smoke eval-pair --ref fp32_s0 --cand fp32_s0_r1

# FLOOR pair + codec point
python3 -m acc.cli --root runs_cuda --preset smoke train --codec fp32 --seed 1
python3 -m acc.cli --root runs_cuda --preset smoke eval-pair --ref fp32_s0 --cand fp32_s1
python3 -m acc.cli --root runs_cuda --preset smoke train --codec rot-int4 --seed 0
python3 -m acc.cli --root runs_cuda --preset smoke eval-pair --ref fp32_s0 --cand rot-int4_s0

# OPTIONAL second pass at m1 scale for the paper number (~4h):
# repeat the six commands above with --preset m1 (needs ~70 GB disk).
# Bring home: runs_cuda/eval/*.json (a few KB)
```

## Interpretation
- ceiling pair survived == 1.0 -> CUDA is deterministic too; strongest outcome.
- ceiling < 1.0 -> report it as the measured CUDA ceiling; codec effects
  are then read between the measured ceiling and floor (the protocol's
  designed behavior).
- Expect floor ~ +-2pp of the MPS value at matched preset; large
  deviations would be a finding in their own right.
