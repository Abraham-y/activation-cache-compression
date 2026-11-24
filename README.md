# Artifact — camera-ready release

Companion to "How Small Can We Go? Calibrating Activation-Cache
Compression for SAE Training" (Abraham Yeung, NeurIPS 2026 Workshop:
Interpretability as a Science; [arXiv:2611.08551](https://arxiv.org/abs/2611.08551)). Everything needed to check the paper's
numbers, and to re-run the pipeline from scratch.

## What is here

- `src/acc/` — the library (102 tests): codecs, sharded compressed store
  with a deterministic loader (CPU, matmul-free CPU and GPU-side decode,
  all yielding bit-identical batches), TopK/ReLU/JumpReLU SAEs and
  trainer, streaming fidelity metrics, probes, causal steering vectors
  (`steering.py`), a SAELens drop-in adapter.
- `scripts/` — restartable drivers reproducing every result, plus the
  table/figure generators and the number checker.
- `runs_*/eval/` — **every evaluation artifact the paper's numbers come
  from** (JSON + per-feature NPZ).
- `runs_*/saes/*/summary.json`, `train_log.jsonl` — per-run config and
  training curves.
- `runs_*/cache/*/manifest.json` — store manifests (measured bytes/token),
  and `cache_manifests_snapshot/` for caches reclaimed from disk.
- `runs_timing/` — the A100 timing runs (Appendix H), one JSON per run.
- `runs_m1/tokens/meta*.json` — how the token sets were drawn.
- `docs/appendix-semantic-pairs-*.md` — side-by-side activating contexts
  for matched feature pairs (Appendix B).
- `runs_m4.log` — the Pythia driver log, which holds the Pythia
  misalignment-check values (Appendix I).
- `paper/` — LaTeX source, generated tables, figures; the checker below
  reads it.

## What is NOT here, and why

- **Token id arrays** (`runs_*/tokens/*.npy`, ~230 MB). Deterministic:
  the train and eval sets regenerate from the seed and dataset in
  `runs_*/tokens/meta.json` (`scripts/run_m1.sh`), and the fresh-text set from
  `python scripts/fresh_store.py tokens`.
- **Reference PDFs** (`paper/refs/`) — third-party papers, not ours to
  redistribute; all are cited by arXiv id.
- **SAE checkpoints** (`sae.pt`, ~12 GB) and **activation caches** (about
  450 GB at peak). Neither is needed to verify a number: every table and
  figure regenerates from the evaluation artifacts alone, which is checked
  below. They are needed only to re-run the evaluations themselves, and
  both regenerate from the drivers.

## Verify the paper in three commands

```bash
PYTHONPATH=src python -m pytest -q                        # 102 tests (10 need sae-lens)
PYTHONPATH=src python scripts/appendix_tables.py          # regenerates every appendix table
PYTHONPATH=src python scripts/verify_paper_numbers.py --appendix
```

The SAELens adapter tests skip when `sae-lens` is not installed (92 pass,
1 skips); `pip install sae-lens` runs all 102.

The third checks that every number in the paper's main text and appendix
prose traces to an artifact. Run `--selftest` first: it reports its own
power. It is a lint, not a proof — the real guarantee is that the tables
are generated, not typed.

## Reproduce from scratch

`scripts/run_*.sh` are guarded and restartable; each sources
`scripts/_env.sh`, which resolves an interpreter that can import the
dependencies and fails fast otherwise.

**Drivers (training and evaluation), in order:** `run_m1.sh` (main
grid), `run_m1_followup.sh` (rotation codecs), `run_seeds.sh` (extra
seeds), `run_m4.sh` (Pythia scale check; also transcodes the Pythia int8
stores), `run_arch.sh` (JumpReLU), `run_m6.sh` (sparsity x seed),
`run_m2.sh` (probes), `run_controls_a.sh` (controls, including the
Pythia int8 SAEs), `run_budget_scaling.sh` and `run_int4_budget.sh`
(budget tiers), `run_fwdprec.sh` and `run_fwdprec_b3x.sh` (bf16 forward
at 1x and 3x), `run_ksweep_codec.sh`, `run_pythia_frozen.sh`,
`run_controls_b.sh`, `run_noise_overlap.sh`, `run_sparse_rung.sh`,
`run_oneulp.sh`, `run_clump.sh` (locality control), `run_redraws.sh`
(repetitions and the initialization-only floor), `run_nbracket.sh`
(count bracket), `run_stdint4.sh` and `run_stdint4_seeds.sh` (std-int4,
the rotation ablation), `run_steering.sh` (causal steering, non-SAE);
then the camera-ready additions: `run_subspace.sh` and
`scripts/feature_ablation.py` (Appendix B, no training),
`run_fresh_b3x.sh` with `scripts/fresh_store.py`, `run_r3_resolution.sh`
(Appendix G), `scripts/modal_timing.py` (Appendix H, below), and
`run_geom_dynamics.sh` with `scripts/geom_dynamics.py` (Appendix J's
training-dynamics runs; it needs the fp32 training cache and about 7 h). Total
~218 GPU-hours of SAE training on one laptop, plus the 7 h of Appendix J's
dynamics runs.

**Analysis scripts** (read eval artifacts, write one JSON each to
`runs_*/eval/`, no training; run after the drivers, then
`appendix_tables.py` and `figures.py`, which writes to `docs/figures/`;
copy the PNGs into `paper/`):

| script | produces |
|---|---|
| `threshold_sweep.py`, `firing_rate_sweep.py`, `stable_floor.py`, `stratify_singlerun.py` | robustness of the ratio to threshold, alive cut, stable-only floor, single-run strata, one-to-one matching |
| `consensus_bias.py`, `paired_feature_tests.py`, `budget_fixedset.py` | consensus losses, paired feature tests, fixed stable set across budgets (Fig. 4) |
| `proxy_ranking.py`, `overlap_slopes.py`, `fragility.py`, `weight_divergence.py`, `misalignment_null.py` | proxy rankings, dose slopes, fragility predictors, weight divergence, matching null |
| `sparse_counts.py`, `entropy.py`, `bench_std_vs_rot.py`, `bench_throughput.py`, `bench_forward_dtype.py`, `streaming_comparison.py` | perturbed-value counts, zstd entropy, loader and forward throughput, streaming comparison |
| `probe_floor.py`, `steering_floor_splits.py`, `steering_matched_prompts.py`, `steering_split_counts.py` | probe and steering floors (Appendix I) |
| `subspace_summary.py`, `feature_ablation_summary.py`, `semantic_pairs.py`, `damaged_pair_semantics.py` | Appendix B summaries and the side-by-side contexts in `docs/` |
| `fresh_stable.py`, `fresh_summary.py`, `timing_summary.py` | Appendix G and H summaries |
| `geom_margins.py`, `geom_identifiability.py`, `geom_summary.py` | Appendix J: frozen-test margins, seed stability from one SAE, and the summary every Appendix J number comes from (eval caches and checkpoints only) |
| `summarize_controls.py`, `snapshot_m4_cache_throughput.py`, `verify_caption_numbers.py` | control readings, Pythia cache throughput, three caption sources |

Helpers the drivers call themselves: `calibrate_l1.py` (`run_arch.sh`),
`forward_precision_cache.py` (`run_fwdprec.sh`), `subspace_recovery.py`
(`run_subspace.sh`), `transcode_eval_only.py` (`run_pythia_frozen.sh`); `proof_scan.py` is a
style lint for the LaTeX.

`forward_dtype_throughput.json` (the bf16 forward's 2.21x) came from an
inline timing loop; `bench_forward_dtype.py` re-implements it and writes
a separate `_rerun` file, since timings depend on the machine.

Each driver that tests a hypothesis carries its pre-registered reading in
its header comment, written before the run launched.

## Running on Modal

The CUDA replication and the timing experiment (Appendix H) run on
[Modal](https://modal.com); the deploy code is included. With a Modal
account and the `modal` client authenticated (`modal setup`):

```bash
modal run scripts/modal_cuda_ceiling.py --scale m1      # CUDA ceiling/floor replication
modal run scripts/modal_timing.py --stage prepare       # std-int4 transcode on the volume
modal run scripts/modal_timing.py --stage smoke         # 200-step plumbing check
modal deploy scripts/modal_timing.py
modal run scripts/modal_timing.py --stage deploy-run    # the grid, server-side
modal run scripts/modal_timing.py --stage fetch         # results -> runs_timing/
PYTHONPATH=src python scripts/timing_summary.py
```

Launch the grid from the deployed app as shown: a detached `modal run`
can stop if the local client loses its connection. No credentials are
stored in this repository; Modal reads its token from the client's own
configuration.

## Paths

Absolute local paths in shipped files are rewritten to `<repo>` or `~`.
