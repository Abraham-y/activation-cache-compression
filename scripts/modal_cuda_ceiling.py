"""CUDA ceiling replication on Modal (scripts/cuda_ceiling_runbook.md).

Runs the runbook's command sequence on an A100: tokens -> cache ->
transcode -> null+canary -> ceiling pair (same seed twice) -> floor pair
-> rot-int4 codec point, all at --preset smoke or m1. runs_cuda_* lives
on a Modal Volume so a crashed/resumed run skips completed stages (the
CLI's skip-if-artifact-exists guards do the rest). Eval JSONs are
returned and written locally under runs_cuda_modal/<scale>/eval/.

Usage:
    modal run scripts/modal_cuda_ceiling.py --scale smoke   # ~15 min, sanity
    modal run scripts/modal_cuda_ceiling.py --scale m1      # ~2-4 h, paper number
"""
import json
import os
import pathlib
import subprocess
import sys

import modal

REPO = pathlib.Path(__file__).parent.parent

app = modal.App("acc-cuda-ceiling")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "datasets", "numpy", "scipy",
                 "matplotlib", "tqdm", "pytest", "zstandard")
    .add_local_dir(str(REPO / "src"), remote_path="/repo/src")
    .add_local_dir(str(REPO / "tests"), remote_path="/repo/tests")
    .add_local_file(str(REPO / "conftest.py"), remote_path="/repo/conftest.py")
    .add_local_file(str(REPO / "pyproject.toml"), remote_path="/repo/pyproject.toml")
)

vol = modal.Volume.from_name("acc-runs-cuda", create_if_missing=True)


@app.function(image=image, gpu="A100", timeout=8 * 3600, volumes={"/runs": vol})
def run_ceiling(scale: str = "smoke") -> dict:
    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
          f"cuda {torch.version.cuda}")
    env = {**os.environ, "PYTHONPATH": "/repo/src"}
    root = f"/runs/runs_cuda_{scale}"

    def sh(args, skip_if=None, retries=1):
        """skip_if: artifact path (relative to root) that marks this stage
        done — needed for `train`, whose CLI skip-guard exits nonzero.
        retries: rerun once on SIGSEGV (observed as a CUDA teardown race
        after the checkpoint was already written)."""
        if skip_if and (pathlib.Path(root) / skip_if).exists():
            print(f"\n=== skip {' '.join(args)}: {skip_if} exists ===", flush=True)
            return
        for attempt in range(retries + 1):
            print(f"\n=== {' '.join(args)} ===", flush=True)
            try:
                subprocess.run([sys.executable, "-m", "acc.cli",
                                "--root", root, "--preset", scale, *args],
                               check=True, cwd="/repo", env=env)
                vol.commit()
                return
            except subprocess.CalledProcessError as e:
                vol.commit()  # persist whatever the stage wrote before dying
                if e.returncode == -11 and attempt < retries:
                    if skip_if and (pathlib.Path(root) / skip_if).exists():
                        print("SIGSEGV at teardown but artifact exists — continuing")
                        return
                    print("SIGSEGV — retrying stage once")
                    continue
                raise

    subprocess.run([sys.executable, "-m", "pytest", "-q"],
                   check=True, cwd="/repo", env=env)

    sh(["tokens"])
    sh(["cache"])
    sh(["transcode", "--codec", "fp16"])  # the null check reads the fp16 store
    sh(["transcode", "--codec", "rot-int4"])
    sh(["null", "--codec", "fp16", "--canary"], skip_if="eval/null_fp16_raw.json")
    # ceiling: same seed twice (deterministic algorithms NOT forced —
    # we want the stack's real behavior)
    sh(["train", "--codec", "fp32", "--seed", "0"], skip_if="saes/fp32_s0/sae.pt")
    sh(["train", "--codec", "fp32", "--seed", "0", "--rep", "1"],
       skip_if="saes/fp32_s0_r1/sae.pt")
    sh(["eval-pair", "--ref", "fp32_s0", "--cand", "fp32_s0_r1"],
       skip_if="eval/pair_fp32_s0__fp32_s0_r1.json")
    # floor pair + codec point
    sh(["train", "--codec", "fp32", "--seed", "1"], skip_if="saes/fp32_s1/sae.pt")
    sh(["eval-pair", "--ref", "fp32_s0", "--cand", "fp32_s1"],
       skip_if="eval/pair_fp32_s0__fp32_s1.json")
    sh(["train", "--codec", "rot-int4", "--seed", "0"],
       skip_if="saes/rot-int4_s0/sae.pt")
    sh(["eval-pair", "--ref", "fp32_s0", "--cand", "rot-int4_s0"],
       skip_if="eval/pair_fp32_s0__rot-int4_s0.json")

    out = {}
    eval_dir = pathlib.Path(root) / "eval"
    for f in sorted(eval_dir.glob("*.json")):
        out[f.name] = json.loads(f.read_text())
    return out


@app.local_entrypoint()
def main(scale: str = "smoke"):
    results = run_ceiling.remote(scale=scale)
    dest = REPO / "runs_cuda_modal" / scale / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    for name, content in results.items():
        (dest / name).write_text(json.dumps(content, indent=2))
    print(f"\nwrote {len(results)} eval JSONs to {dest}")
    for name, r in results.items():
        if name.startswith("pair_"):
            print(f"{name}: survived={r.get('survived'):.4f} "
                  f"mean_r={r.get('mean_corr'):.4f}")
