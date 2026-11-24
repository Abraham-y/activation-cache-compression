"""End-to-end training time by cache format and decode path, on an A100
(camera-ready addition).

WHY AN A100. On the M5 laptop SAE training is compute-bound (~10k tok/s;
compressed caches were no faster at 3x). The August A100 run was the
opposite: fp32 trained at ~100k tok/s with the loader busy 99% of the
time, and rot-int4 was 2.2x SLOWER because Modal's CPUs decoded it at
~45k tok/s. Whether a smaller cache speeds training therefore depends on
where decoding happens, so this compares seven loader configurations that
all feed bit-identical batches to identical training:
  fp32 / cpu, fp32 / gpu, fp16 / cpu, fp16 / gpu,
  std-int4 / cpu (the decode every paper run used, identity matmul and all),
  std-int4 / cpu-nomatmul, std-int4 / gpu.
Each is a full 1x run (8,000 steps x 4,096 = 32.8M token-visits, seed 0),
in its own fresh container (no page cache carried between configurations),
reading either straight from the Modal network volume ("volume", runs one
at a time so they never share its bandwidth) or from a copy on the
container's local disk ("local", copy time excluded). Two repeats each.

PRE-REGISTERED READINGS, fixed 7 Oct 2026 before any timing run.
Throughput = training tokens per second over the whole run (end to end).
"Best fp32" and "best std-int4" = the fastest decode path of each.
  V   Validity, must hold before any timing is read: within a codec, every
      decode path yields bit-identical trained weights (same SHA-256), and
      the decode-mode tests pass on CUDA in every container.
  T2  Volume: best std-int4 / best fp32 > 1.10 -> a smaller cache speeds
      training when loading is the bottleneck; 0.90-1.10 -> no material
      gain; < 0.90 -> compression costs throughput.
  T3  Local disk: the same ratio and branches. T2 and T3 together say
      whether any gain depends on storage bandwidth.
  T4  Descriptive: std-int4 cpu vs cpu-nomatmul vs gpu throughput, i.e. what
      the decoder implementation alone is worth.
  (T1, the laptop, lives in scripts/run_timing_m5.sh.)
Peak disk read rates (loader stats) are reported beside every number.

Usage:
  modal run scripts/modal_timing.py --stage prepare      # std-int4 transcode (CPU)
  modal run scripts/modal_timing.py --stage smoke        # 200-step plumbing check
  modal deploy scripts/modal_timing.py
  modal run scripts/modal_timing.py --stage deploy-run   # the grid, server-side
  modal run scripts/modal_timing.py --stage fetch        # results -> runs_timing/
(--stage run also works but can stop if this client loses its network.)
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import modal

REPO = pathlib.Path(__file__).parent.parent
GPU = "A100-80GB"
VOL_ROOT = "/runs/runs_cuda_m1"
OUT = "/runs/timing"
STEPS = 8000
STORE = {"fp32": "master", "fp16": "fp16", "std-int4": "std-int4"}
CONFIGS = [("fp32", "cpu"), ("fp32", "gpu"), ("fp16", "cpu"), ("fp16", "gpu"),
           ("std-int4", "cpu"), ("std-int4", "cpu-nomatmul"), ("std-int4", "gpu")]
REPEATS = 2

app = modal.App("acc-timing")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "numpy==2.4.2", "scipy==1.15.3", "pytest",
                 "zstandard", "transformers==5.9.0")
    .add_local_dir(str(REPO / "src"), remote_path="/repo/src")
    .add_local_dir(str(REPO / "tests"), remote_path="/repo/tests")
    .add_local_file(str(REPO / "conftest.py"), remote_path="/repo/conftest.py")
    .add_local_file(str(REPO / "pyproject.toml"), remote_path="/repo/pyproject.toml")
)
vol = modal.Volume.from_name("acc-runs-cuda")
ENV = {**os.environ, "PYTHONPATH": "/repo/src"}


def acc(root, *args):
    subprocess.run([sys.executable, "-m", "acc.cli", "--root", root,
                    "--preset", "m1", *args], check=True, cwd="/repo", env=ENV)


@app.function(image=image, cpu=8, memory=32768, timeout=6 * 3600,
              volumes={"/runs": vol})
def prepare() -> dict:
    vol.reload()
    m = pathlib.Path(VOL_ROOT) / "cache" / "std-int4-train" / "manifest.json"
    if not m.exists():
        t0 = time.time()
        acc(VOL_ROOT, "transcode", "--codec", "std-int4")
        vol.commit()
        print(f"std-int4 transcode: {time.time() - t0:.0f}s")
    return {c: json.loads((pathlib.Path(VOL_ROOT) / "cache" / f"{s}-train" /
                           "manifest.json").read_text())["payload_bytes_per_token"]
            for c, s in STORE.items()}


def _weights_sha(path):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        if hasattr(v, "numpy"):
            h.update(k.encode())
            h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()


@app.function(image=image, gpu=GPU, cpu=8, memory=32768,
              ephemeral_disk=512 * 1024,   # Modal's minimum (MiB)
              single_use_containers=True,  # a fresh container per run: no page
                                           # cache or scratch state carried over
              timeout=8 * 3600,   # volume reads have varied 13-655 MB/s
              volumes={"/runs": vol})
def bench(codec: str, decode: str, location: str, rep: int, steps: int) -> dict:
    import torch
    vol.reload()
    name = f"{location}_{codec}_{decode}_r{rep}_{steps}"
    done = pathlib.Path(OUT) / f"{name}.json"
    if done.exists():
        return json.loads(done.read_text())
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_store.py"],
                   check=True, cwd="/repo", env=ENV)
    import tempfile
    root = pathlib.Path(tempfile.mkdtemp(prefix="root_", dir="/tmp"))
    (root / "cache").mkdir(parents=True)
    src = pathlib.Path(VOL_ROOT) / "cache" / f"{STORE[codec]}-train"
    dst = root / "cache" / f"{STORE[codec]}-train"
    copy_s = None
    if location == "volume":
        dst.symlink_to(src, target_is_directory=True)
    else:
        t0 = time.time()
        shutil.copytree(src, dst)
        copy_s = time.time() - t0
    # page-cache drop is best effort (may be unsupported in the sandbox)
    try:
        subprocess.run(["sync"], check=False)
        pathlib.Path("/proc/sys/vm/drop_caches").write_text("3\n")
        dropped = True
    except Exception:
        dropped = False
    tag = f"t{location[:3]}{decode.replace('-', '')}"
    t0 = time.time()
    acc(str(root), "train", "--codec", codec, "--seed", "0", "--steps", str(steps),
        "--tag", tag, "--decode", decode)
    wall = time.time() - t0
    sdir = root / "saes" / f"{codec}_{tag}_s0"
    summ = json.loads((sdir / "summary.json").read_text())
    log = [json.loads(l) for l in open(sdir / "train_log.jsonl")]
    res = {"name": name, "codec": codec, "decode": decode, "location": location,
           "rep": rep, "steps": steps, "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "cpu_count": os.cpu_count(),
           "copy_seconds": copy_s, "page_cache_dropped": dropped,
           "wall_seconds": wall, "train_seconds": summ["train_seconds"],
           "tokens_per_s": steps * 4096 / summ["train_seconds"],
           "loader": summ.get("loader"), "final_log": log[-1],
           "weights_sha256": _weights_sha(sdir / "sae.pt")}
    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(res, indent=2))
    vol.commit()
    print(json.dumps({k: res[k] for k in ("name", "tokens_per_s", "train_seconds",
                                          "weights_sha256")}))
    return res


@app.function(image=image, cpu=1, memory=2048, timeout=24 * 3600,  # Modal max
              volumes={"/runs": vol})
def orchestrate(steps: int = STEPS, repeats: int = REPEATS,
                do_local: bool = True) -> list:
    """Runs the grid server-side. Launch it from a DEPLOYED app
    (`--stage deploy-run`): on 7 Oct a detached `modal run` died with the
    laptop's network and took the volume half with it. Local-disk runs in
    parallel (copy excluded); volume runs strictly one at a time so they
    never share its bandwidth."""
    local = [bench.spawn(c, d, "local", r, steps)
             for r in range(repeats) for c, d in CONFIGS] if do_local else []
    out = []
    for r in range(repeats):
        for c, d in CONFIGS:
            res = bench.remote(c, d, "volume", r, steps)
            print(res["name"], round(res["tokens_per_s"]), flush=True)
            out.append(res["name"])
    for h in local:
        res = h.get()
        print(res["name"], round(res["tokens_per_s"]), flush=True)
        out.append(res["name"])
    return out


@app.local_entrypoint()
def main(stage: str = "smoke", do_local: bool = True):
    if stage == "prepare":
        print(prepare.remote())
    elif stage == "smoke":
        for c, d in (("std-int4", "gpu"), ("std-int4", "cpu-nomatmul")):
            r = bench.remote(c, d, "volume", 0, 200)
            print(r["name"], round(r["tokens_per_s"]), r["weights_sha256"][:12])
    elif stage == "run":
        print(orchestrate.remote())
    elif stage == "deploy-run":
        # needs `modal deploy scripts/modal_timing.py` first; the call then
        # lives on Modal and survives this client going away. Runs whose
        # result JSON already exists on the volume return immediately, so
        # this is restartable. (Our 7 Oct relaunch used --no-do-local: the
        # local-disk half had already finished.)
        fc = modal.Function.from_name("acc-timing", "orchestrate").spawn(
            STEPS, REPEATS, do_local)
        print(f"spawned deployed orchestrator, call id {fc.object_id}")
    elif stage == "fetch":
        dest = REPO / "runs_timing"
        dest.mkdir(exist_ok=True)
        n = 0
        for entry in vol.listdir("timing"):
            p = entry.path
            data = b"".join(vol.read_file(p))
            (dest / pathlib.Path(p).name).write_bytes(data)
            n += 1
        print(f"fetched {n} result files to {dest}")
