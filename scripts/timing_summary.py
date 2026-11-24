"""Apply the pre-registered timing readings (V, T1-T4; headers of
scripts/run_timing_m5.sh and scripts/modal_timing.py) to the finished runs.

Inputs: M5 runs runs_m1/saes/<codec>_tm5<decode>_s0/ and the A100 result
JSONs fetched to runs_timing/ (modal_timing.py --stage fetch).
Writes runs_m1/eval/timing_summary.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

CONFIGS = [("fp32", "cpu"), ("fp32", "gpu"), ("fp16", "cpu"), ("fp16", "gpu"),
           ("std-int4", "cpu"), ("std-int4", "cpu-nomatmul"), ("std-int4", "gpu")]


def sha(path):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].contiguous().numpy().tobytes())
    return h.hexdigest()


def ratio_branch(r):
    return ("smaller cache speeds training" if r > 1.10
            else "compression costs throughput" if r < 0.90 else "no material gain")


def validity(rows):
    """Within a codec every decode path must give the same weights."""
    out = {}
    for codec in ("fp32", "fp16", "std-int4"):
        hs = {r["decode"]: r["sha"] for r in rows if r["codec"] == codec}
        out[codec] = len(set(hs.values())) == 1 if hs else None
    return out


def main():
    res = {}
    # ---- M5
    m5 = []
    for codec, dec in CONFIGS:
        d = Path(f"runs_m1/saes/{codec}_tm5{dec.replace('-', '')}_s0")
        if not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        steps = s["cfg"]["steps"]
        m5.append({"codec": codec, "decode": dec,
                   "tokens_per_s": steps * s["cfg"]["batch_size"] / s["train_seconds"],
                   "train_seconds": s["train_seconds"],
                   "read_disk_MB_per_s": s.get("loader", {}).get("read_disk_MB_per_s"),
                   # checkpoints do not ship with the artifact
                   "sha": sha(d / "sae.pt") if (d / "sae.pt").exists() else None})
    if m5:
        base = next(r["tokens_per_s"] for r in m5 if r["codec"] == "fp32" and r["decode"] == "cpu")
        for r in m5:
            r["vs_fp32_cpu"] = r["tokens_per_s"] / base
        complete = len(m5) == len(CONFIGS)
        res["m5"] = {"rows": m5, "V": validity(m5), "complete": complete,
                     # T1 is undefined until all seven laptop configurations ran
                     "T1_all_within_10pct": (bool(all(abs(r["vs_fp32_cpu"] - 1) <= 0.10
                                                      for r in m5)) if complete else None)}
    # ---- A100
    files = sorted(Path("runs_timing").glob("*.json"))
    a100 = [json.loads(f.read_text()) for f in files]
    a100 = [r for r in a100 if r.get("steps") == 8000]
    for loc in ("volume", "local"):
        rows = []
        for codec, dec in CONFIGS:
            reps = [r for r in a100 if r["location"] == loc and r["codec"] == codec and r["decode"] == dec]
            if not reps:
                continue
            tps = [r["tokens_per_s"] for r in reps]
            rows.append({"codec": codec, "decode": dec, "n": len(reps),
                         "tokens_per_s": float(np.mean(tps)), "tps_range": [min(tps), max(tps)],
                         "read_disk_MB_per_s": float(np.mean([r["loader"]["read_disk_MB_per_s"] for r in reps])),
                         "read_share": float(np.mean([r["loader"]["read_seconds"] / r["train_seconds"] for r in reps])),
                         "sha": reps[0]["weights_sha256"],
                         "sha_consistent_across_reps": len({r["weights_sha256"] for r in reps}) == 1,
                         "gpu": reps[0]["gpu"], "page_cache_dropped": [r["page_cache_dropped"] for r in reps]})
        if not rows:
            continue
        best = lambda c: max((r for r in rows if r["codec"] == c), key=lambda r: r["tokens_per_s"])
        b32, b4 = best("fp32"), best("std-int4")
        ratio = b4["tokens_per_s"] / b32["tokens_per_s"]
        res[f"a100_{loc}"] = {"rows": rows, "V": validity(rows),
                              "best_fp32": b32["decode"], "best_std_int4": b4["decode"],
                              "ratio_best_std_int4_over_best_fp32": ratio,
                              "reading": ratio_branch(ratio)}
    # ---- derived numbers the paper quotes
    der = {}
    for loc in ("volume", "local"):
        v = res.get(f"a100_{loc}")
        if not v:
            continue
        tp = {(r["codec"], r["decode"]): r["tokens_per_s"] for r in v["rows"]}
        b32 = max(t for (c, _), t in tp.items() if c == "fp32")
        der[loc] = {
            "paper_cpu_decoder_slowdown_vs_best_fp32": b32 / tp[("std-int4", "cpu")],
            # like for like: fp32 through the same CPU loader
            "paper_cpu_decoder_slowdown_vs_fp32_cpu": tp[("fp32", "cpu")] / tp[("std-int4", "cpu")],
            "fp32_bytes_per_s_needed_MB": 3072 * b32 / 1e6,
            "std_int4_bytes_per_s_needed_MB": 388 * tp[("std-int4", "gpu")] / 1e6,
            "min_tokens_per_s_gpu_paths": min(t for (_, d), t in tp.items() if d == "gpu"),
            "max_tokens_per_s_gpu_paths": max(t for (_, d), t in tp.items() if d == "gpu")}
    vol = [r for r in a100 if r["location"] == "volume"]
    if vol:
        rates = [r["loader"]["read_disk_MB_per_s"] for r in vol]
        der["volume_read_MB_per_s_range"] = [min(rates), max(rates)]
        slow = min((r for r in vol if r["codec"] == "fp32"), key=lambda r: r["tokens_per_s"])
        der["slowest_fp32_volume_run"] = {
            "tokens_per_s": slow["tokens_per_s"], "decode": slow["decode"],
            "read_MB_per_s": slow["loader"]["read_disk_MB_per_s"],
            "fraction_of_best_fp32": slow["tokens_per_s"] / max(
                r["tokens_per_s"] for r in vol if r["codec"] == "fp32")}
    # the same headline ratio on SXM4 containers only (Modal mixed in PCIe)
    for loc in ("local", "volume"):
        sx = [r for r in a100 if r["location"] == loc and "SXM4" in r["gpu"]]
        best = {}
        for c in ("fp32", "std-int4"):
            means = [np.mean([r["tokens_per_s"] for r in sx if r["codec"] == c and r["decode"] == d])
                     for d in {r["decode"] for r in sx if r["codec"] == c}]
            best[c] = max(means) if means else None
        if all(best.values()):
            der.setdefault(loc, {})["ratio_sxm4_only"] = best["std-int4"] / best["fp32"]
    gp = [der[l][k] for l in ("local", "volume") if l in der
          for k in ("min_tokens_per_s_gpu_paths", "max_tokens_per_s_gpu_paths")]
    if gp:   # quoted in thousands in the text ("between 118.5 and 129.1")
        der["gpu_paths_k_tokens_per_s_range_all"] = [min(gp) / 1e3, max(gp) / 1e3]
    der["gpu_variants"] = sorted({r["gpu"] for r in a100})
    der["a100_runs"] = len(a100)
    der["a100_wall_hours_incl_copies"] = sum(r["wall_seconds"] + (r["copy_seconds"] or 0)
                                             for r in a100) / 3600
    m5 = {}
    for name in ("fp32_b3x_s0", "fp16_b3x_s0", "rot-int8_b3x_s0", "rot-int4_b3x_s0"):
        p = Path(f"runs_m1/saes/{name}/summary.json")
        if p.exists():
            s = json.loads(p.read_text())
            m5[name] = {"train_hours": s["train_seconds"] / 3600,
                        "loader_busy": s["loader"]["read_seconds"] / s["train_seconds"]}
    der["m5_3x"] = m5
    res["derived"] = der
    Path("runs_m1/eval/timing_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(der, indent=1))
    for k, v in res.items():
        if k == "derived":
            continue
        print(f"== {k}  V={v['V']}")
        for r in v["rows"]:
            extra = f"  x{r['vs_fp32_cpu']:.2f}" if "vs_fp32_cpu" in r else \
                    f"  read share {r['read_share']:.2f}  disk {r['read_disk_MB_per_s']:.0f} MB/s"
            print(f"  {r['codec']:>9} {r['decode']:>13}: {r['tokens_per_s']:>9,.0f} tok/s{extra}")
        if "reading" in v:
            print(f"  best std-int4 ({v['best_std_int4']}) / best fp32 ({v['best_fp32']}) = "
                  f"{v['ratio_best_std_int4_over_best_fp32']:.2f} -> {v['reading']}")
        if "T1_all_within_10pct" in v:
            print(f"  T1 all within 10% of fp32/cpu: {v['T1_all_within_10pct']}")


if __name__ == "__main__":
    main()
