"""Fresh-token stores for the camera-ready budget experiment.

WHY. Camera-ready addition: the paper's 3x and 10x
budgets re-read one 20M-token cache, so a rising floor could be
convergence on a fixed corpus, and codecs were compared at equal tokens,
not equal bytes. This builds a 98.5M-token store whose first 77 shards
ARE the existing 20M-token cache (symlinked through the manifest, not
copied) and whose remaining 300 shards are new text, so:
  - fresh arms read 98.3M distinct tokens once at the 3x budget;
  - replay arms (existing) read the first 20M about five times;
  - the two share their first ~4,864 steps bit for bit (same shards, same
    seeded shuffle), which is itself a pipeline check.
The matched-bytes store is the first 302 shards of the fresh std-int4
store: 78.98M tokens in 30.64 GB, against the fp16 20M-token cache's
30.72 GB.

New text comes from the same local parquet file, starting at the first
document the original build never touched (meta n_docs_consumed), so no
document is shared with the train or eval rows; rows are cut exactly as
acc.data.build_tokens cuts them and shuffled with a fixed seed.

Subcommands (run in this order; each is idempotent and refuses to
overwrite):
  tokens          build runs_m1/tokens/tokens_freshtail.npy (+ overlap check)
  check-forward   re-cache the first master-train shard from its tokens and
                  require bit equality with the stored shard
  cache-tail      fp32 forward over the new rows -> cache/master-freshtail-train
  transcode-tail  std-int4 over the tail, codec state fitted exactly as the
                  existing std-int4-train was; require identical state
  merge           write cache/{master,std-int4}-fresh and std-int4-mb
                  manifests pointing at the shard files in place
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.cli import PRESETS, cache_dir, paths   # noqa: E402
from acc.store import Store, transcode          # noqa: E402
from acc.utils import get_device, read_json, write_json  # noqa: E402

ROOT = "runs_m1"
P = PRESETS["m1"]
TAIL_TOKENS = 78_500_000        # 20.0M + 78.5M > 98.304M token-visits at 3x
TAIL_SEED = 1                   # row shuffle for the new rows (train used 0)
MB_SHARDS = 302                 # matched-bytes prefix of std-int4-fresh


def cmd_tokens(_):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    tdir = paths(ROOT)["tokens"]
    out = tdir / "tokens_freshtail.npy"
    if out.exists():
        sys.exit(f"{out} exists")
    meta = read_json(tdir / "meta.json")
    row_len = meta["row_len"]
    start_doc = meta["n_docs_consumed"]
    local = hf_hub_download(repo_id=P["dataset"], filename=P["data_files"][0],
                            repo_type="dataset")
    col = pq.read_table(local, columns=[P["text_field"]]).column(P["text_field"])
    n_rows = -(-TAIL_TOKENS // row_len)
    # Documents are disjoint from the original build, but OpenWebText
    # repeats boilerplate and degenerate text ("i i i ...", newsletter
    # sign-ups), so a few 127-token rows recur verbatim (the original split
    # itself has 2 train rows equal to eval rows). Skip any row equal to an
    # existing train or eval row so the new text is strictly new.
    key = lambda r: hashlib.blake2b(r.tobytes(), digest_size=16).digest()
    old = {key(r) for split in ("train", "eval")
           for r in np.load(tdir / f"tokens_{split}.npy")}
    rows = np.empty((n_rows, row_len), dtype=np.uint16)
    buf, filled, doc, skipped = [], 0, start_doc, 0
    while filled < n_rows:
        if doc >= len(col):
            sys.exit(f"dataset file exhausted at {filled}/{n_rows} rows")
        buf.extend(col[doc].as_py())
        doc += 1
        while len(buf) >= row_len and filled < n_rows:
            r = np.asarray(buf[:row_len], dtype=np.uint16)
            del buf[:row_len]
            if key(r) in old:
                skipped += 1
                continue
            rows[filled] = r
            filled += 1
    rows = rows[np.random.default_rng(TAIL_SEED).permutation(n_rows)]
    dup = sum(key(r) in old for r in rows)
    np.save(out, rows)
    m = {"dataset": P["dataset"], "data_file": P["data_files"][0],
         "first_doc": start_doc, "docs_end": doc, "row_len": row_len,
         "n_rows": int(n_rows), "n_tokens": int(n_rows * row_len),
         "shuffle_seed": TAIL_SEED, "rows_skipped_as_repeats": int(skipped),
         "rows_equal_to_train_or_eval": int(dup)}
    write_json(tdir / "meta_freshtail.json", m)
    print(json.dumps(m, indent=1))
    if dup:
        sys.exit("FAIL: new rows repeat train/eval rows")


def cmd_check_forward(args):
    """Bit-equality of a fresh forward with the stored master-train shard 0,
    at the preset's batch shape. The fresh store mixes old and new shards,
    so the forward that makes the new ones must reproduce the old ones."""
    import tempfile
    from acc.model import cache_activations
    tdir = paths(ROOT)["tokens"]
    master = Store(cache_dir(ROOT, "fp32", "train"))
    n0 = master.shards[0]["n_tokens"]
    row_len = read_json(tdir / "meta.json")["row_len"]
    # whole 64-row batches, so every batch has the shape the original had
    n_rows = -(-(-(-n0 // row_len)) // P["batch_rows"]) * P["batch_rows"]
    rows = np.load(tdir / "tokens_train.npy")[:n_rows]
    with tempfile.TemporaryDirectory(dir=paths(ROOT)["cache"]) as tmp:
        np.save(Path(tmp) / "rows.npy", rows)
        cache_activations(Path(tmp) / "rows.npy", Path(tmp) / "store",
                          layer=P["layer"], batch_rows=P["batch_rows"],
                          shard_tokens=P["shard_tokens"],
                          device=args.device or get_device())
        x_new, _ = Store(Path(tmp) / "store").read_shard(0)
    x_old, _ = master.read_shard(0)
    same = x_new.shape == x_old.shape and np.array_equal(x_new, x_old)
    diff = float(np.abs(x_new.astype(np.float64) - x_old).max()) if x_new.shape == x_old.shape else float("nan")
    res = {"shard": 0, "n_tokens": int(n0), "bit_identical": bool(same),
           "max_abs_diff": diff}
    write_json(paths(ROOT)["eval"] / "fresh_forward_check.json", res)
    print(res)
    if not same:
        sys.exit("FAIL: forward does not reproduce the stored master shard")


def cmd_cache_tail(args):
    from acc.model import cache_activations
    out = cache_dir(ROOT, "fp32", "freshtail-train")
    if (out / "manifest.json").exists():
        sys.exit(f"{out} exists")
    m = cache_activations(paths(ROOT)["tokens"] / "tokens_freshtail.npy", out,
                          layer=P["layer"], batch_rows=P["batch_rows"],
                          shard_tokens=P["shard_tokens"],
                          device=args.device or get_device())
    print(f"cached tail: {m['n_tokens']} tokens, {m['throughput']}")


def cmd_transcode_tail(_):
    out = cache_dir(ROOT, "std-int4", "freshtail-train")
    if (out / "manifest.json").exists():
        sys.exit(f"{out} exists")
    m = transcode(cache_dir(ROOT, "fp32", "freshtail-train"), out, "std-int4",
                  calib_tokens=P["calib_tokens"],
                  calib_master_dir=cache_dir(ROOT, "fp32", "train"))
    a = np.load(cache_dir(ROOT, "std-int4", "train") / "codec_state.npz")
    b = np.load(out / "codec_state.npz")
    same = (sorted(a.files) == sorted(b.files)
            and all(np.array_equal(a[k], b[k]) for k in a.files))
    print(f"transcoded tail: {m['n_tokens']} tokens, {m['transcode_seconds']:.0f}s; "
          f"codec state identical to std-int4-train: {same}")
    if not same:
        sys.exit("FAIL: tail codec state differs from std-int4-train")


def _merged(name, parts, n_shards=None):
    """Manifest for a store whose shards live in other store dirs."""
    out = paths(ROOT)["cache"] / name
    if (out / "manifest.json").exists():
        sys.exit(f"{out} exists")
    base = read_json(parts[0] / "manifest.json")
    shards = []
    for d in parts:
        m = read_json(d / "manifest.json")
        assert m["codec"] == base["codec"] and m["codec_state"] == base["codec_state"]
        shards += [{**s, "file": f"../{d.name}/{s['file']}"} for s in m["shards"]]
    if n_shards is not None:
        shards = shards[:n_shards]
    n = sum(s["n_tokens"] for s in shards)
    pay = sum(s["payload_bytes"] for s in shards)
    man = {**base, "shards": shards, "n_tokens": n, "payload_bytes": pay,
           "payload_bytes_per_token": pay / n,
           "disk_bytes_per_token": sum(s["disk_bytes"] for s in shards) / n,
           "source": " + ".join(str(d) for d in parts),
           "codec_state_file": (f"../{parts[0].name}/{base['codec_state_file']}"
                                if base.get("codec_state_file") else None)}
    out.mkdir(parents=True)
    write_json(out / "manifest.json", man)
    return man


def cmd_merge(_):
    c = paths(ROOT)["cache"]
    mf = _merged("master-fresh", [c / "master-train", c / "master-freshtail-train"])
    sf = _merged("std-int4-fresh", [c / "std-int4-train", c / "std-int4-freshtail-train"])
    mb = _merged("std-int4-mb", [c / "std-int4-train", c / "std-int4-freshtail-train"],
                 n_shards=MB_SHARDS)
    fp16 = read_json(c / "fp16-train" / "manifest.json")
    res = {"master_fresh_tokens": mf["n_tokens"], "std_int4_fresh_tokens": sf["n_tokens"],
           "mb_tokens": mb["n_tokens"], "mb_payload_bytes": mb["payload_bytes"],
           "fp16_payload_bytes": fp16["payload_bytes"],
           "mb_bytes_over_fp16": mb["payload_bytes"] / fp16["payload_bytes"],
           "mb_tokens_over_fp16": mb["n_tokens"] / fp16["n_tokens"]}
    write_json(paths(ROOT)["eval"] / "fresh_store_layout.json", res)
    print(json.dumps(res, indent=1))
    assert mf["n_tokens"] >= 24_000 * P["batch_size"], "fresh store too small"
    assert res["mb_bytes_over_fp16"] <= 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tokens", "check-forward", "cache-tail",
                                    "transcode-tail", "merge"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--root", default=ROOT,
                    help="runs root (runs_m1 locally; the Modal root on CUDA)")
    args = ap.parse_args()
    globals()["ROOT"] = args.root
    {"tokens": cmd_tokens, "check-forward": cmd_check_forward,
     "cache-tail": cmd_cache_tail, "transcode-tail": cmd_transcode_tail,
     "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
