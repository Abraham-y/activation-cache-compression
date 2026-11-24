"""CLI for the Milestone 1 pipeline.

    python -m acc.cli tokens     --preset smoke
    python -m acc.cli cache      --preset smoke
    python -m acc.cli transcode  --preset smoke --codec int8
    python -m acc.cli train      --preset smoke --codec int8 --seed 0
    python -m acc.cli null       --preset smoke
    python -m acc.cli eval-frozen --preset smoke --codec int8
    python -m acc.cli eval-pair  --preset smoke --ref fp32_s0 --cand int8_s0
    python -m acc.cli plot       --preset smoke

Store layout under --root:
    tokens/  cache/master-{train,eval}  cache/<codec>-{train,eval}
    saes/<codec>_s<seed>[_r<rep>]  eval/  plots/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .codecs import get_codec, theoretical_bytes_per_token
from .metrics import (StreamingCross, StreamingPaired, band_fractions,
                      decoder_cosine_match, direct_distortion, topk_to_csr)
from .sae import SAEConfig, load_sae, train_sae
from .store import BatchLoader, Store, transcode
from .utils import get_device, read_json, write_json

PRESETS = {
    "smoke": dict(
        model="gpt2", tokenizer="gpt2",
        dataset="NeelNanda/pile-10k", streaming=False,
        n_train_tokens=250_000, n_eval_tokens=60_000, ctx=128, layer=6,
        batch_rows=64, shard_tokens=32_768,
        d_sae=1536, k=16, aux_k=128, batch_size=1024, steps=600,
        dead_tokens=100_000, lr=1e-3, log_every=50,
        eval_tokens=60_000, calib_tokens=65_536,
    ),
    "m1": dict(
        model="gpt2", tokenizer="gpt2",
        dataset="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        streaming=False, pretokenized=True, text_field="input_ids",
        data_files=["data/train-00000-of-00073.parquet"],
        n_train_tokens=20_000_000, n_eval_tokens=1_000_000, ctx=128, layer=6,
        batch_rows=64, shard_tokens=262_144,
        d_sae=12288, k=32, aux_k=512, batch_size=4096, steps=8000,
        dead_tokens=2_000_000, lr=3e-4, log_every=100,
        eval_tokens=500_000, calib_tokens=262_144,
    ),
    # M4 scale check (dataset verified streaming 2026-07-31). Laptop-sized:
    # 10M train tokens, bf16 forward, 6000 steps. Layer set per-root via
    # `cache --layer` (driver runs layers 8 and 16 in separate roots).
    "m4-pythia": dict(
        model="EleutherAI/pythia-1.4b", tokenizer="EleutherAI/pythia-1.4b",
        model_dtype="bf16",
        dataset="monology/pile-uncopyrighted", streaming=True,
        pretokenized=False,
        n_train_tokens=10_000_000, n_eval_tokens=1_000_000, ctx=128, layer=8,
        batch_rows=32, shard_tokens=131_072,
        d_sae=16384, k=32, aux_k=512, batch_size=4096, steps=6000,
        dead_tokens=2_000_000, lr=3e-4, log_every=100,
        eval_tokens=500_000, calib_tokens=262_144,
    ),
}


def paths(root: str):
    r = Path(root)
    return {
        "tokens": r / "tokens",
        "cache": r / "cache",
        "saes": r / "saes",
        "eval": r / "eval",
        "plots": r / "plots",
    }


def cache_dir(root, codec, split):
    name = "master" if codec == "fp32" else codec
    return paths(root)["cache"] / f"{name}-{split}"


def cmd_tokens(args, p):
    from .data import build_tokens
    meta = build_tokens(
        paths(args.root)["tokens"], p["dataset"],
        p["n_train_tokens"], p["n_eval_tokens"],
        ctx=p["ctx"], seed=0, streaming=p["streaming"],
        pretokenized=p.get("pretokenized", False),
        text_field=p.get("text_field", "text"),
        data_files=p.get("data_files"),
        tokenizer_name=p.get("tokenizer", "gpt2"),
    )
    print(f"tokens: {meta['n_train_tokens']} train / {meta['n_eval_tokens']} eval"
          f" ({meta['n_docs_consumed']} docs)")


def cmd_cache(args, p):
    from .model import cache_activations
    device = args.device or get_device()
    tokens_meta = read_json(paths(args.root)["tokens"] / "meta.json")
    for split in ("train", "eval"):
        if (cache_dir(args.root, "fp32", split) / "manifest.json").exists():
            print(f"skip master-{split}: manifest exists")
            continue
        t0 = time.perf_counter()
        m = cache_activations(
            paths(args.root)["tokens"] / f"tokens_{split}.npy",
            cache_dir(args.root, "fp32", split),
            layer=args.layer or p["layer"], batch_rows=p["batch_rows"],
            shard_tokens=p["shard_tokens"], device=device,
            model_name=p.get("model", "gpt2"),
            prepend_id=tokens_meta.get("eot_id", 50256),
            model_dtype=p.get("model_dtype", "fp32"),
        )
        print(f"cached master-{split}: {m['n_tokens']} tokens, "
              f"{m['payload_bytes_per_token']:.0f} B/tok, "
              f"{m['throughput']['tokens_per_s']:.0f} tok/s, "
              f"{time.perf_counter()-t0:.1f}s")


def cmd_transcode(args, p):
    for split in ("train", "eval"):
        m = transcode(
            cache_dir(args.root, "fp32", split),
            cache_dir(args.root, args.codec, split),
            args.codec,
            calib_tokens=p["calib_tokens"],
            calib_master_dir=cache_dir(args.root, "fp32", "train"),
        )
        print(f"{args.codec}-{split}: {m['n_tokens']} tokens, "
              f"{m['payload_bytes_per_token']:.1f} B/tok payload "
              f"({m['disk_bytes_per_token']:.1f} on disk), "
              f"{m['transcode_seconds']:.1f}s")


def sae_dir(root, codec, seed, rep=0, tag=""):
    name = codec + (f"_{tag}" if tag else "") + f"_s{seed}" \
        + (f"_r{rep}" if rep else "")
    return paths(root)["saes"] / name


def cmd_train(args, p):
    device = args.device or get_device()
    store = Store(cache_dir(args.root, args.codec, args.split))
    cfg = SAEConfig(
        d_in=store.d, d_sae=args.d_sae or p["d_sae"], k=args.k or p["k"],
        aux_k=p["aux_k"], dead_tokens=p["dead_tokens"], lr=p["lr"],
        batch_size=p["batch_size"], steps=args.steps or p["steps"],
        log_every=p["log_every"], arch=args.arch, l1_coef=args.l1,
    )
    tag = args.tag
    if args.arch != "topk" and args.arch not in tag:
        tag = f"{args.arch}{('-' + tag) if tag else ''}"
    out = sae_dir(args.root, args.codec, args.seed, args.rep, tag)
    if (out / "sae.pt").exists() and not args.overwrite:
        sys.exit(f"{out} exists; pass --overwrite to retrain")
    loader = BatchLoader(store, cfg.batch_size, seed=args.seed, device=device,
                         decode=args.decode)
    t0 = time.perf_counter()
    train_sae(loader, cfg, seed=args.seed, device=device, out_dir=out,
              codec_name=args.codec, init_seed=args.init_seed)
    s = loader.stats()
    print(f"trained {out.name}: {cfg.steps} steps in {time.perf_counter()-t0:.0f}s "
          f"({cfg.steps*cfg.batch_size/(time.perf_counter()-t0):.0f} tok/s, "
          f"disk read {s['read_disk_MB_per_s']:.0f} MB/s)")


def _iter_eval_pairs(root, codec, max_tokens):
    """Aligned (master fp32, codec-decoded) eval batches."""
    master = Store(cache_dir(root, "fp32", "eval"))
    other = Store(cache_dir(root, codec, "eval"))
    assert len(master.shards) == len(other.shards)
    seen = 0
    for i in range(len(master.shards)):
        if seen >= max_tokens:
            break
        xa, _ = master.read_shard(i)
        xb, _ = other.read_shard(i)
        assert xa.shape == xb.shape
        take = min(xa.shape[0], max_tokens - seen)
        seen += take
        yield xa[:take], xb[:take]


def _dense_feature_acts(sae, x_np, device, chunk=8192):
    """Dense feature activations [n, d_sae] float32 (CPU) via TopK scatter."""
    outs = []
    for c0 in range(0, x_np.shape[0], chunk):
        x = torch.from_numpy(x_np[c0 : c0 + chunk]).to(device)
        vals, idx = sae.feature_acts(x)
        z = torch.zeros(x.shape[0], sae.d_sae, device=device)
        z.scatter_(1, idx, vals)
        outs.append(z.cpu().numpy())
    return np.concatenate(outs, axis=0)


def cmd_eval_frozen(args, p):
    """Duan-style: frozen SAE, per-feature Pearson between codec and fp32."""
    device = args.device or get_device()
    ref_name = args.sae or "fp32_s0"
    sae, _ = load_sae(paths(args.root)["saes"] / ref_name, device)
    from .metrics import direct_distortion_arrays
    paired = StreamingPaired(sae.d_sae)
    rels, coss = [], []
    for xa, xb in _iter_eval_pairs(args.root, args.codec, p["eval_tokens"]):
        rel, cos = direct_distortion_arrays(xa, xb)
        rels.append(rel)
        coss.append(cos)
        # chunked: whole-shard dense [n, d_sae] is ~8.6 GB/side at M4 scale
        for c0 in range(0, xa.shape[0], 8192):
            za = _dense_feature_acts(sae, xa[c0 : c0 + 8192], device)
            zb = _dense_feature_acts(sae, xb[c0 : c0 + 8192], device)
            paired.update(za, zb)
    corr = paired.pearson()
    alive = paired.firing_rate_x() > 1e-3
    bands = band_fractions(corr, alive)
    rel_all, cos_all = np.concatenate(rels), np.concatenate(coss)
    direct_agg = {  # global statistics over all eval tokens
        "rel_l2_mean": float(rel_all.mean()),
        "rel_l2_p99": float(np.percentile(rel_all, 99)),
        "cos_mean": float(cos_all.mean()),
        "cos_min": float(cos_all.min()),
    }
    result = {
        "mode": "frozen", "codec": args.codec, "sae": ref_name,
        "n_eval_tokens": paired.n, "direct": direct_agg, **bands,
    }
    out = paths(args.root)["eval"] / f"frozen_{args.codec}__{ref_name}.json"
    write_json(out, result)
    np.savez(out.with_suffix(".npz"), corr=corr, alive=alive,
             firing_rate=paired.firing_rate_x())
    print(f"frozen {args.codec} vs fp32 through {ref_name}: "
          f"mean r={bands.get('mean_corr', float('nan')):.4f} "
          f"survived={bands.get('survived', 0):.3f} "
          f"damaged={bands.get('damaged', 0):.3f}  -> {out}")


def cmd_null(args, p):
    """fp16-vs-fp16 null: identical-precision comparison must give r=1.

    Runs the full measurement pipeline with the SAME store on both sides
    (two independent decode passes). Any deviation from 1.0 is pipeline
    error, not signal. With --canary, also misaligns one side by one token
    and requires correlation to DROP, proving the metric detects drift.
    """
    device = args.device or get_device()
    codec = args.codec or "fp16"
    store_dir = cache_dir(args.root, codec, "eval")
    a_store, b_store = Store(store_dir), Store(store_dir)
    sae = None
    sae_path = paths(args.root)["saes"] / (args.sae or "fp32_s0") / "sae.pt"
    if sae_path.exists():
        sae, _ = load_sae(sae_path.parent, device)
    m = sae.d_sae if sae else a_store.d
    paired = StreamingPaired(m)
    canary = StreamingPaired(m) if args.canary else None
    seen = 0
    for i in range(len(a_store.shards)):
        if seen >= p["eval_tokens"]:
            break
        xa, _ = a_store.read_shard(i)
        xb, _ = b_store.read_shard(i)
        take = min(xa.shape[0], p["eval_tokens"] - seen)
        xa, xb = xa[:take], xb[:take]
        seen += take
        xb_rolled = np.roll(xb, 1, axis=0) if canary is not None else None
        # chunked: whole-shard dense [n, d_sae] blows memory at M4 scale
        for c0 in range(0, xa.shape[0], 8192):
            sl = slice(c0, c0 + 8192)
            za = _dense_feature_acts(sae, xa[sl], device) if sae else xa[sl]
            zb = _dense_feature_acts(sae, xb[sl], device) if sae else xb[sl]
            paired.update(za, zb)
            if canary is not None:
                zc = (_dense_feature_acts(sae, xb_rolled[sl], device)
                      if sae else xb_rolled[sl])
                canary.update(za, zc)
    corr = paired.pearson()
    alive = (paired.firing_rate_x() > 1e-3) if sae else np.ones(m, bool)
    if not alive.any():
        sys.exit(f"null check FAIL [{codec}]: frozen SAE has NO features "
                 f"firing >0.1% on eval tokens — dead/collapsed SAE")
    bands = band_fractions(corr, alive)
    basis = f"frozen SAE {sae_path.parent.name}" if sae else "raw channels"
    ok = bands["mean_corr"] > 0.999999 and bands["damaged"] == 0.0
    print(f"null check [{codec} vs {codec}, {basis}, {paired.n} tokens]: "
          f"mean r={bands['mean_corr']:.7f} min r={np.nanmin(corr[alive]):.7f} "
          f"damaged={bands['damaged']:.1%} -> {'PASS' if ok else 'FAIL'}")
    canary_result = {}
    if canary is not None:
        cb = band_fractions(canary.pearson(), alive)
        sens = cb["mean_corr"] < 0.99
        print(f"misalignment canary: mean r={cb['mean_corr']:.4f} "
              f"-> {'PASS (metric is sensitive)' if sens else 'FAIL (metric blind to drift!)'}")
        ok = ok and sens
        canary_result = {"canary_mean_corr": cb["mean_corr"],
                         "canary_pass": bool(sens)}
    write_json(paths(args.root)["eval"] / f"null_{codec}_{basis.split()[0]}.json",
               {"mode": "null", "codec": codec, "basis": basis, **bands,
                "pass": bool(ok), **canary_result})
    if not ok:
        sys.exit(1)


def cmd_eval_pair(args, p):
    """Cross-SAE comparison: for each alive ref feature, best-match Pearson
    in the cand SAE, on identical fp32 eval tokens for both sides."""
    device = args.device or get_device()
    ref, _ = load_sae(paths(args.root)["saes"] / args.ref, device)
    cand, _ = load_sae(paths(args.root)["saes"] / args.cand, device)
    master = Store(cache_dir(args.root, "fp32", "eval"))
    cross = StreamingCross(ref.d_sae, cand.d_sae)
    fvu_num = {"ref": 0.0, "cand": 0.0}
    censored = {"ref": 0, "cand": 0}
    x_sum = np.zeros(master.d)
    x2_sum = 0.0
    seen = 0
    for i in range(len(master.shards)):
        if seen >= p["eval_tokens"]:
            break
        xa, _ = master.read_shard(i)
        take = min(xa.shape[0], p["eval_tokens"] - seen)
        xa = xa[:take]
        seen += take
        x_sum += xa.sum(axis=0, dtype=np.float64)
        x2_sum += float(np.einsum("ij,ij->", xa, xa, dtype=np.float64))
        for c0 in range(0, xa.shape[0], 8192):
            x = torch.from_numpy(xa[c0 : c0 + 8192]).to(device)
            outs = {}
            for name, sae in (("ref", ref), ("cand", cand)):
                vals, idx = sae.feature_acts(x)
                recon, _ = sae.decode(vals, idx)
                # float64 on CPU: a float32 .sum() over 8192x768 on MPS is
                # nondeterministic across processes and was observed 12.5%
                # wrong. best_corr is unaffected (StreamingCross is float64
                # numpy) but FVU was silently corrupted in two eval files.
                fvu_num[name] += float(
                    np.square((recon - x).detach().cpu().numpy()
                              .astype(np.float64)).sum())
                # non-topk archs cap feature_acts at eval_k: if the last
                # (smallest) slot is active, true L0 >= eval_k — censored.
                # For topk, eval_k == k and the last slot is almost always
                # active by construction — no censoring is possible.
                if sae.arch != "topk":
                    censored[name] += int((vals[:, -1] > 0).sum())
                outs[name] = topk_to_csr(
                    vals.cpu().numpy(), idx.cpu().numpy(), sae.d_sae)
            cross.update(outs["ref"], outs["cand"])
    n = cross.n
    var_total = x2_sum - (x_sum**2).sum() / n  # total sum of squares about mean
    best_corr, best_idx = cross.match()
    rate = cross.firing_rate_x()
    alive = rate > 1e-3
    if not alive.any():
        sys.exit(f"eval-pair: ref {args.ref} has NO features firing >0.1% "
                 f"on eval tokens — dead/collapsed SAE, refusing to write "
                 f"a meaningless eval")
    bands = band_fractions(best_corr, alive)
    dec_cos = decoder_cosine_match(
        ref.W_dec.detach().cpu().numpy(), cand.W_dec.detach().cpu().numpy())
    result = {
        "mode": "pair", "ref": args.ref, "cand": args.cand,
        "n_eval_tokens": n, **bands,
        "decoder_cos_mean_alive": float(dec_cos[alive].mean()),
        "fvu_ref": fvu_num["ref"] / var_total,
        "fvu_cand": fvu_num["cand"] / var_total,
        "alive_frac_ref": float(alive.mean()),
        # eval-validity flags: censored = tokens whose true L0 hit the
        # eval_k cap (non-topk archs); cand_dead = codec SAE that diverged
        "censored_frac_ref": censored["ref"] / n,
        "censored_frac_cand": censored["cand"] / n,
        "cand_dead": bool(fvu_num["cand"] / var_total > 0.95),
    }
    if result["censored_frac_ref"] > 0.01 or result["censored_frac_cand"] > 0.01:
        print(f"WARNING: eval_k censoring — ref {result['censored_frac_ref']:.1%}, "
              f"cand {result['censored_frac_cand']:.1%} of tokens hit the cap; "
              f"correlations are computed on truncated traces")
    if result["cand_dead"]:
        print(f"WARNING: cand {args.cand} reconstructs nothing "
              f"(fvu={result['fvu_cand']:.3f}) — codec killed training; "
              f"survival numbers reflect divergence, not feature drift")
    out = paths(args.root)["eval"] / f"pair_{args.ref}__{args.cand}.json"
    # npz FIRST, guard-checked json LAST: a crash between the two writes
    # must not leave a guard-passing json with missing per-feature arrays
    np.savez(out.with_suffix(".npz"), best_corr=best_corr, best_idx=best_idx,
             firing_rate_ref=rate, alive=alive, decoder_cos=dec_cos)
    write_json(out, result)
    print(f"pair {args.ref} vs {args.cand}: mean r={bands['mean_corr']:.4f} "
          f"survived={bands['survived']:.3f} degraded={bands['degraded']:.3f} "
          f"damaged={bands['damaged']:.3f}  -> {out}")


def cmd_eval_semantic(args, p):
    """Semantic validation of the correlation proxy: do matched features
    fire on the same inputs? For each alive ref feature and its matched
    cand feature (from the saved pair npz), compare top activating
    positions (strict, same fixed token set) and top activating token ids
    (looser, 'fires on the same token type'), grouped by survival band.
    If r>0.9 features show high overlap and damaged ones don't, the
    Pearson banding means what the paper claims it means."""
    from .metrics import (jaccard, overlap_fraction, top_positions_per_feature)
    device = args.device or get_device()
    pair_npz = paths(args.root)["eval"] / f"pair_{args.ref}__{args.cand}.npz"
    z = np.load(pair_npz)
    best_idx, best_corr, alive = z["best_idx"], z["best_corr"], z["alive"]

    ref, _ = load_sae(paths(args.root)["saes"] / args.ref, device)
    cand, _ = load_sae(paths(args.root)["saes"] / args.cand, device)
    master = Store(cache_dir(args.root, "fp32", "eval"))
    max_tokens = args.tokens

    triples = {"ref": [], "cand": []}
    seen = 0
    for i in range(len(master.shards)):
        if seen >= max_tokens:
            break
        x, _ = master.read_shard(i)
        x = x[: max_tokens - seen]
        for c0 in range(0, x.shape[0], 8192):
            xb = torch.from_numpy(x[c0 : c0 + 8192]).to(device)
            gpos = seen + c0 + np.arange(xb.shape[0])
            for name, sae in (("ref", ref), ("cand", cand)):
                vals, idx = sae.feature_acts(xb)
                v = vals.cpu().numpy()
                j = idx.cpu().numpy()
                fired = v > 0
                triples[name].append((
                    j[fired].astype(np.int64),
                    v[fired],
                    np.broadcast_to(gpos[:, None], v.shape)[fired],
                ))
        seen += x.shape[0]

    tops = {}
    for name, m in (("ref", ref.d_sae), ("cand", cand.d_sae)):
        feat = np.concatenate([t[0] for t in triples[name]])
        val = np.concatenate([t[1] for t in triples[name]])
        pos = np.concatenate([t[2] for t in triples[name]])
        tops[name] = top_positions_per_feature(feat, val, pos, m, args.top)

    tok_rows = np.load(paths(args.root)["tokens"] / "tokens_eval.npy")
    row_len = tok_rows.shape[1]

    def pos_to_tokens(positions):
        return tok_rows[positions // row_len, positions % row_len]

    bands = {"survived": [], "degraded": [], "damaged": []}
    tok_bands = {"survived": [], "degraded": [], "damaged": []}
    corr_np = np.nan_to_num(best_corr)
    for f in np.nonzero(alive)[0]:
        band = ("survived" if corr_np[f] > 0.9
                else "degraded" if corr_np[f] >= 0.5 else "damaged")
        a, b = tops["ref"][f], tops["cand"][best_idx[f]]
        bands[band].append(overlap_fraction(a, b))
        tok_bands[band].append(jaccard(pos_to_tokens(a), pos_to_tokens(b)))

    result = {"mode": "semantic", "ref": args.ref, "cand": args.cand,
              "n_tokens": seen, "top_n": args.top}
    for band in bands:
        po = np.array(bands[band], dtype=np.float64)
        tj = np.array(tok_bands[band], dtype=np.float64)
        result[band] = {
            "n": int(po.size),
            "pos_overlap_mean": float(np.nanmean(po)) if po.size else None,
            "tok_jaccard_mean": float(np.nanmean(tj)) if tj.size else None,
        }
    out = paths(args.root)["eval"] / f"semantic_{args.ref}__{args.cand}.json"
    write_json(out, result)
    for band in ("survived", "degraded", "damaged"):
        r = result[band]
        if r["n"]:
            print(f"{band:>9}: n={r['n']:>5}  pos-overlap "
                  f"{r['pos_overlap_mean']:.3f}  tok-jaccard "
                  f"{r['tok_jaccard_mean']:.3f}")
    print(f"-> {out}")


def cmd_eval_hungarian(args, p):
    """Paulo & Belrose-style alignment: one-to-one Hungarian matching that
    maximizes mean encoder+decoder cosine similarity. Weight-space only —
    no activation pass — so it runs from checkpoints in seconds and is
    directly comparable to the seed-stability literature."""
    from scipy.optimize import linear_sum_assignment
    ref, _ = load_sae(paths(args.root)["saes"] / args.ref, "cpu")
    cand, _ = load_sae(paths(args.root)["saes"] / args.cand, "cpu")

    def unit(a, axis):
        return a / np.linalg.norm(a, axis=axis, keepdims=True).clip(min=1e-12)

    enc_r = unit(ref.W_enc.detach().numpy(), 0)    # [d, m] columns
    enc_c = unit(cand.W_enc.detach().numpy(), 0)
    dec_r = unit(ref.W_dec.detach().numpy(), 1)    # [m, d] rows
    dec_c = unit(cand.W_dec.detach().numpy(), 1)
    sim = ((enc_r.T @ enc_c) + (dec_r @ dec_c.T)) / 2
    row, col = linear_sum_assignment(-sim)
    matched = sim[row, col]
    result = {
        "mode": "hungarian", "ref": args.ref, "cand": args.cand,
        "mean_cos": float(matched.mean()),
        "median_cos": float(np.median(matched)),
        "frac_above_0.7": float((matched > 0.7).mean()),
        "frac_above_0.9": float((matched > 0.9).mean()),
        "cos_deciles": [float(v) for v in
                        np.percentile(matched, np.arange(0, 101, 10))],
    }
    out = paths(args.root)["eval"] / f"hungarian_{args.ref}__{args.cand}.json"
    write_json(out, result)
    np.savez(out.with_suffix(".npz"), matched_cos=matched, cand_idx=col)
    print(f"hungarian {args.ref} vs {args.cand}: mean cos={matched.mean():.4f} "
          f">0.7: {(matched > 0.7).mean():.3f}  -> {out}")


def cmd_stability(args, p):
    """Stratify codec feature-survival by cross-seed stability (per arXiv
    2606.12138: unstable features are individually non-reproducible anyway,
    so codec damage to them is not a real loss). A ref feature is 'stable'
    if its best match exceeds --threshold in EVERY listed fp32 seed pair."""
    eval_dir = paths(args.root)["eval"]
    stable = None
    for other in args.seeds:
        z = np.load(eval_dir / f"pair_{args.ref}__{other}.npz")
        ok = np.nan_to_num(z["best_corr"]) > args.threshold
        stable = ok if stable is None else (stable & ok)
        alive = z["alive"]
    rows = []
    for f in sorted(eval_dir.glob(f"pair_{args.ref}__*.npz")):
        cand = f.stem.split("__", 1)[1]
        z = np.load(f)
        corr = np.nan_to_num(z["best_corr"])
        surv = corr > 0.9
        a = z["alive"]
        rows.append({
            "cand": cand,
            "survived_all": float(surv[a].mean()),
            "survived_stable": float(surv[a & stable].mean()),
            "survived_unstable": float(surv[a & ~stable].mean()),
        })
    result = {"mode": "stability", "ref": args.ref, "seed_pairs": args.seeds,
              "threshold": args.threshold,
              "n_alive": int(alive.sum()),
              "stable_frac_of_alive": float(stable[alive].mean()),
              "rows": rows}
    thr = str(args.threshold).replace(".", "p")
    write_json(eval_dir / f"stability_{args.ref}_t{thr}.json", result)
    print(f"stable features (r>{args.threshold} vs {len(args.seeds)} seeds): "
          f"{stable[alive].mean():.1%} of alive")
    print(f"{'cand':<16} {'all':>7} {'stable':>7} {'unstable':>9}")
    for r in rows:
        print(f"{r['cand']:<16} {r['survived_all']:>7.3f} "
              f"{r['survived_stable']:>7.3f} {r['survived_unstable']:>9.3f}")


FAMILY_COLORS = {  # dataviz categorical slots 1-5 (light mode)
    "float": "#2a78d6", "int": "#eb6834", "int-o": "#1baf7a",
    "proj": "#eda100", "rot": "#e87ba4",
}


def codec_family(name):
    if name.startswith(("fp", "bf")):
        return "float"
    if name.startswith("rot"):
        return "rot"
    if name.startswith("int"):
        return "int-o" if "-o" in name else "int"
    return "proj"


def cmd_plot(args, p):
    """Rate-distortion: measured bytes/token vs feature survival, with the
    across-seed fp32 noise floor as the reference band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def seed_of(name):
        return name.rsplit("_s", 1)[1].split("_r")[0]

    eval_dir = paths(args.root)["eval"]
    points, floors = [], []
    for f in sorted(eval_dir.glob("pair_fp32_s*__fp32_s*.json")):
        r = read_json(f)
        if seed_of(r["ref"]) != seed_of(r["cand"]):  # replicates are the ceiling, not the floor
            floors.append(r)
    for f in sorted(eval_dir.glob("pair_fp32_s0__*.json")):
        r = read_json(f)
        cand_codec = r["cand"].rsplit("_s", 1)[0]
        if cand_codec == "fp32":
            continue
        if "_r" in r["cand"].rsplit("_s", 1)[1]:
            continue  # same-seed replicate (ceiling check), not a codec point
        man_path = cache_dir(args.root, cand_codec, "train") / "manifest.json"
        if not man_path.exists():
            continue  # tagged cand (relu/k-sweep) — not a plain codec point
        man = read_json(man_path)
        points.append((man["payload_bytes_per_token"], r, cand_codec))
    if not points:
        sys.exit("no pair_fp32_s0__<codec> eval results found")
    d = read_json(cache_dir(args.root, "fp32", "train") / "manifest.json")["d"]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.grid(True, which="major", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if floors:
        fl = min(f["survived"] for f in floors)
        fh = max(f["survived"] for f in floors)
        ax.axhspan(fl, fh, color="#bbbbbb", alpha=0.35, zorder=1,
                   label=None, lw=0)
        ax.axhline((fl + fh) / 2, color="#8a8a8a", lw=1.0, ls="--", zorder=1)
        ax.annotate("fp32 seed-noise floor", xy=(0.99, (fl + fh) / 2),
                    xycoords=("axes fraction", "data"), ha="right",
                    va="bottom", fontsize=8, color="#5a5a5a")
    seen_fam = set()
    for bpt, r, codec in sorted(points, key=lambda t: (t[0], t[2])):
        fam = codec_family(codec)
        ax.scatter(bpt, r["survived"], s=55, zorder=3,
                   color=FAMILY_COLORS[fam],
                   label=fam if fam not in seen_fam else None)
        seen_fam.add(fam)
        ax.annotate(codec, xy=(bpt, r["survived"]), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#333333")
    ax.axvline(4 * d, color="#8a8a8a", lw=0.8, ls=":", zorder=1)
    ax.annotate("fp32", xy=(4 * d, 0.5), xycoords=("data", "axes fraction"),
                fontsize=8, color="#5a5a5a", ha="right", rotation=90)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("bytes per token (measured payload)")
    ax.set_ylabel("fraction of features survived (r > 0.9)")
    ax.set_ylim(0, 1.02)
    if len(seen_fam) > 1:
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Feature survival vs. cache size", fontsize=11)
    out = paths(args.root)["plots"] / "rate_distortion.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def cmd_probe_cache(args, p):
    from .probes import TASKS, cache_task_activations
    device = args.device or get_device()
    task_dir = Path(args.root) / "probes" / args.task
    for split in TASKS[args.task]["splits"]:
        m = cache_task_activations(
            args.task, split, task_dir / f"master-{split}",
            layer=p["layer"], device=device, max_examples=args.max_examples,
            model_name=p.get("model", "gpt2"))
        print(f"{args.task} {split}: {m['n_examples']} examples cached")


def cmd_probe_transcode(args, p):
    from .probes import TASKS
    task_dir = Path(args.root) / "probes" / args.task
    for split in TASKS[args.task]["splits"]:
        out = task_dir / f"{args.codec}-{split}"
        m = transcode(task_dir / f"master-{split}", out, args.codec,
                      calib_tokens=p["calib_tokens"],
                      calib_master_dir=task_dir / "master-train")
        # labels travel with every store so loaders stay self-contained
        import shutil
        shutil.copy(task_dir / f"master-{split}" / "labels.npy",
                    out / "labels.npy")
        print(f"{args.task} {args.codec}-{split}: "
              f"{m['payload_bytes_per_token']:.1f} B/example-vector")


def cmd_probe_run(args, p):
    from .probes import TASKS, run_probe_comparison
    task_dir = Path(args.root) / "probes" / args.task
    r = run_probe_comparison(task_dir, args.codec,
                             TASKS[args.task]["n_classes"])
    r["task"] = args.task
    out = paths(args.root)["eval"] / f"probe_{args.task}__{args.codec}.json"
    write_json(out, r)
    print(f"probe {args.task} {args.codec}: acc {r['acc_codec']:.4f} "
          f"(fp32 {r['acc_fp32']:.4f}, delta {r['acc_codec']-r['acc_fp32']:+.4f}) "
          f"direction cos {r['direction_cos']:.4f}  -> {out}")


def cmd_codecs(args, p):
    d = args.d
    names = ["fp32", "fp16", "bf16", "int8", "int8-o8", "int6", "int6-o8",
             "int4", "int4-o16", "pca256", "pca384", "rp256"]
    print(f"{'codec':<10} {'B/tok (d=' + str(d) + ')':>14} {'vs fp32':>8} {'vs fp16':>8}")
    for n in names:
        b = theoretical_bytes_per_token(n, d)
        print(f"{n:<10} {b:>14.1f} {4*d/b:>7.1f}x {2*d/b:>7.1f}x")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="acc")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--preset", default="m1", choices=list(PRESETS))
    ap.add_argument("--device", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tokens")
    s = sub.add_parser("cache")
    s.add_argument("--layer", type=int, default=None)
    s = sub.add_parser("transcode")
    s.add_argument("--codec", required=True)
    s = sub.add_parser("train")
    s.add_argument("--codec", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--rep", type=int, default=0)
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--k", type=int, default=None)
    s.add_argument("--d-sae", type=int, default=None)
    s.add_argument("--tag", default="")
    s.add_argument("--init-seed", type=int, default=None,
                   help="decouple parameter init from the data-order seed "
                        "(init-only floor); default: init follows --seed")
    s.add_argument("--arch", default="topk",
                   choices=["topk", "relu", "jumprelu"])
    s.add_argument("--l1", type=float, default=0.0)
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--split", default="train",
                   help="store split to train on: <codec>-<split> under "
                        "cache/ (e.g. 'fresh' for the fresh-token store)")
    s.add_argument("--decode", default="cpu", choices=BatchLoader.DECODE_MODES,
                   help="where the loader decodes (same batches in every mode)")
    s = sub.add_parser("eval-frozen")
    s.add_argument("--codec", required=True)
    s.add_argument("--sae", default=None)
    s = sub.add_parser("eval-pair")
    s.add_argument("--ref", required=True)
    s.add_argument("--cand", required=True)
    s = sub.add_parser("eval-hungarian")
    s.add_argument("--ref", required=True)
    s.add_argument("--cand", required=True)
    s = sub.add_parser("eval-semantic")
    s.add_argument("--ref", required=True)
    s.add_argument("--cand", required=True)
    s.add_argument("--tokens", type=int, default=200_000)
    s.add_argument("--top", type=int, default=50)
    s = sub.add_parser("stability")
    s.add_argument("--ref", default="fp32_s0")
    s.add_argument("--seeds", nargs="+", default=["fp32_s1", "fp32_s2"])
    s.add_argument("--threshold", type=float, default=0.9)
    s = sub.add_parser("null")
    s.add_argument("--codec", default="fp16")
    s.add_argument("--sae", default=None)
    s.add_argument("--canary", action="store_true")
    sub.add_parser("plot")
    s = sub.add_parser("probe-cache")
    s.add_argument("--task", required=True)
    s.add_argument("--max-examples", type=int, default=None)
    s = sub.add_parser("probe-transcode")
    s.add_argument("--task", required=True)
    s.add_argument("--codec", required=True)
    s = sub.add_parser("probe-run")
    s.add_argument("--task", required=True)
    s.add_argument("--codec", required=True)
    s = sub.add_parser("codecs")
    s.add_argument("--d", type=int, default=768)
    args = ap.parse_args(argv)
    p = PRESETS[args.preset]
    {
        "tokens": cmd_tokens, "cache": cmd_cache, "transcode": cmd_transcode,
        "train": cmd_train, "eval-frozen": cmd_eval_frozen,
        "eval-pair": cmd_eval_pair, "eval-hungarian": cmd_eval_hungarian,
        "eval-semantic": cmd_eval_semantic,
        "stability": cmd_stability, "null": cmd_null, "plot": cmd_plot,
        "probe-cache": cmd_probe_cache, "probe-transcode": cmd_probe_transcode,
        "probe-run": cmd_probe_run, "codecs": cmd_codecs,
    }[args.cmd](args, p)


if __name__ == "__main__":
    main()
