"""Fixed token set construction.

The token set is built once, shuffled and split once, and every downstream
condition (codec, seed) consumes the identical rows. Correlations are only
meaningful on identical token positions, so this file is the root of that
guarantee.

Each row is ctx-1 real tokens; at caching time an EOT/BOS token is prepended
and its activation dropped (GPT-2's position-0 residual is an extreme
attention-sink outlier that would otherwise dominate per-token scales).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import write_json

EOT = 50256


def build_tokens(out_dir: str | Path, dataset: str, n_train_tokens: int,
                 n_eval_tokens: int, ctx: int = 128, seed: int = 0,
                 streaming: bool = False, text_field: str = "text",
                 pretokenized: bool = False,
                 data_files: list[str] | None = None,
                 tokenizer_name: str = "gpt2") -> dict:
    """pretokenized=True reads GPT-2 token ids straight from `text_field`
    (e.g. apollo-research/*-tokenizer-gpt2 'input_ids'); the stream already
    carries document separators, so no extra EOT is appended.

    data_files: explicit repo-relative parquet files to hf_hub_download and
    read locally — one full-speed CDN fetch instead of streaming's flaky
    row-group range requests.
    """
    from datasets import load_dataset

    from transformers import AutoTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    eot = tok.eos_token_id
    id_dtype = np.uint16 if len(tok) <= 65535 else np.uint32

    row_len = ctx - 1  # activations per row after dropping the prepended EOT
    n_rows_train = int(np.ceil(n_train_tokens / row_len))
    n_rows_eval = int(np.ceil(n_eval_tokens / row_len))
    n_rows = n_rows_train + n_rows_eval

    if data_files:
        from huggingface_hub import hf_hub_download
        local = [hf_hub_download(repo_id=dataset, filename=f,
                                 repo_type="dataset") for f in data_files]
        ds = load_dataset("parquet", data_files=local, split="train")
    else:
        ds = load_dataset(dataset, split="train", streaming=streaming)
    rows = np.empty((n_rows, row_len), dtype=id_dtype)
    buf: list[int] = []
    filled = 0
    n_docs = 0
    for ex in ds:
        if pretokenized:
            buf.extend(ex[text_field])
        else:
            buf.extend(tok(ex[text_field])["input_ids"])
            buf.append(eot)
        n_docs += 1
        while len(buf) >= row_len and filled < n_rows:
            rows[filled] = np.asarray(buf[:row_len], dtype=id_dtype)
            del buf[:row_len]
            filled += 1
        if filled >= n_rows:
            break
    if filled < n_rows:
        raise RuntimeError(
            f"dataset exhausted: needed {n_rows} rows, got {filled}"
        )

    rng = np.random.default_rng(seed)
    rows = rows[rng.permutation(n_rows)]
    train, eval_ = rows[:n_rows_train], rows[n_rows_train:]
    np.save(out_dir / "tokens_train.npy", train)
    np.save(out_dir / "tokens_eval.npy", eval_)
    meta = {
        "dataset": dataset,
        "tokenizer": tokenizer_name,
        "eot_id": int(eot),
        "pretokenized": pretokenized,
        "ctx": ctx,
        "row_len": row_len,
        "prepend_eot": True,
        "seed": seed,
        "n_docs_consumed": n_docs,
        "n_rows_train": int(n_rows_train),
        "n_rows_eval": int(n_rows_eval),
        "n_train_tokens": int(n_rows_train * row_len),
        "n_eval_tokens": int(n_rows_eval * row_len),
    }
    write_json(out_dir / "meta.json", meta)
    return meta
