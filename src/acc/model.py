"""GPT-2 activation extraction.

The master cache is always fp32, computed in fp32, and every codec store is a
transcode of the master — the model forward runs exactly once per token set,
so codec comparisons can never be confounded by forward-pass nondeterminism.

Layer convention: `layer` is the HuggingFace hidden_states index. Index L is
the residual stream after transformer block L (index 0 = embeddings), i.e.
TransformerLens resid_post of block L-1 == resid_pre of block L.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .codecs import get_codec
from .data import EOT
from .store import StoreWriter
from .utils import Throughput


def cache_activations(tokens_path: str | Path, out_dir: str | Path,
                      layer: int = 6, batch_rows: int = 32,
                      shard_tokens: int = 65536, device: str = "cpu",
                      model_name: str = "gpt2",
                      prepend_id: int = EOT,
                      model_dtype: str = "fp32") -> dict:
    """model_dtype: forward-pass precision ('fp32' | 'bf16' | 'fp16').
    The master store is always fp32 of whatever the model computed; a
    lower-precision forward redefines the reference self-consistently
    (some pipelines compute in bf16) and roughly halves caching time."""
    from transformers import AutoModel

    tokens = np.load(tokens_path)  # [n_rows, row_len] uint16/uint32
    n_rows, row_len = tokens.shape
    model = AutoModel.from_pretrained(model_name)
    if model_dtype != "fp32":
        model = model.to(torch.bfloat16 if model_dtype == "bf16"
                         else torch.float16)
    model.eval().to(device)
    d = model.config.hidden_size

    writer = StoreWriter(out_dir, get_codec("fp32"), d, source=str(tokens_path))
    tp = Throughput()
    buf = np.empty((shard_tokens, d), dtype=np.float32)
    buf_fill = 0
    tp.start()
    with torch.inference_mode():
        for r0 in range(0, n_rows, batch_rows):
            rows = tokens[r0 : r0 + batch_rows].astype(np.int64)
            ids = np.concatenate(
                [np.full((rows.shape[0], 1), prepend_id, dtype=np.int64), rows],
                axis=1,
            )
            input_ids = torch.from_numpy(ids).to(device)
            out = model(input_ids, output_hidden_states=True)
            h = out.hidden_states[layer][:, 1:, :]  # drop EOT sink position
            acts = h.reshape(-1, d).float().cpu().numpy()
            a0 = 0
            while a0 < acts.shape[0]:
                take = min(shard_tokens - buf_fill, acts.shape[0] - a0)
                buf[buf_fill : buf_fill + take] = acts[a0 : a0 + take]
                buf_fill += take
                a0 += take
                if buf_fill == shard_tokens:
                    writer.write_shard(buf)
                    buf_fill = 0
    if buf_fill:
        writer.write_shard(buf[:buf_fill].copy())
    manifest = writer.close()
    tp.stop(tokens=n_rows * row_len)
    manifest["hf_hidden_state_index"] = layer
    manifest["model"] = model_name
    manifest["model_dtype"] = model_dtype
    manifest["throughput"] = tp.summary()
    return manifest
