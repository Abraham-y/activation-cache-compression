"""Milestone 2: linear-probe fidelity benchmark.

Broadens the claim beyond SAEs: train logistic probes on codec-decoded
activations, evaluate on clean fp32 activations (deployment scenario,
mirroring the SAE eval), compare accuracy and learned-direction cosine
against the fp32-trained probe.

Probes are convex and zero-initialized, so the fp32-vs-fp32 "floor" is exact
determinism — codec effects are directly measurable with no seed noise.
This contrasts deliberately with the SAE benchmark, where seed noise is the
dominant reference frame.

Tasks are sentence-level HF datasets; the probed activation is GPT-2's
residual stream (same layer as M1) at the final real token of each example.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .codecs import get_codec
from .data import EOT
from .store import Store, StoreWriter
from .utils import write_json

TASKS = {
    "sst2": dict(hf="stanfordnlp/sst2", text="sentence", label="label",
                 splits={"train": "train", "val": "validation"}, n_classes=2),
    "ag_news": dict(hf="fancyzhx/ag_news", text="text", label="label",
                    splits={"train": "train", "val": "test"}, n_classes=4),
}


def cache_task_activations(task: str, split: str, out_dir: str | Path,
                           layer: int = 6, ctx: int = 128,
                           batch_size: int = 64, device: str = "cpu",
                           model_name: str = "gpt2",
                           max_examples: int | None = None) -> dict:
    """Cache the last-token layer activation per example + labels."""
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    spec = TASKS[task]
    ds = load_dataset(spec["hf"], split=spec["splits"][split])
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    tok = AutoTokenizer.from_pretrained(model_name)
    eot = tok.eos_token_id
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    d = model.config.hidden_size

    out_dir = Path(out_dir)
    writer = StoreWriter(out_dir, get_codec("fp32"), d, source=spec["hf"])
    labels = np.asarray(ds[spec["label"]], dtype=np.int64)
    texts = ds[spec["text"]]

    acts = np.empty((len(texts), d), dtype=np.float32)
    with torch.inference_mode():
        for b0 in range(0, len(texts), batch_size):
            batch = texts[b0 : b0 + batch_size]
            ids = [[eot] + tok(t)["input_ids"][: ctx - 1] for t in batch]
            lens = [len(s) for s in ids]
            m = max(lens)
            input_ids = torch.full((len(ids), m), eot, dtype=torch.long)
            mask = torch.zeros((len(ids), m), dtype=torch.long)
            for i, s in enumerate(ids):
                input_ids[i, : len(s)] = torch.tensor(s)
                mask[i, : len(s)] = 1
            out = model(input_ids.to(device), attention_mask=mask.to(device),
                        output_hidden_states=True)
            h = out.hidden_states[layer]
            last = torch.tensor(lens, device=device) - 1
            acts[b0 : b0 + len(ids)] = (
                h[torch.arange(len(ids), device=device), last]
                .float().cpu().numpy()
            )
    writer.write_shard(acts)
    manifest = writer.close()
    np.save(out_dir / "labels.npy", labels)
    manifest["task"] = task
    manifest["split"] = split
    manifest["n_examples"] = len(texts)
    return manifest


def load_probe_data(store_dir: str | Path):
    store = Store(store_dir)
    x = np.concatenate([xs for xs, _ in store.iter_shards()])
    y = np.load(Path(store_dir) / "labels.npy")
    assert x.shape[0] == y.shape[0]
    return x, y


def train_probe(x: np.ndarray, y: np.ndarray, n_classes: int,
                mean: np.ndarray, std: np.ndarray,
                max_iter: int = 300) -> np.ndarray:
    """Multinomial logistic probe via full-batch LBFGS, zero-init.

    Convex + deterministic: the returned [d+1, C] weights (bias last row)
    are the unique optimum, so cross-run cosine is exactly 1 by design.
    Features are standardized with the provided (train-split) stats.
    """
    xt = torch.from_numpy((x - mean) / std)
    yt = torch.from_numpy(y)
    w = torch.zeros(x.shape[1] + 1, n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=max_iter, tolerance_grad=1e-9,
                            tolerance_change=1e-11, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = xt @ w[:-1] + w[-1]
        loss = torch.nn.functional.cross_entropy(logits, yt)
        loss = loss + 1e-4 * w[:-1].pow(2).sum()  # small ridge, fixed
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()


def eval_probe(w: np.ndarray, x: np.ndarray, y: np.ndarray,
               mean: np.ndarray, std: np.ndarray) -> float:
    logits = ((x - mean) / std) @ w[:-1] + w[-1]
    return float((logits.argmax(axis=1) == y).mean())


def direction_cosine(w_a: np.ndarray, std_a: np.ndarray,
                     w_b: np.ndarray, std_b: np.ndarray) -> float:
    """Cosine between probe directions in RAW activation space (weights are
    learned in each store's standardized space, so de-standardize first),
    averaged over classes. For binary tasks this is the cosine of the
    separating normal."""
    a = w_a[:-1] / std_a[:, None]
    b = w_b[:-1] / std_b[:, None]
    cos = np.einsum("dc,dc->c", a, b) / (
        np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    ).clip(min=1e-12)
    return float(cos.mean())


def run_probe_comparison(task_dir: str | Path, codec: str,
                         n_classes: int) -> dict:
    """Train probes on fp32 and codec stores (identical examples), evaluate
    BOTH on clean fp32 validation activations, compare."""
    task_dir = Path(task_dir)
    x_ref, y_tr = load_probe_data(task_dir / "master-train")
    x_val, y_val = load_probe_data(task_dir / "master-val")
    x_cod, y_cod = load_probe_data(task_dir / f"{codec}-train")
    assert np.array_equal(y_tr, y_cod)

    # standardization stats always from the store the probe trains on
    stats = {}
    for name, x in (("ref", x_ref), ("cod", x_cod)):
        stats[name] = (x.mean(axis=0), x.std(axis=0) + 1e-6)
    w_ref = train_probe(x_ref, y_tr, n_classes, *stats["ref"])
    w_cod = train_probe(x_cod, y_tr, n_classes, *stats["cod"])
    return {
        "codec": codec,
        "acc_fp32": eval_probe(w_ref, x_val, y_val, *stats["ref"]),
        "acc_codec": eval_probe(w_cod, x_val, y_val, *stats["cod"]),
        "direction_cos": direction_cosine(w_ref, stats["ref"][1],
                                          w_cod, stats["cod"][1]),
        "n_train": int(x_ref.shape[0]),
        "n_val": int(x_val.shape[0]),
    }
