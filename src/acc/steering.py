"""A non-SAE, non-probe instrument: causal steering vectors.

Why this exists. Every fidelity measure in this paper so far reads the
cache through an SAE, or through a linear probe that is itself a
classifier fit to the same activations. A audit is right to ask whether
the protocol says anything about interpretability artifacts that are
neither. This module supplies one that is causal rather than
correlational: a steering vector extracted from the cache, injected back
into the model's forward pass, and scored by what it does to the output.

The instrument. For a labelled task, the concept direction is the
difference of class means over cached activations at the probed layer --
the standard difference-in-means steering vector. It is DETERMINISTIC
given the data, which matters for the protocol: there is no training seed
to vary, so the same-process control cannot be a seed floor. The
analogous control is the estimator's own nuisance variation, which here
is the data sample. We therefore split the fp32 activations into two
disjoint halves, extract a direction from each, and use the agreement
between those two fp32 directions as the floor -- exactly the role three
seeds play for an SAE.

That substitution is the point. The protocol is not "compare SAEs across
seeds"; it is "compare a treatment against the same estimator's own
nuisance variation, measured on the same data". For a seed-unstable
trainer that means reseeding; for a deterministic estimator it means
resampling. A codec is acceptable on this instrument if it perturbs the
direction, and the direction's causal effect, less than a data resample
does.

CAUTION on the floor. This floor is a data-resample estimate and it is
strongly n-dependent: on sst2 it reads 0.35 at n=200, 0.83 at n=872 (the
validation split), 0.985 at n=67k (train), still rising. A small split
gives a LOOSE floor and everything passes, which is a statement about the
estimator's noise, not about the codecs. Always use the largest split
available and report n beside the floor. The first run of this arm used
the validation split and had to be redone.

Two readouts, reported separately because they can disagree:
  1. GEOMETRIC -- cosine between the fp32 direction and the codec
     direction. Cheap, and the same quantity the probe arm reports, so it
     is the bridge between the two.
  2. CAUSAL -- inject each direction into the residual stream at the
     probed layer during a forward pass and measure the induced shift in
     the model's own output distribution. This is what no other arm in
     the paper measures: whether the codec preserves what the vector
     DOES, not merely where it points. A direction can survive on cosine
     and still steer differently, and vice versa.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .probes import load_probe_data
from .utils import write_json


def direction(x: np.ndarray, y: np.ndarray, cls_a: int = 0,
              cls_b: int = 1) -> np.ndarray:
    """Difference-in-means steering vector, unit-normalised."""
    a = x[y == cls_a].mean(axis=0)
    b = x[y == cls_b].mean(axis=0)
    v = b - a
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def split_halves(x: np.ndarray, y: np.ndarray, seed: int = 0):
    """Disjoint, class-stratified halves — the data-resample control."""
    rng = np.random.default_rng(seed)
    ia, ib = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        h = len(idx) // 2
        ia.append(idx[:h])
        ib.append(idx[h:2 * h])
    return np.concatenate(ia), np.concatenate(ib)


def causal_effect(vec: np.ndarray, prompts: list[str], layer: int,
                  strength: float, model, tok, device: str) -> np.ndarray:
    """Mean shift in next-token log-probs when `vec` is added at `layer`.

    Returns the per-prompt log-prob delta vector over the vocabulary,
    averaged across prompts. Two directions that steer alike produce
    similar deltas; two that steer differently do not, even if their
    cosine is high.
    """
    v = torch.from_numpy(vec.astype(np.float32)).to(device) * strength

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] = h[:, -1, :] + v
        return (h,) + out[1:] if isinstance(out, tuple) else h

    deltas = []
    block = model.transformer.h[layer - 1]
    with torch.inference_mode():
        for p in prompts:
            ids = tok(p, return_tensors="pt").to(device)
            base = torch.log_softmax(model(**ids).logits[0, -1], dim=-1)
            hd = block.register_forward_hook(hook)
            try:
                stee = torch.log_softmax(model(**ids).logits[0, -1], dim=-1)
            finally:
                hd.remove()
            deltas.append((stee - base).cpu().numpy())
    return np.mean(deltas, axis=0)


def run_steering_comparison(task_dir_fp32: str | Path,
                            task_dir_codec: str | Path,
                            codec: str, layer: int, out: str | Path,
                            prompts: list[str] | None = None,
                            strength: float = 5.0,
                            device: str = "cpu") -> dict:
    """Codec direction vs fp32 direction, against the data-resample floor."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    xf, yf = load_probe_data(task_dir_fp32)
    xc, yc = load_probe_data(task_dir_codec)
    if not np.array_equal(yf, yc):
        raise ValueError("label mismatch: codec cache is not the same examples")

    v_fp32 = direction(xf, yf)
    v_codec = direction(xc, yc)
    ia, ib = split_halves(xf, yf)
    v_a, v_b = direction(xf[ia], yf[ia]), direction(xf[ib], yf[ib])

    cos = lambda p, q: float(np.dot(p, q))
    # direction() contrasts classes 0 and 1 only, so on a >2-class task the
    # effective n is the size of those two classes, not the whole split.
    n_eff = int(((yf == 0) | (yf == 1)).sum())
    res = {
        "codec": codec, "layer": layer, "n": int(len(yf)),
        "n_effective": n_eff, "classes_contrasted": [0, 1],
        "note": ("Difference-in-means steering vector. The floor is two "
                 "disjoint halves of the SAME fp32 data — the deterministic "
                 "estimator's own nuisance variation, standing in for the "
                 "seed floor used with SAEs. Only classes 0 and 1 are "
                 "contrasted, so n_effective is what the direction and its "
                 "floor are estimated from."),
        "geometric": {
            "cos_fp32_vs_codec": cos(v_fp32, v_codec),
            "cos_floor_half_vs_half": cos(v_a, v_b),
        },
    }
    if prompts:
        tok = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to(device)
        d_fp32 = causal_effect(v_fp32, prompts, layer, strength, model, tok, device)
        d_codec = causal_effect(v_codec, prompts, layer, strength, model, tok, device)
        d_a = causal_effect(v_a, prompts, layer, strength, model, tok, device)
        d_b = causal_effect(v_b, prompts, layer, strength, model, tok, device)
        cs = lambda p, q: float(np.dot(p, q) /
                                (np.linalg.norm(p) * np.linalg.norm(q) + 1e-30))
        res["causal"] = {
            "strength": strength, "n_prompts": len(prompts),
            "cos_effect_fp32_vs_codec": cs(d_fp32, d_codec),
            "cos_effect_floor_half_vs_half": cs(d_a, d_b),
            "l2_effect_fp32_vs_codec": float(np.linalg.norm(d_fp32 - d_codec)),
            "l2_effect_floor_half_vs_half": float(np.linalg.norm(d_a - d_b)),
        }
        g = res["causal"]
        res["causal"]["ratio_to_floor_l2"] = (
            g["l2_effect_fp32_vs_codec"] / g["l2_effect_floor_half_vs_half"]
            if g["l2_effect_floor_half_vs_half"] > 0 else None)
    write_json(Path(out), res)
    return res
