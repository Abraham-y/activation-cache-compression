"""Standing invariants over the released eval artifacts.

These are free, exact, and catch a class of bug that unit tests on the
codecs cannot: silent numerical corruption in the eval path. The FVU one
below caught a real bug — a float32 .sum() over 8192x768 on MPS is
nondeterministic across processes and was observed 12.5% wrong, which
corrupted fvu_ref in two published eval files while leaving best_corr
(accumulated in float64 numpy) untouched.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
EVAL = ROOT / "runs_m1" / "eval"


def _pairs():
    return sorted(EVAL.glob("pair_fp32_s0__*.json"))


@pytest.mark.skipif(not EVAL.exists(), reason="eval artifacts not present")
def test_fvu_ref_is_constant_across_pairs():
    """Same reference SAE and same eval tokens => identical fvu_ref.

    Any spread here is numerical corruption, not a property of a codec.
    """
    vals = {}
    for f in _pairs():
        d = json.loads(f.read_text())
        if "fvu_ref" in d:
            vals[f.name] = d["fvu_ref"]
    assert vals, "no pair files carry fvu_ref"
    lo, hi = min(vals.values()), max(vals.values())
    spread = (hi - lo) / lo
    # tolerance admits float32-vs-float64 accumulation (~1e-9 relative) but
    # not the corruption this guards against, which was 12.5%
    assert spread < 1e-6, (
        f"fvu_ref spans {lo:.9f}-{hi:.9f} ({100*spread:.2f}% relative) "
        "across pair files, which is impossible if the reference SAE and "
        "eval set are fixed. Suspect a nondeterministic float32 reduction. "
        f"Outliers: {sorted(vals.items(), key=lambda kv: kv[1])[:2]}")


@pytest.mark.skipif(not EVAL.exists(), reason="eval artifacts not present")
def test_alive_population_is_shared_across_pairs():
    """Every pair scores the same reference features, so n_alive is fixed."""
    seen = {}
    for f in _pairs():
        d = json.loads(f.read_text())
        if "n_alive" in d:
            seen.setdefault(d["n_alive"], []).append(f.name)
    assert len(seen) == 1, f"n_alive differs across pair files: {seen}"
