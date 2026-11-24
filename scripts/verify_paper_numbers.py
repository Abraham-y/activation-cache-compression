"""Check that every number in the paper's main text traces to an artifact.

The appendix tables are generated from JSONs, so they cannot drift. The
MAIN TEXT is typed by hand, and that asymmetry is exactly what produced
the one error an internal audit found (a 0.8-threshold range written 73.9 when
the artifact said 73.8 — the single hand-transcribed number in the main
text). This script closes that gap permanently.

It is deliberately NOT a list of expected sentences. Phrasing-based
checks rot the moment you rewrite a paragraph, which is when you most
need the check. Instead it inverts the question:

    for every number that appears in the main text, is there an
    artifact value it could have come from?

So it keeps working through an arbitrary rewrite. A number is accepted
if it equals some artifact-derived value at the precision it is written
to (as a fraction, a percentage, a ratio to a floor, a count, ...), or
if it is a declared constant with a stated reason. Anything else is
reported with its surrounding text so a human can look.

    PYTHONPATH=src python scripts/verify_paper_numbers.py
    PYTHONPATH=src python scripts/verify_paper_numbers.py --appendix
    PYTHONPATH=src python scripts/verify_paper_numbers.py --explain 51.2

KNOW ITS POWER BEFORE YOU TRUST IT. Run --selftest: it corrupts every
number in the paper by +1 in its last digit and reports what fraction it
catches. Measured 14.8% (33/223, pool 8835 values) on the current
build; re-run it rather than trusting this line, which goes stale every
time the artifact set grows. This is a LINT, not a proof — it
reliably catches a number with no plausible source at all, and it misses
most small transcription slips, because the admissible pool is thousands
of values and almost any plausible figure lands within tolerance of one
of them. A 0-unsourced result means "nothing is obviously invented", not
"every number is right". The only real guarantee comes from generating
main-text numbers rather than typing them.

Exit status is 1 if anything is unaccounted for, so it can gate a
commit or a submission.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Numbers that are definitional, structural, or bibliographic — not
# measurements. Each needs a reason; an unexplained entry here is just a
# way of hiding an unsourced number.
CONSTANTS: dict[float, str] = {
    0.9: "survival threshold (Duan banding)",
    15360: ("ClumpUlpCodec's hardcoded n_perturb (codecs.py): the clump "
            "arm really does move exactly 15,360 values, all inside one "
            "shard. The SPARSE rung is 15,335 (per-shard k summed), which "
            "is why the paper prints both and calls them matched to 0.2%"),
    15335: ("derived: SparseUlpCodec applies k=round(n_shard*d*1e-6) PER "
            "SHARD, so the corpus total is sum over the 77 master-train "
            "shards = 15,335, not the 15,360 a whole-corpus fraction gives"),
    1.12: ("derived in tab:steering's caption by the generator: worst "
           "x2-convention ratio for pca256 on ag_news over the recorded "
           "floor splits (2*0.20183/0.3601)"),
    0.00005: ("derived: max-min of rot-int8 frozen r across the three "
              "sites, rounded to 5 dp in tab:transfer's caption"),
    1.05: ("derived: pca256 ag_news steering ratio 0.5235 under the "
           "strictest like-for-like floor referent (x2), stated in "
           "tab:steering to bound the half-sample convention"),
    2000: ("derived, rounded: fp16 rel-L2 2.0336e-4 / noise7 1.0131e-7 = "
           "2007x, quoted as 2000x in the abstract and sec:dose"),
    0.5: "damaged/degraded band edge (Duan banding)",
    0.8: "stability-threshold sensitivity arm",
    0.95: "stability-threshold sensitivity arm",
    0.7: "Hungarian cosine threshold (Paulo & Belrose)",
    0.1: "alive cutoff, percent form (firing >0.1%)",
    0.001: "alive cutoff, fraction form",
    2.1: "derived: 8.5 pp/decade x 0.25 decades, stated in text",
    3.4: ("derived: 13.4 pp/decade x 0.25 decades, stated in text. NOTE this "
          "value also occurs as an asymmetry ratio in sec:dose; a bare-value "
          "whitelist cannot tell them apart, so neither is really checked"),
    1.0: "ceiling / identity",
    1.00: "ceiling / identity",
    2.0: "bytes per token of text; 2x; 2-fold",
    3.0: "3x budget tier",
    4.0: "4 bits; 4x",
    5.0: "five decimals; five audits; 5x",
    8.0: "8 bits; 8 KB; 8x",
    9.0: "nine SAEs / ninefold",
    10.0: "10x budget tier / 10x compression",
    16.0: "16 bits; k=16; 16 epochs",
    11.0: "11x parameter jump",
    12.0: "twelve",
    32.0: "TopK k=32",
    64.0: "k=64",
    100.0: "percent",
    2024.0: "citation year",
    2025.0: "citation year",
    2026.0: "citation year",
    4000.0: "4,000x (Llama Scope, rounded from 4,096)",
    2.13: "torch version in the CUDA replication",
    1.0e-5: "scientific-notation mantissa in '3 x 10^-5'",
    3.0e-5: "frozen-metric resolution, stated in the text as 3 x 10^-5",
    450.0: ("peak disk, rounded operational note (reproduction appendix): "
            "`du -sh runs_m1/cache` read 433G on 8 Oct 2026 with the fresh "
            "store, plus SAEs and evals"),
    1.10: "pre-registered timing threshold for a material gain (modal_timing.py)",
    7950.0: ("pre-registered A3 impulse step (scripts/geom_dynamics.py "
             "docstring; a row of tab:dynamics, where it is a dict key, not "
             "a value, so the artifact pool cannot see it)"),
}

# Section/figure/table numbers and other LaTeX scaffolding to strip
# before scanning, so cross-references are not mistaken for claims.
_STRIP = [
    (re.compile(r"(?<!\\)%.*$", re.M), ""),            # comments
    (re.compile(r"\\(?:label|ref|cite[a-z]*|citep|citet)\{[^}]*\}"), " "),
    (re.compile(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}"), " "),
    (re.compile(r"\\(?:usepackage|documentclass)(\[[^\]]*\])?\{[^}]*\}"), " "),
    (re.compile(r"\\S(?=\\?ref)"), " "),
    (re.compile(r"\\[a-zA-Z]+"), " "),                 # remaining macros
    (re.compile(r"\{,\}"), ""),                        # 20{,}000 -> 20000
    (re.compile(r"[{}$~\\]"), " "),
]

NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])")


def main_text(include_appendix: bool) -> str:
    """Body text only. The bibliography (titles, years, arXiv ids) sits
    BETWEEN the main text and the appendix in this document, so it has to
    be cut out of the middle rather than truncated at."""
    t = (ROOT / "paper" / "main.tex").read_text()
    t = t.split(r"\begin{document}", 1)[1]
    body = re.split(r"\\bibliographystyle|\\begin\{thebibliography\}", t)[0]
    if not include_appendix:
        return body
    tail = t.split(r"\appendix", 1)
    if len(tail) == 1:
        return body
    # Appendix prose only. NOTE: \input{appendix_tables} is generated, so
    # its CELLS are safe by construction -- but its CAPTIONS are hand-written
    # inside scripts/appendix_tables.py and are checked by nothing. Two
    # caption defects survived full audits for exactly this reason. That
    # blind spot is now covered by caption_text() below, which scans the
    # generated table file the \\input pulls in.
    return body + "\n" + tail[1].split(r"\end{document}")[0]


def _braced(text: str, start: int) -> str:
    """Contents of the {...} beginning at `start`, brace-balanced."""
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
        i += 1
    return ""


def caption_text() -> list[tuple[str, str]]:
    """(label, caption) for every caption in the GENERATED table file.

    These are hand-written string literals in scripts/appendix_tables.py.
    Table CELLS are computed from artifacts and safe by construction;
    captions are prose and would otherwise be checked by nothing.

    POWER IS LOW: measured 10% on last-digit corruptions of caption
    numbers (against 14.8% for main text), because the artifact value set
    is dense enough that a corrupted value often coincides with some
    other real number. It reliably catches numbers with NO artifact
    behind them and malformed literals -- it caught two raw-comma bugs
    and two undeclared derived values on first run -- but it is a lint,
    not a guarantee. It also cannot see a caption that is numerically
    correct while asserting something the main text retracts, which is
    the failure mode that actually recurred. Read captions by hand."""
    out = []
    # BOTH files: the generated tables AND main.tex, whose figure captions
    # are equally hand-written. Scanning only the former let a figure
    # caption assert something the text withheld.
    for f in (ROOT / "paper" / "appendix_tables.tex",
              ROOT / "paper" / "main.tex"):
        if not f.exists():
            continue
        t = f.read_text()
        for m in re.finditer(r"\\caption\{", t):
            cap = _braced(t, m.end() - 1)
            lab = re.search(r"\\label\{([^}]*)\}", t[m.end():m.end() + 4000])
            out.append((lab.group(1) if lab else "?", cap))
    return out


def load_values() -> dict[float, list[str]]:
    """Every number the artifacts can justify, mapped to where it came from.

    Includes the derived forms the paper actually writes: percentages,
    ratios to a site floor, mean and half-range over seed pairs,
    complements (100 - x), and token-visit products.
    """
    vals: dict[float, list[str]] = {}
    totals = {"train_seconds": 0.0, "cache_seconds": 0.0, "n_saes": 0}

    def add(v, src):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if not np.isfinite(v):
            return
        vals.setdefault(round(v, 10), []).append(src)
        # the same quantity written as a percentage
        vals.setdefault(round(100.0 * v, 10), []).append(src + " [as %]")

    def j(p):
        p = ROOT / p
        return json.loads(p.read_text()) if p.exists() else None

    roots = ["runs_m1", "runs_m4_L8", "runs_m4_L16", "runs_cuda_modal/m1"]
    floors: dict[str, float] = {}

    for root in roots:
        ev = ROOT / root / "eval"
        if not ev.is_dir():
            continue
        # --- seed floors per BUDGET TIER / architecture tag, their spread,
        # and every ratio to the matching floor. The paper quotes ratios
        # against a site-AND-budget-local floor, so a single floor per root
        # is not enough (that is what 1.18x/1.47x are measured against).
        tags = {""}
        for p in ev.glob("pair_fp32_*__fp32_*.json"):
            m = re.match(r"pair_fp32(_[a-z0-9]+)?_s\d", p.name)
            if m and m.group(1):
                tags.add(m.group(1))
        for tag in sorted(tags):
            fl = [j(f"{root}/eval/pair_fp32{tag}_s{a}__fp32{tag}_s{b}.json")
                  for a, b in ((0, 1), (0, 2), (1, 2))]
            fl = [x["survived"] for x in fl if x]
            if not fl:
                continue
            f = float(np.mean(fl))
            floors[root + tag] = f
            name = f"{root}{tag or ''}"
            add(f, f"{name} seed floor (mean of {len(fl)} pairs)")
            add((max(fl) - min(fl)) / 2, f"{name} floor half-range")
            add(min(fl), f"{name} floor min")
            add(max(fl), f"{name} floor max")
            add(1 - f, f"{name} 1 - floor (features lost to a seed change)")
        for p in sorted(ev.glob("*.json")):
            # a post-paper follow-up output the paper does not cite (decoder
            # subspaces) would only add chance matches
            if p.name.startswith("geom_subspaces"):
                continue
            d = json.loads(p.read_text())
            src = f"{root}/eval/{p.name}"
            _walk(d, src, add)
            if d.get("mode") == "pair":
                m = re.match(r"pair_fp32(_[a-z0-9]+)?_s\d", p.name)
                tag = (m.group(1) or "") if m else ""
                fkey = root + tag
                if fkey in floors:
                    add(d["survived"] / floors[fkey],
                        f"{src} survived/floor[{tag or '1x'}]")
                add(1 - d["survived"], f"{src} 1 - survived")
        # --- store manifests: measured bytes/token, codec state, forward
        # throughput (the cache-build rate lives here and nowhere else) ---
        # A reclaimed cache (scripts/cleanup.sh) leaves its manifest in
        # cache_manifests_snapshot/ as <root>_cache_<store>_manifest.json;
        # read that when the live one is gone, or the pool silently shrinks.
        live = {m.parent.name: m for m in (ROOT / root / "cache").glob("*/manifest.json")}
        prefix = root.replace("/", "_") + "_cache_"
        for s in (ROOT / "cache_manifests_snapshot").glob(f"{prefix}*_manifest.json"):
            live.setdefault(s.name[len(prefix):-len("_manifest.json")], s)
        for name, m in sorted(live.items()):
            d = json.loads(m.read_text())
            src = f"{root}/{name}"
            add(d.get("payload_bytes_per_token"), f"{src} B/tok")
            add(d.get("disk_bytes_per_token"), f"{src} disk B/tok")
            if d.get("d"):
                add(d["payload_bytes_per_token"] * 8.0 / d["d"], f"{src} bits/dim")
            _walk(d.get("codec_state", {}), f"{src} codec_state", add)
            _walk(d.get("throughput", {}), f"{src} forward throughput", add)
            totals["cache_seconds"] += d.get("throughput", {}).get("seconds", 0.0)
            add(d.get("n_tokens"), f"{src} n_tokens")
            # per-shard token count: captions quote it as the measurement
            # scope for shard-level statistics (entropy, throughput)
            for sh in d.get("shards", [])[:1]:
                add(sh.get("n_tokens"), f"{src} shard n_tokens")
            if d.get("n_tokens"):
                add(d["n_tokens"] / 1e6, f"{src} n_tokens (M)")
                add(d["n_tokens"] * d.get("payload_bytes_per_token", 0) / 1e9,
                    f"{src} store size (GB)")
        # --- SAE configs, training summaries, and CONSUMPTION RATES ---
        rates = {}
        for s in sorted((ROOT / root / "saes").glob("*/summary.json")):
            if s.parent.name.startswith("geomdyn_"):
                continue   # post-paper geometry follow-up runs, not in the paper
            d = json.loads(s.read_text())
            src = f"{root}/saes/{s.parent.name}"
            _walk(d, src, add)
            totals["train_seconds"] += d.get("train_seconds", 0.0)
            totals["n_saes"] += 1
            cfg = d.get("cfg", {})
            if cfg.get("steps") and cfg.get("batch_size"):
                tv = cfg["steps"] * cfg["batch_size"]
                add(tv, f"{src} token-visits")
                add(tv / 1e6, f"{src} token-visits (M)")
                add(round(tv / 1e6), f"{src} token-visits (M, rounded)")
                if d.get("train_seconds"):
                    r = tv / d["train_seconds"]
                    add(r, f"{src} training tok/s")
                    if re.fullmatch(r"fp32_s\d", s.parent.name):
                        rates.setdefault(root, []).append(r)
            if cfg.get("steps"):
                add(cfg["steps"] / 1000.0, f"{src} steps (k)")
        # mean whole-run consumption over the fp32 seeds: this is the
        # number streaming_comparison.py derives and the paper quotes
        if rates.get(root):
            add(float(np.mean(rates[root])), f"{root} mean fp32 training tok/s")
    # cache-build time recovered for sites whose manifests were reclaimed
    for ct in sorted(ROOT.glob("runs_*/eval/cache_throughput.json")):
        d = json.loads(ct.read_text())
        for split in ("train", "eval"):
            totals["cache_seconds"] += d.get(split, {}).get("seconds", 0.0)

    # --- cross-condition derivations the paper quotes: stable-set sizes,
    # floor-to-floor ratios, and effect sizes in units of seed noise
    import itertools
    stab_counts, floor_vals = {}, {}
    for root in roots:
        ev = ROOT / root / "eval"
        if not ev.is_dir():
            continue
        for tag in sorted({m for m in ("", "_k16", "_k64", "_b3x", "_b10x",
                                       "_jumprelu")}):
            f1 = ev / f"pair_fp32{tag}_s0__fp32{tag}_s1.npz"
            f2 = ev / f"pair_fp32{tag}_s0__fp32{tag}_s2.npz"
            if not (f1.exists() and f2.exists()):
                continue
            z1, z2 = np.load(f1), np.load(f2)
            alive = z1["alive"]
            stable = ((np.nan_to_num(z1["best_corr"]) > 0.9)
                      & (np.nan_to_num(z2["best_corr"]) > 0.9) & alive)
            stab_counts[root + tag] = int(stable.sum())
            add(int(stable.sum()), f"{root}{tag} stable-set size")
            if root + tag in floors:
                floor_vals[root + tag] = floors[root + tag]
    # keys are "<root><tag>", e.g. "runs_m1_k16"; the root is everything
    # before the tag, NOT the first underscore-delimited token (which is
    # just "runs" and made this guard always true, leaking cross-site
    # ratios into the pool)
    def _root_of(key):
        for r in roots:
            if key == r or key.startswith(r + "_"):
                return r
        return key
    for (ka, va), (kb, vb) in itertools.permutations(floor_vals.items(), 2):
        if _root_of(ka) == _root_of(kb):
            add(va / vb, f"floor ratio {ka}/{kb}")
    # effect sizes: binomial sigma on the stable sets
    for key, n in stab_counts.items():
        for p_ in (0.9, 0.93, 0.95):
            add(np.sqrt(p_ * (1 - p_) / n) * 100, f"{key} binomial SE (pp)")
    # z for a difference of two stable-set proportions, quadrature-combined
    def _stab(key):
        r = _root_of(key)
        d = j(f"{r}/eval/stability_fp32{key[len(r):]}_s0_t0p9.json")
        return {x["cand"]: x["survived_stable"] for x in (d or {"rows": []})["rows"]}
    for (ka, na), (kb, nb) in itertools.combinations(stab_counts.items(), 2):
        sa, sb = _stab(ka), _stab(kb)
        for ca, va in sa.items():
            for cb, vb in sb.items():
                if ca.split("_s")[0] != cb.split("_s")[0]:
                    continue
                se = np.hypot(np.sqrt(va * (1 - va) / na),
                              np.sqrt(vb * (1 - vb) / nb))
                if se > 0:
                    add(abs(va - vb) / se, f"{ca} vs {cb} difference in sigma")

    # --- project-wide compute totals, as quoted in the Reproduction appendix
    cuda = [json.loads(f.read_text())["train_seconds"] for f in
            (ROOT / "runs_cuda_modal" / "m1" / "sae_summaries").glob("*/summary.json")]
    if cuda:
        add(sum(cuda) / 3600.0, "A100 SAE training hours (App. D runs)")
    add(totals["n_saes"], "total SAE runs (summary.json count)")
    add(totals["train_seconds"] / 3600.0, "total SAE training hours")
    add(totals["cache_seconds"] / 3600.0, "total activation-caching hours")
    add((totals["train_seconds"] + totals["cache_seconds"]) / 3600.0,
        "total training + caching hours")

    # theoretical byte sizes, so the check survives a cache cleanup
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from acc.codecs import theoretical_bytes_per_token
        for d_model in (768, 2048):
            for c in ("fp32", "fp16", "bf16", "int8", "int8-o8", "int6",
                      "int6-o8", "int4", "int4-o16", "rot-int8", "rot-int6",
                      "rot-int4", "pca256", "rp256"):
                b = theoretical_bytes_per_token(c, d_model)
                add(b, f"{c} B/tok at d={d_model} (formula)")
                add(b * 8.0 / d_model, f"{c} bits/dim at d={d_model}")
                add(4 * d_model / b, f"{c} compression vs fp32 at d={d_model}")
                add(2 * d_model / b, f"{c} compression vs fp16 at d={d_model}")
    except Exception as e:  # pragma: no cover
        print(f"note: byte formulas unavailable ({e})", file=sys.stderr)
    return vals


# Distributional summaries (deciles, histograms) densely tile [0,100].
# Admitting them makes the checker accept almost any plausible value —
# measured at ~89% of the 1-dp percentage grid before this exclusion.
# They are never quoted as claims, so they are not admissible evidence.
_NOT_EVIDENCE = ("deciles", "histogram", "quantiles", "corr_deciles",
                 "positions")   # random probe coordinates (geom_dynamics hooks)


def _walk(obj, src, add, key=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(bad in str(k) for bad in _NOT_EVIDENCE):
                continue
            _walk(v, src, add, f"{key}.{k}" if key else str(k))
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, src, add, key)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        add(obj, f"{src}:{key}")


def match(text_val: float, decimals: int, vals: dict) -> str | None:
    """Is `text_val`, written to `decimals` places, some artifact value?"""
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for v, srcs in vals.items():
        if abs(v - text_val) <= tol:
            return srcs[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appendix", action="store_true",
                    help="also scan appendix prose (tables are generated)")
    ap.add_argument("--explain", type=float, default=None,
                    help="show every artifact source for one value")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="measure this checker's own power: corrupt every "
                         "paper number by +1 in its last digit and report "
                         "what fraction is caught")
    args = ap.parse_args()

    vals = load_values()
    if args.explain is not None:
        hits = [(v, s) for v, s in vals.items()
                if abs(v - args.explain) <= 0.05]
        if not hits:
            print(f"no artifact value near {args.explain}")
            return 1
        for v, srcs in sorted(hits):
            print(f"{v:g}")
            for s in sorted(set(srcs))[:12]:
                print(f"    {s}")
        return 0

    if args.selftest:
        raw = main_text(True)
        stripped = raw
        for pat, rep in _STRIP:
            stripped = pat.sub(rep, stripped)
        seen, caught, total = set(), 0, 0
        for m in NUM_RE.finditer(stripped):
            s = m.group(1)
            if s in seen:
                continue
            seen.add(s)
            dec = len(s.split(".")[1]) if "." in s else 0
            bad = float(s) + 10 ** (-dec)          # +1 in the last digit
            if any(abs(bad - c) < 1e-12 for c in CONSTANTS):
                continue
            total += 1
            if match(bad, dec, vals) is None:
                caught += 1
        print(f"SELFTEST: last-digit corruption caught {caught}/{total} "
              f"= {100*caught/max(total,1):.1f}%  (pool = {len(vals)} values)")
        return 0

    raw = main_text(args.appendix)
    stripped = raw
    for pat, rep in _STRIP:
        stripped = pat.sub(rep, stripped)

    seen, unsourced = {}, []
    for m in NUM_RE.finditer(stripped):
        s = m.group(1)
        v = float(s)
        decimals = len(s.split(".")[1]) if "." in s else 0
        if s in seen:
            continue
        ctx = " ".join(stripped[max(0, m.start() - 70):m.end() + 40].split())
        src = match(v, decimals, vals)
        if src is None and not any(abs(v - c) < 1e-12 for c in CONSTANTS):
            unsourced.append((s, ctx))
        seen[s] = src

    n_ok = sum(1 for v in seen.values() if v is not None)
    n_const = len(seen) - n_ok - len(unsourced)
    print(f"main text{' + appendix' if args.appendix else ''}: "
          f"{len(seen)} distinct numbers — {n_ok} matched to artifacts, "
          f"{n_const} declared constants, {len(unsourced)} UNSOURCED")
    if not args.quiet and unsourced:
        print("\nUnsourced (check each: either wrong, or a constant that "
              "belongs in CONSTANTS with a reason):")
        for s, ctx in unsourced:
            print(f"\n  {s}\n      ...{ctx}...")

    # --- captions of the generated table file (the historical blind spot)
    cap_bad = []
    for label, cap in caption_text():
        c = cap
        for pat, rep in _STRIP:
            c = pat.sub(rep, c)
        for m in NUM_RE.finditer(c):
            cs = m.group(1)
            v = float(cs)
            dec = len(cs.split(".")[1]) if "." in cs else 0
            if match(v, dec, vals) is None and not any(
                    abs(v - k) < 1e-12 for k in CONSTANTS):
                ctx = " ".join(c[max(0, m.start() - 60):m.end() + 30].split())
                cap_bad.append((label, cs, ctx))
    n_caps = len(caption_text())
    print(f"captions ({n_caps}, tables + figures): "
          f"{len(cap_bad)} UNSOURCED")
    if not args.quiet and cap_bad:
        print("\nUnsourced caption numbers:")
        for label, cs, ctx in cap_bad:
            print(f"\n  [{label}] {cs}\n      ...{ctx}...")
    return 1 if (unsourced or cap_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
