"""Generate paper/appendix_tables.tex directly from eval JSONs.

Every appendix number is emitted by this script from its source artifact
— no hand transcription (an internal audit's lesson). Rerun after any eval change.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acc.utils import read_json  # noqa: E402
from acc.codecs import theoretical_bytes_per_token  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = []


def j(path):
    p = ROOT / path
    return read_json(p) if p.exists() else None


def bytes_per_token(root, codec):
    """Measured bytes/token from the store manifest; if the cache has been
    reclaimed (scripts/cleanup.sh) fall back to the snapshot, then to the
    exact byte formula with d from a checkpoint. Measured == theoretical
    for every codec here (asserted in tests/test_codecs.py), so no
    regeneration silently degrades to dashes."""
    m = j(f"{root}/cache/{codec}-train/manifest.json")
    if m is None:
        m = j(f"cache_manifests_snapshot/{root}_cache_{codec}-train_manifest.json")
    if m is not None:
        return m["payload_bytes_per_token"]
    from acc.codecs import theoretical_bytes_per_token
    d = _d_in(root)
    return theoretical_bytes_per_token(codec, d) if d else None


def _d_in(root):
    """d_model for a run. Prefer summary.json (a few hundred bytes) over the
    checkpoint (hundreds of MB) so the released artifact does not have to
    ship weights just to label an axis."""
    s = j(f"{root}/saes/fp32_s0/summary.json")
    if s and s.get("cfg", {}).get("d_in"):
        return s["cfg"]["d_in"]
    import torch
    ck = ROOT / root / "saes" / "fp32_s0" / "sae.pt"
    if not ck.exists():
        return None
    return torch.load(ck, map_location="cpu", weights_only=True)["cfg"]["d_in"]


def pct(x, dp=1):
    return f"{100*x:.{dp}f}" if x is not None else "---"


def seeds_stat(root, codec, seed_cands):
    """mean +- range/2 over available same-seed pairs; None if <2 seeds."""
    vals = []
    for s, cand in seed_cands:
        r = j(f"{root}/eval/pair_fp32_s{s}__{cand}.json")
        if r:
            vals.append(r["survived"])
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], None
    return float(np.mean(vals)), float((max(vals) - min(vals)) / 2)


# Short captions (1 Sept): one to three sentences, what is
# shown plus the one caveat that matters. Keyed by label; any label not
# listed keeps its generated caption. Numbers here are lint-checked.
SHORT = {
 "tab:m1full": r"Full GPT-2 codec panel. $\pm$ rows are three-seed means ($\pm$ half the range), bare cells seed 0, so fp16 reads $59.3 \pm 1.1$ here and $60.6$ (seed 0) in the text. survived: best match above $r = 0.9$ against fp32 at the same seed; FVU: fraction of variance unexplained; mean $r$: the retrained SAE's best-match correlation, averaged over alive features (not the frozen reading); stable: survival among the $774$ features reproduced in both other fp32 seeds; Hung.\ cos: mean matched encoder and decoder cosine over all latents, dead ones included; frozen: the thresholded reading through the frozen fp32 SAE. FVU is flat across every viable codec ($0.1355$ to $0.1361$) while seed-0 agreement falls from $60.6$ to $37.6\%$. The floor row's FVU is the fp32 reference's own, and its Hungarian cosine averages the two seed-0 pairs.",
 "tab:m4l8": r"Pythia-1.4B layer 8, same construction as Table~\ref{tab:m1full}. Codec runs are seed 0 only, so gaps under the $2$\,pp tie band are unresolved; the bracketed intervals resample features against the seed-0/seed-1 floor pair, not seeds. fp16 is bit-lossless under the bf16 master, so its row is a same-seed ceiling.",
 "tab:transfer": r"Frozen-SAE mean $r$ beside retrained agreement at three sites. Within a site the frozen SAE ranks the codecs as retraining does; across sites the same reading carries no level (rot-int8 reads within $0.00005$ everywhere and retrains to $2.09$--$2.73\times$ the local floor). Codec rows are seed 0; floors are three-pair means.",
 "tab:jumprelu": r"JumpReLU panel (GPT-2, calibrated $\lambda=8.68$). The viable codecs land within a few points of the floor (fp16 $26.4$, rot-int4 $26.1$, rot-int8 $25.3$ against $23.7$ to $24.2$), too close together for a graded verdict. mean $r$: the retrained best-match correlation averaged over alive features. The fp32 and viable-codec runs end with about a third of latents dead (the na\"ive int4 run about a quarter), and the na\"ive int4 run converged to $L_0 = 38.2$ against fp32's 18.6, so its row is shown but excluded from any verdict as confounded.",
 "tab:budget": r"Budget scaling on a fixed stable set ($N = 774$ features at every tier, binomial SE about $1.1$\,pp). fp16 is flat, the rotated codecs improve. fp16's count is $741$ of $774$ at all three tiers by coincidence: the three pairs are distinct runs whose all-alive agreement is $60.6$, $49.2$ and $50.6\%$. Budget was scaled by re-reading one 20M-token cache (up to about 16 passes over the same tokens), so this isolates optimization from data diversity but not from convergence on a fixed corpus; at $3\times$, fresh text gives the same floor (Table~\ref{tab:fresh}).",
 "tab:probes": r"Linear probes: accuracy change against the fp32-trained probe, and learned-direction cosine, each scored against a seven-split resample floor ($0.008$/$0.002$ accuracy and $0.80$/$0.77$ cosine on sst2/ag\_news, mean over splits, with cosine minima $0.788$/$0.759$; \texttt{probe\_floor.json}). Accuracy resolution is one validation example, so read the deltas as counts. PCA-256 keeps accuracy on sst2 but not on ag\_news (its drop there is outside the floor) while its directions collapse to about $0.5$ on both; int4-o16 fails the sst2 direction floor ($0.778$ against a floor minimum of $0.788$) on a test the retraining test passes. The two other codecs with probe runs, int8 and rot-int6, clear both floors on both tasks ($0.991$/$0.995$ and $0.998$/$0.998$ cosine) and are omitted for width.",
 "tab:firingrate": r"The alive cut, swept. Tightening it raises agreement and the floor together and lowers rot-int4's ratio monotonically ($1.63$ to $1.13\times$ on GPT-2); our $0.001$ cut is the loosest of the five. rot-int4 only, so this bounds a ratio, not an ordering; the floor here averages the two seed-0 pairs at each site ($23.1$ against the three-pair $23.0$ on GPT-2).",
 "tab:sparsity": r"Sparsity moves the floor without touching a byte: over a $1.95\times$ swing in the floor, rot-int8's ratio spreads $1.72$--$3.23\times$ while its survival among seed-stable features spreads only $92.8$--$97.1\%$.",
 "tab:forward": r"A treatment that is not a storage codec: computing the cache with a bf16 forward (stored in fp32) beside the storage codecs. Over three seeds the bf16 forward retains $49.7$--$51.2\%$ against rot-int8's $51.6$--$53.8$, disjoint but with $n=3$ (a sign test cannot go below $p=0.25$; within-seed deltas $-2.7$, $-1.8$, $-3.0$\,pp). At $3\times$ budget the comparison reads $44.4$ against $45.3\%$ (one seed each), inside the tie band. It runs $2.21\times$ faster.",
 "tab:entropy": r"Lossless zstd-3 over the quantized payloads, measured on one shard per codec. Rotated codecs compress only $1.04$--$1.26\times$ further; na\"ive scaling leaves headroom that zstd recovers. Standardization without the rotation lands with the rotated codec ($303.8$ against $307.5$\,B/token).",
 "tab:throughput": r"Serial loader throughput per store (prefetch $= 0$, warm page cache, one Apple M5). The $1\times$ training run consumes $10{,}017$ tokens/s, so even the slowest store (int4-o16, 55k tokens/s) outruns training; the standardize-and-rotate decode path costs real time (rot-int8 reads at $64\%$ of int8's rate). One-machine numbers.{PAIRED}",
 "tab:threshold": r"Ratio to floor as the counting threshold moves. Levels change with the threshold; the ordering and the below-floor verdict for na\"ive int4 do not.",
 "tab:dose": r"The noise levels and sparse changes (seed 0; pp/decade is the change in agreement per decade of relative $L_2$ to the next row): isotropic noise at an exact relative $L_2$ and one-ulp changes to a fixed count of values, all stored in fp32 on both sides. Everything from $n=77$ to $15{,}335$ is flat; the one-value change also differs in when it reaches training (Appendix~\ref{app:margins}). The fp16-scale noise and the millionth-of-values change were re-run twice with fresh randomness ($60.6$/$60.4$/$60.8$ and $70.5$/$69.7$/$71.6\%$); the table prints the original runs.",
 "tab:dosestrat": r"The noise levels and codecs on the seed-stable stratum ($n = 774$) beside all alive features. The response is flatter still on the stable stratum ($0.35$\,pp per decade from noise $10^{-7}$ to $10^{-4}$, against $2.6$ from fp16 to rot-int4), and the ratio of the two rates depends on the window, so we quote no single ratio.",
 "tab:dosemargin": r"How far the two smallest perturbations sit above the seed floor at each counting threshold. The margin is positive everywhere ($11$--$43$\,pp dense, $12$--$47$ sparse) but its size is threshold-dependent.",
 "tab:steering": r"The causal test: a difference-in-means steering vector injected into the residual stream, scored against the variation between two disjoint halves of the same data (the half-versus-half floor runs about twice a full-data comparison, so it is conservative only for a fail; the like-for-like referent is half the floor). Na\"ive int4 fails at least $399$ of $400$ resample splits on both tasks; PCA-256 passes on average but fails $51$ of $400$ ag\_news splits against the like-for-like referent; task-matched prompts and a strength sweep leave every verdict unchanged.",
 "tab:consensus": r"Is codec damage a shared bias or seed-like noise? On the fp32 seed-stable set ($n=774$), losses are enriched above independence for every codec, so a shared component exists, but under rot-int8 only $0.13\%$ of seed-stable features are lost in all three seeds against $24.2\%$ under na\"ive int4. Recomputed from released pair files; no training.",
 "tab:proxyrank": r"Spearman rank agreement of three cheap readings with retrained agreement over every released condition per site. The 15 GPT-2 conditions are the 13 storage codecs of Table~\ref{tab:m1full} plus the bf16 forward and the std-int4 ablation. None of the three needs a retrained SAE, and none ranks better than the others.",
 "tab:fragility": r"Predicting per-feature survival from fp32 statistics. CV AUC: a logistic model on all three, 5-fold cross-validated; the other columns are single-statistic AUCs. Mean activation is the strongest single predictor, unlike the peak activation reported for weight quantization.",
}


PAIRED = {}


def emit(caption, label, header, rows, note=""):
    caption = SHORT.get(label, caption)
    caption = caption.replace("{PAIRED}", PAIRED.get(label, ""))
    cols = "l" + "r" * (len(header) - 1)
    # wide tables overflow the text block at \small; step down rather than
    # let an overfull box through (the project standard is zero)
    size = r"\footnotesize" if len(header) >= 8 else r"\small"
    OUT.append(r"\begin{table}[htbp]")
    OUT.append(rf"\caption{{{caption}}}")
    OUT.append(rf"\label{{{label}}}")
    OUT.append(rf"\centering{size}")
    OUT.append(rf"\begin{{tabular}}{{{cols}}}")
    OUT.append(r"\toprule")
    OUT.append(" & ".join(header) + r" \\")
    OUT.append(r"\midrule")
    for row in rows:
        cells = [str(c) for c in row]
        if cells[0].startswith(r"\midrule"):
            # a section-header first row would otherwise double the rule
            # emit() already writes after the column headers
            if OUT[-1] != r"\midrule":
                OUT.append(r"\midrule")
            cells[0] = cells[0].replace(r"\midrule", "").strip()
        OUT.append(" & ".join(cells) + r" \\")
    OUT.append(r"\bottomrule")
    OUT.append(r"\end{tabular}")
    if note:
        OUT.append(rf"\par\smallskip\footnotesize {note}")
    OUT.append(r"\end{table}")
    OUT.append("")


def main_m1_table():
    codecs = ["fp16", "bf16", "rot-int8", "int8", "int8-o8", "rot-int6",
              "int6", "int6-o8", "rot-int4", "int4-o16", "int4",
              "pca256", "rp256"]
    seed_cands = {
        "fp16": [(0, "fp16_s0"), (1, "fp16_s1"), (2, "fp16_s2")],
        "rot-int8": [(0, "rot-int8_s0"), (1, "rot-int8_k32_s1"),
                     (2, "rot-int8_k32_s2")],
        "rot-int6": [(0, "rot-int6_s0"), (1, "rot-int6_s1"), (2, "rot-int6_s2")],
        "int6-o8": [(0, "int6-o8_s0"), (1, "int6-o8_s1"), (2, "int6-o8_s2")],
        "rot-int4": [(0, "rot-int4_s0"), (1, "rot-int4_s1"), (2, "rot-int4_s2")],
        "int4": [(0, "int4_s0"), (1, "int4_s1"), (2, "int4_s2")],
    }
    strat = {r["cand"]: r for r in
             j("runs_m1/eval/stability_fp32_s0_t0p9.json")["rows"]}
    rows = []
    for c in codecs:
        bpt_v = bytes_per_token("runs_m1", c)
        bpt = f"{bpt_v:.0f}" if bpt_v is not None else "---"
        cands = seed_cands.get(c, [(0, f"{c}_s0")])
        mean, half = seeds_stat("runs_m1", c, cands)
        surv = (f"${pct(mean)} \\pm {100*half:.1f}$" if half is not None
                else pct(mean))
        pair = j(f"runs_m1/eval/pair_fp32_s0__{c}_s0.json")
        hung = j(f"runs_m1/eval/hungarian_fp32_s0__{c}_s0.json")
        froz = j(f"runs_m1/eval/frozen_{c}__fp32_s0.json")
        st = strat.get(f"{c}_s0", {})
        # 'bf16' here is the STORAGE codec; the bf16 FORWARD arm is a
        # different treatment (tab:forward). Unlabelled they read alike.
        rows.append([
            "bf16 storage" if c == "bf16" else c, bpt, surv,
            f"{pair['fvu_cand']:.4f}" if pair else "---",
            f"{pair['mean_corr']:.3f}" if pair else "---",
            pct(st.get("survived_stable"), 1),
            f"{hung['mean_cos']:.3f}" if hung else "---",
            pct(froz["survived"], 1) if froz else "---",
        ])
    fl = [j(f"runs_m1/eval/pair_fp32_s{a}__fp32_s{b}.json")["survived"]
          for a, b in ((0, 1), (0, 2), (1, 2))]
    fl_r = [j(f"runs_m1/eval/pair_fp32_s{a}__fp32_s{b}.json")["mean_corr"]
            for a, b in ((0, 1), (0, 2), (1, 2))]
    fl_h = [j(f"runs_m1/eval/hungarian_fp32_s0__fp32_s{s_}.json")["mean_cos"]
            for s_ in (1, 2)]
    fvu32 = j("runs_m1/eval/pair_fp32_s0__fp16_s0.json")["fvu_ref"]
    rows.append([r"\midrule seed floor", "---",
                 f"${pct(np.mean(fl))} \\pm {100*(max(fl)-min(fl))/2:.1f}$",
                 f"{fvu32:.4f}",
                 f"{np.mean(fl_r):.3f}", "(def.)", f"{np.mean(fl_h):.3f}", "---"])
    emit("Full GPT-2 codec panel. Conventions as in \\S\\ref{sec:protocol}: "
         "$\\pm$ rows are three-seed means (mean $\\pm$ range/2), "
         "bare cells seed 0, so the same codec can show both (fp16: "
         "$59.3 \\pm 1.1$ and $60.6$); survived $=$ best-match $r>0.9$ vs "
         "fp32 at the same seed. Stable $=$ agreement among features "
         "reproduced in both other fp32 seeds ($n = 774$, 15.9\\% of alive, "
         "binomial SE $\\sim$1.1\\,pp, threshold 0.9); Hung.\\ $=$ mean "
         "matched encoder$+$decoder cosine; frozen $=$ thresholded reading "
         "through the frozen fp32 SAE; FVU $=$ fraction of variance "
         "unexplained. FVU is flat across every viable codec ($0.1355$ "
         "untreated, on the floor row, through $0.1361$ at int4-o16) while "
         "agreement falls $60.6$ to $37.6\\%$; only the at-or-below-floor "
         "codecs move it: na\\\"ive int4 ($0.1968$), PCA-256 ($0.3640$), "
         "RP-256 ($0.8626$). Codec-trained SAEs reconstruct as well and "
         "find different features. Mean $L_0$ is $k=32$ by construction "
         "and not shown. Every row is a storage codec at an fp32 "
         "forward; the bf16 forward treatment is "
         "Table~\\ref{tab:forward}. Sources: pair/hungarian/frozen/"
         "stability JSONs in \\texttt{runs\\_m1/eval}.",
         "tab:m1full",
         ["codec", "B/tok", "survived \\%", "FVU", "mean $r$",
          "stable \\%", "Hung.\\ cos", "frozen \\%"],
         rows)


def m4_tables():
    for L, fkey in (("8", "runs_m4_L8"), ("16", "runs_m4_L16")):
        strat = {r["cand"]: r for r in
                 j(f"{fkey}/eval/stability_fp32_s0_t0p9.json")["rows"]}
        fl = [j(f"{fkey}/eval/pair_fp32_s{a}__fp32_s{b}.json")["survived"]
              for a, b in ((0, 1), (0, 2), (1, 2))]
        flm = float(np.mean(fl))
        rows = []
        for c in ["fp16", "rot-int8", "int8", "rot-int4", "int4"]:
            pair = j(f"{fkey}/eval/pair_fp32_s0__{c}_s0.json")
            if not pair:
                continue
            s = pair["survived"]
            st = strat.get(f"{c}_s0", {})
            note = " (bit-identical ceiling)" if c == "fp16" else ""
            ci = (j("runs_m1/eval/paired_feature_tests.json") or {}) \
                .get("ratio_cis", {}).get(f"{'pythia_l8' if L == '8' else 'pythia_l16'}/{c}")
            rat = (f"{s/flm:.2f} [{ci['ci95'][0]:.2f}, {ci['ci95'][1]:.2f}]"
                   if ci else f"{s/flm:.2f}$\\times$")
            rows.append([c + note, pct(s), rat,
                         pct(st.get("survived_stable"), 1)])
        rows.append([r"\midrule seed floor",
                     f"${pct(flm)} \\pm {100*(max(fl)-min(fl))/2:.1f}$",
                     "1.00$\\times$", "(def.)"])
        if L == "8":
            cap = ("Pythia-1.4B layer 8. Same construction as the GPT-2 "
                   "panel; every codec arm is seed 0 (the floor carries "
                   "three pairs), so single-seed cells closer than "
                   "$\\pm$1.1\\,pp are not resolved. Ratios carry a 95\\% "
                   "paired feature-bootstrap interval (20k resamples over "
                   "the shared alive set); every one excludes 1. Read the "
                   "interval narrowly: it resamples features only, against "
                   "a seed-0 numerator and a single floor pair, so seed "
                   "variance (the larger component) is zero by "
                   "construction; GPT-2's three-seed half-range of "
                   "$\\pm$1.1\\,pp maps to roughly $\\pm$0.07 on these "
                   "ratios, comparable to the interval width. fp16 storage "
                   "is bit-lossless under the bf16 forward, so its row is "
                   "a run-noise ceiling measured the hard way: a second, "
                   "independent training run at the same seed and bytes "
                   "converged to bit-identical weights, max $|\\Delta| = 0$ "
                   "over all four parameter tensors. Source: "
                   "\\texttt{runs\\_m4\\_L8/eval}.")
        else:
            cap = ("Pythia-1.4B layer 16. Identical construction, interval "
                   "convention and bit-identical fp16 ceiling reading to "
                   "Table~\\ref{tab:m4l8}. Source: "
                   "\\texttt{runs\\_m4\\_L16/eval}.")
        emit(cap,
             f"tab:m4l{L}",
             ["codec", "survived \\%", r"$\times$ floor", "stable \\%"],
             rows)


def jumprelu_table():
    fl = [j(f"runs_m1/eval/pair_fp32_jumprelu_s0__fp32_jumprelu_s{s}.json")
          ["survived"] for s in (1, 2)]
    fl.append(j("runs_m1/eval/pair_fp32_jumprelu_s1__fp32_jumprelu_s2.json")
              ["survived"])
    rows = []
    for c in ["fp16", "rot-int8", "rot-int4", "int4"]:
        p = j(f"runs_m1/eval/pair_fp32_jumprelu_s0__{c}_jumprelu_s0.json")
        if p:
            rows.append([c, pct(p["survived"]), f"{p['mean_corr']:.3f}"])
    fl_r = [j("runs_m1/eval/pair_fp32_jumprelu_s0__fp32_jumprelu_s1.json")["mean_corr"],
            j("runs_m1/eval/pair_fp32_jumprelu_s0__fp32_jumprelu_s2.json")["mean_corr"],
            j("runs_m1/eval/pair_fp32_jumprelu_s1__fp32_jumprelu_s2.json")["mean_corr"]]
    rows.append([r"\midrule seed floor",
                 f"{pct(min(fl))}--{pct(max(fl))}", f"{np.mean(fl_r):.3f}"])
    def final_l0(sae):
        """Last logged L0; summary.json does not carry it."""
        p = ROOT / "runs_m1" / "saes" / sae / "train_log.jsonl"
        if not p.exists():
            return None
        for line in reversed(p.read_text().splitlines()):
            rec = json.loads(line)
            if "l0" in rec:
                return rec["l0"]
        return None
    jr_ref = j("runs_m1/saes/fp32_jumprelu_s0/summary.json") or {}
    l0_c = final_l0("int4_jumprelu_s0")
    l0_r = final_l0("fp32_jumprelu_s0")
    dead = jr_ref.get("final_dead_frac")
    disc = " Two disclosures. "
    if dead is not None:
        disc += (f"These runs end with {100 * dead:.0f}\\% of latents dead, "
                 f"so the alive fraction is driven mostly by dead latents "
                 f"rather than by the firing-rate filter. ")
    if l0_c and l0_r:
        disc += (f"The na\\\"ive int4 arm converged to $L_0 = {l0_c:.1f}$ "
                 f"against fp32's {l0_r:.1f}, a 2$\\times$ sparsity "
                 f"mismatch, so that below-floor cell is confounded with "
                 f"the incorrect-$L_0$ effect and we exclude it; the "
                 f"three below-floor settings counted in the main text are "
                 f"all at matched sparsity. ")
    disc += ("Fixed-coefficient ReLU$+$L1 collapsed ($L_0 \\to 0$) at the "
             "calibrated coefficient, with a 10$\\times$ weaker probe "
             "entering the same spiral; we report it as an observed failure "
             "mode rather than a panel.")
    emit("JumpReLU architecture panel (GPT-2, calibrated $\\lambda=8.68$, "
         "final $L_0\\approx 19$). fp16/rot-int8/rot-int4 are "
         "statistically inseparable from each other and sit at the seed "
         "floor's edge." + disc + " Source: "
         "\\texttt{runs\\_m1/eval/pair\\_fp32\\_jumprelu*}.",
         "tab:jumprelu",
         ["codec", "survived \\%", "mean $r$"], rows)


def sweep_budget_tables():
    rows = []
    for k in (16, 32, 64):
        vals = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            p = j(f"runs_m1/eval/pair_rot-int8_k{k}_s{a}__rot-int8_k{k}_s{b}.json")
            if p:
                vals.append(p["survived"])
        rows.append([f"$k={k}$", f"{pct(min(vals))}--{pct(max(vals))}"])

    d = j("runs_m1/eval/budget_fixedset.json")
    rows = []
    for k, label in (("1x", "$1\\times$ (33M visits)"),
                     ("3x", "$3\\times$ (98M)"),
                     ("10x", "$10\\times$ (328M)")):
        t = d["tiers"][k]
        rows.append([label, f"{t['stability_threshold_at_N']:.3f}",
                     pct(t["fp16"]), pct(t["rot-int8"]), pct(t["rot-int4"]),
                     f"{t['gap_fp16_minus_rotint4_pp']:.1f}"])
    tiers = d["tiers"]
    counts = "/".join(str(tiers[k]["fp16_count"]) for k in ("1x", "3x", "10x"))
    overlaps = "/".join(str(tiers[k]["overlap_with_1x_set"])
                        for k in ("3x", "10x"))
    emit("Budget scaling at a FIXED stable-set size (GPT-2). Features are "
         f"ranked by cross-seed stability and the top $N = {d['n_fixed']}$ "
         "are taken at every tier, so the population size and statistical "
         "power do not drift with budget (binomial SE "
         f"$\\approx${d['binomial_se_pp']:.1f}\\,pp). fp16 is "
         "budget-invariant; the rotated codecs improve. The threshold "
         "column is the stability score at rank $N$, which rises with "
         "budget. Budget scales by epoch-recycling one 20M-token cache, "
         "isolating optimization budget rather than data diversity. "
         f"fp16's column repeats to full numeric precision, not merely at "
         f"one decimal, because it lands on "
         f"the same count ({counts} of {d['n_fixed']}) at all three tiers; "
         "this is a coincidence at the counting level, not a copied number"
         f": the tiers' fixed sets share only {overlaps} of "
         f"{d['n_fixed']} features with the $1\\times$ set. "
         "Source: \\texttt{budget\\_fixedset.json}.",
         "tab:budget",
         ["budget", "stab.\\ thresh", "fp16 \\%", "rot-int8 \\%",
          "rot-int4 \\%", "gap pp"], rows)


def probes_table():
    rows = []
    for task in ("sst2", "ag_news"):
        for c in ("fp16", "rot-int8", "rot-int4", "int4-o16", "int4",
                  "pca256"):
            r = j(f"runs_m1/eval/probe_{task}__{c}.json")
            if r:
                rows.append([task.replace("_", r"\_"), c,
                             f"{r['acc_codec']-r['acc_fp32']:+.4f}",
                             f"{r['direction_cos']:.3f}"])
    emit("Linear probes: accuracy delta vs fp32-trained probe (evaluated "
         "on clean fp32 validation activations) and learned-direction "
         "cosine in raw activation space (multinomial LBFGS, fixed ridge "
         "$10^{-4}$, unswept: the cosine levels depend on that "
         "coefficient and we did not vary it). Accuracy resolution is one validation "
         "example, $1/872$ on sst2 and $1/7600$ on ag\\_news, so read the "
         "deltas as counts. Both against a resample floor of "
         "$0.008$/$0.002$ accuracy and $0.80$/$0.77$ cosine over seven "
         "disjoint-half splits (\\texttt{probe\\_floor.json}). PCA's accuracy "
         "straddles that floor (inside on sst2, outside on ag\\_news), "
         "while its direction falls below every split. "
         "\\textbf{int4-o16 also fails the direction floor on sst2}: "
         "$0.778$ against a floor whose minimum over seven splits "
         "is $0.788$, though it clears it on ag\\_news ($0.937$). That is a "
         "codec the retraining test passes comfortably ($1.63\\times$ floor), "
         "so retrained agreement is not uniformly the strictest readout. "
         "The two remaining codecs with probe runs and no row "
         "(\\texttt{int8}, \\texttt{rot-int6}) clear both floors on both "
         "tasks ($0.991$/$0.995$ and $0.998$/$0.998$) and are omitted for "
         "width. Source: "
         "\\texttt{probe\\_*}.",
         "tab:probes",
         ["task", "codec", r"$\Delta$acc", "direction cos"], rows)


def entropy_throughput_tables():
    ent = j("runs_m1/eval/entropy.json")
    rows = []
    for c in ("fp16", "int8", "rot-int8", "int6", "rot-int6", "int4",
              "rot-int4", "std-int4", "int4-o16"):
        if c in ent:
            e = ent[c]
            rows.append([c, f"{e['raw_B_per_tok']:.0f}",
                         f"{e['zstd_B_per_tok']:.1f}",
                         f"{e['ratio']:.2f}$\\times$"])
    # entropy.py compresses shard 0 of each store; report that scope rather
    # than letting the numbers read as whole-store measurements
    man = j("runs_m1/cache/fp16-train/manifest.json")
    shard_n = man["shards"][0]["n_tokens"] if man else None
    scope = (f"one shard ({shard_n:,} tokens)".replace(",", "{,}") if shard_n
             else "one shard")
    emit("Lossless entropy stage (zstd-3) over quantized payloads. "
         "Rotated codecs barely compress (their bits are already "
         "near-maximum-entropy), while na\\\"ive scaling wastes bits on "
         f"outlier headroom. Measured on {scope} per codec rather than the "
         "whole store, so the ratios and rates are a shard-level estimate. "
         "Source: \\texttt{entropy.json}.",
         "tab:entropy",
         ["codec", "raw B/tok", "zstd B/tok", "ratio"], rows)

    th = j("runs_m1/eval/throughput_bench.json")
    rows = []
    for c in ("fp32", "fp16", "int8", "rot-int8", "int6", "rot-int6",
              "int4", "rot-int4", "int4-o16", "pca256"):
        if c in th:
            t = th[c]
            rows.append([c, f"{t['disk_bytes_read']/t['tokens_read']:.0f}",
                         f"{t['read_disk_MB_per_s']:.0f}",
                         f"{t['decode_MB_per_s']:.0f}",
                         f"{t['read_tokens_per_s']/1000:.0f}k"])
    if rows:
        s = j("runs_m1/saes/fp32_s0/summary.json")
        cons = s["cfg"]["steps"] * s["cfg"]["batch_size"] / s["train_seconds"]
        slowest = min((th[c]["read_tokens_per_s"], c) for c in th
                      if isinstance(th[c], dict))
        rot_frac = (th["rot-int8"]["read_tokens_per_s"]
                    / th["int8"]["read_tokens_per_s"])
        sr = j("runs_m1/eval/throughput_std_vs_rot.json")
        paired = ""
        if sr and "means" in sr and "std-int4" in sr and "rot-int4" in sr:
            a = sr["means"]["std-int4"]["decode_MB_per_s"]
            b = sr["means"]["rot-int4"]["decode_MB_per_s"]
            paired = (f" In a paired idle-machine measurement (two interleaved runs each), std-int4 decodes at {a:.0f}\\,MB/s against rot-int4's {b:.0f}; our std-int4 keeps the rotation matmul with an identity matrix, so this does not isolate the rotation's own cost.")
        PAIRED["tab:throughput"] = paired
        emit("Serial loader throughput per store: prefetch $= 0$, warm "
             "page cache, one Apple M5, single thread, so the disk/decode "
             "split is clean and every number is a lower bound on a "
             "pipelined loader. For scale, the $1\\times$ training run "
             f"consumes {cons:,.0f} tokens/s".replace(",", "{,}") +
             " (8{,}000 steps $\\times$ 4{,}096 batch over the recorded "
             "wall-clock), so even the slowest store here "
             f"({slowest[1]}, {slowest[0]/1000:.0f}k tokens/s) outruns "
             "training consumption serially on a laptop; loading, not "
             "fidelity, was the argument for streaming, and after "
             "compression it is not the bottleneck at this scale. The "
             "rotation's dense decode matmul costs real throughput: "
             f"rot-int8 reads at {100*rot_frac:.0f}\\% of int8's rate, "
             "which is the measured cost the body cites against it. These "
             "are one-machine numbers; absolute rates on other hardware "
             "and pipelined loaders will differ. Source: "
             "\\texttt{throughput\\_bench.json} via "
             "\\texttt{scripts/bench\\_throughput.py}.",
             "tab:throughput",
             ["codec", "B/tok read", "disk MB/s", "decode MB/s",
              "loader tok/s"], rows)

def transfer_table():
    """Does the frozen -> retrained mapping transfer across sites?

    The frozen SAE orders codecs perfectly within a site. This table is
    the reason that is not enough: the same reading maps to very
    different retrained outcomes at different sites, so it cannot be
    calibrated once and reused."""
    sites = [("GPT-2 L6", "runs_m1"), ("Pythia L8", "runs_m4_L8"),
             ("Pythia L16", "runs_m4_L16")]
    codecs = ["fp16", "rot-int8", "int8", "rot-int4", "int4"]
    if not all(j(f"{r}/eval/frozen_rot-int8__fp32_s0.json") for _, r in sites):
        return
    rows = []
    for c in codecs:
        cells = [c]
        for _, r in sites:
            fr = j(f"{r}/eval/frozen_{c}__fp32_s0.json")
            pr = j(f"{r}/eval/pair_fp32_s0__{c}_s0.json")
            if not (fr and pr):
                cells += ["---", "---"]
                continue
            cells += [f"{fr['mean_corr']:.5f}", pct(pr["survived"])]
        rows.append(cells)
    fl = []
    for _, r in sites:
        v = [j(f"{r}/eval/pair_fp32_s{a}__fp32_s{b}.json")["survived"]
             for a, b in ((0, 1), (0, 2), (1, 2))]
        fl += ["---", pct(float(np.mean(v)))]
    rows.append([r"\midrule seed floor"] + fl)
    r8 = [j(f"{r}/eval/frozen_rot-int8__fp32_s0.json")["mean_corr"]
          for _, r in sites]
    s8 = [j(f"{r}/eval/pair_fp32_s0__rot-int8_s0.json")["survived"]
          for _, r in sites]
    emit("Does the frozen-SAE reading mean the same thing at another "
         "site? No. Within each site it orders these codecs exactly "
         "as retraining does ($\\rho = 1.000$ over these five codecs; "
         "$0.995$ over the 13 storage codecs of the GPT-2 panel, "
         "Table~\\ref{tab:m1full}). Across sites it "
         f"carries no scale: rot-int8's frozen $r$ agrees to "
         f"{max(r8)-min(r8):.5f} while its retrained survival lands at "
         f"$2.09$--$2.73\\times$ each site's own floor (raw survival is not "
         f"comparable across sites). A frozen-SAE reading calibrated at "
         "one site does not transfer to the next. Source: "
         "\\texttt{frozen\\_*} and \\texttt{pair\\_*} JSONs at all three sites.",
         "tab:transfer",
         ["codec", "GPT-2 $r$", "retr.\\%", "L8 $r$", "retr.\\%",
          "L16 $r$", "retr.\\%"], rows)


def sparsity_invariance_table():
    """Which quantity actually transfers across the sparsity axis.

    The k-sweep moves the seed floor by ~2x without touching the codec or
    the bytes, which makes it the cleanest available test of whether
    ratio-to-floor is doing real work. It is not: the ratio spreads ~88%
    while stratified stable-feature survival spreads ~5%."""
    rows, ratios, stables, floors = [], [], [], []
    for k, tag, cand in ((16, "_k16", "rot-int8_k16_s0"),
                         (32, "", "rot-int8_s0"),
                         (64, "_k64", "rot-int8_k64_s0")):
        fl = [j(f"runs_m1/eval/pair_fp32{tag}_s{a}__fp32{tag}_s{b}.json")
              for a, b in ((0, 1), (0, 2), (1, 2))]
        fl = [x["survived"] for x in fl if x]
        pair = j(f"runs_m1/eval/pair_fp32{tag}_s0__{cand}.json")
        strat = j(f"runs_m1/eval/stability_fp32{tag}_s0_t0p9.json")
        if not (fl and pair and strat):
            return
        f = float(np.mean(fl))
        st = {r["cand"]: r for r in strat["rows"]}[cand]
        floors.append(f)
        ratios.append(pair["survived"] / f)
        stables.append(st["survived_stable"])
        rows.append([f"$k={k}$", f"${pct(f)} \\pm {100*(max(fl)-min(fl))/2:.1f}$",
                     pct(pair["survived"]), f"{pair['survived']/f:.2f}$\\times$",
                     pct(st["survived_stable"]), f"{pair['n_alive']:,}",
                     pct(strat["stable_frac_of_alive"])])
    rspread = 100 * (max(ratios) - min(ratios)) / min(ratios)
    sspread = 100 * (max(stables) - min(stables)) / min(stables)
    emit("Sparsity moves the floor without touching the codec, the bytes "
         "or the activations, which makes it a clean test of what "
         f"calibration buys. Over a {max(floors)/min(floors):.2f}$\\times$ swing in the floor, "
         f"ratio-to-floor spreads {min(ratios):.2f}--{max(ratios):.2f}"
         f"$\\times$ ({rspread:.0f}\\% relative) while survival among "
         f"seed-stable features spreads {pct(min(stables))}--"
         f"{pct(max(stables))}\\% ({sspread:.0f}\\% relative). Read that "
         "contrast with care: the stratified quantity is a proportion near "
         "its ceiling, and on the complementary failure scale it spreads "
         "MORE than the ratio does, so the comparison is not "
         "scale-invariant. Caveat: "
         "$n_{\\text{alive}}$ and the stable fraction both co-vary with "
         "$k$, so the three rows score different-sized populations. "
         "Source: \\texttt{pair/stability} JSONs for "
         "\\texttt{fp32\\_k*} and \\texttt{rot-int8\\_k*}.",
         "tab:sparsity",
         ["sparsity", "seed floor \\%", "rot-int8 \\%", r"$\times$floor",
          "stable-only \\%", "$n_{\\text{alive}}$", "stable frac \\%"], rows)


def forward_precision_table():
    """The §7 control: a treatment that is not a storage codec at all.

    Emitted from the same pair/frozen/stability artifacts as every other
    codec row, so the forward-precision point is measured on exactly the
    path the rest of the paper is measured on."""
    meta = j("runs_m1/eval/forward_precision_fwdbf16.json")
    if meta is None:
        return
    fl = np.mean([j(f"runs_m1/eval/pair_fp32_s{a}__fp32_s{b}.json")["survived"]
                  for a, b in ((0, 1), (0, 2), (1, 2))])
    strat = {r["cand"]: r for r in
             j("runs_m1/eval/stability_fp32_s0_t0p9.json")["rows"]}
    order = ["fp16", "rot-int8", "int8-o8", "fwdbf16", "int8"]
    rows = []
    for c in order:
        pair = j(f"runs_m1/eval/pair_fp32_s0__{c}_s0.json")
        froz = j(f"runs_m1/eval/frozen_{c}__fp32_s0.json")
        hung = j(f"runs_m1/eval/hungarian_fp32_s0__{c}_s0.json")
        if not (pair and froz):
            continue
        bpt_v = bytes_per_token("runs_m1", "fp32" if c == "fwdbf16" else c)
        label = (r"\textbf{bf16 forward}" if c == "fwdbf16"
                 else f"{c} storage")
        cells = [label, f"{bpt_v:.0f}" if bpt_v else "---",
                 f"{froz['direct']['rel_l2_mean']:.5f}",
                 f"{froz['mean_corr']:.5f}", pct(pair["survived"]),
                 f"{pair['survived']/fl:.2f}",
                 pct(strat.get(f"{c}_s0", {}).get("survived_stable")),
                 f"{hung['mean_cos']:.3f}" if hung else "---",
                 f"{pair['fvu_cand']:.4f}"]
        if c == "fwdbf16":
            cells = [cells[0]] + [rf"\textbf{{{v}}}" for v in cells[1:]]
        rows.append(cells)
    rows.append([r"\midrule \textbf{seed floor}", "---", "---", "---",
                 rf"\textbf{{{pct(fl)}}}", r"\textbf{1.00}", "(def.)",
                 "---", "---"])
    # Three-seed arms for the two treatments the section compares. The
    # single-seed version of this comparison sat inside our own tie rule,
    # which is not a basis for the paper's generalization claim.
    seeds = {}
    for c in ("fwdbf16", "rot-int8"):
        vals = []
        for s in (0, 1, 2):
            p = (j(f"runs_m1/eval/pair_fp32_s{s}__{c}_s{s}.json")
                 or j(f"runs_m1/eval/pair_fp32_s{s}__{c}_k32_s{s}.json"))
            if p:
                vals.append(100 * p["survived"])
        seeds[c] = vals
    if len(seeds.get("fwdbf16", [])) == 3 and len(seeds.get("rot-int8", [])) == 3:
        f_, r_ = seeds["fwdbf16"], seeds["rot-int8"]
        rows.append([r"\midrule \textbf{bf16 fwd, 3 seeds}", "---", "---",
                     "---", rf"\textbf{{{min(f_):.1f}--{max(f_):.1f}}}",
                     "---", "---", "---", "---"])
        rows.append([r"rot-int8, 3 seeds", "---", "---", "---",
                     f"{min(r_):.1f}--{max(r_):.1f}", "---", "---", "---",
                     "---"])
        deltas = ", ".join(f"{a - b:+.1f}" for a, b in zip(f_, r_))
        seed_note = (f" Three seeds per arm: bf16 forward "
                     f"{min(f_):.1f}--{max(f_):.1f}\\% against rot-int8 "
                     f"{min(r_):.1f}--{max(r_):.1f}\\%, disjoint, and "
                     f"seed-matched the treatment is worse at every seed "
                     f"({deltas}\\,pp). Three seeds carry no across-seed "
                     f"$p$-value: an exact sign test cannot go below "
                     f"$p=0.25$ two-sided. The evidence is the within-seed "
                     f"matched pairs of Table~\\ref{{tab:paired}}.")
    else:
        seed_note = ""
    th = j("runs_m1/eval/forward_dtype_throughput.json")
    b3 = {c: j(f"runs_m1/eval/pair_fp32_b3x_s0__{c}_b3x_s0.json")
          for c in ("fwdbf16", "rot-int8")}
    three = ((f" At the $3\\times$ budget the same comparison reads "
              f"{100*b3['fwdbf16']['survived']:.1f} against "
              f"{100*b3['rot-int8']['survived']:.1f}\\%, a gap of "
              f"{100*(b3['rot-int8']['survived']-b3['fwdbf16']['survived']):.2f}\\,pp, "
              "inside the tie band (Table~\\ref{tab:budget} for that tier's floor).")
             if all(b3.values()) else "")
    speed = (f" The bf16 forward runs {th['speedup_bf16_over_fp32']:.2f}$\\times$ "
             f"faster than fp32 ({th['bf16']['median']:,.0f} against ".replace(",", "{,}", 1)
             + f"{th['fp32']['median']:,.0f}".replace(",", "{,}")
             + "\\,tok/s, interleaved in one "
             f"session to control for thermal drift)." if th else "")
    emit("A treatment that is not a storage codec. Computing the cache "
         "with a bf16 model forward compresses nothing (both "
         "sides are stored in fp32), yet it costs more retrained "
         "agreement than rotated 8-bit storage, and the frozen SAE "
         "certifies it as it certifies fp16. At matched raw distortion "
         "it is indistinguishable from the storage codec directly above "
         "it (int8-o8) on all five retrained measures." + seed_note + speed + three +
         " Sources: \\texttt{pair/frozen/hungarian/stability} JSONs and "
         "\\texttt{forward\\_dtype\\_throughput.json}.",
         "tab:forward",
         ["treatment", "B/tok", "rel-L2", "frozen $r$", "survived \\%",
          r"$\times$floor", "stable \\%", "Hung.\\ cos", "FVU"], rows)


def fragility_table():
    fr = j("runs_m1/eval/fragility.json")
    rows = []
    for c, r in fr.items():
        rows.append([c, f"{r['cv_auc_all_feats']:.3f}",
                     f"{r['marginal_auc']['log_mean_act']:.3f}",
                     f"{r['marginal_auc']['log_peak_act']:.3f}",
                     f"{r['marginal_auc']['log_rate']:.3f}"])
    emit("Fragility prediction: 5-fold CV AUC for predicting per-feature "
         "survival from fp32 feature statistics. Mean activation is the "
         "strongest marginal (contrast: peak activation for weight "
         "quantization in Duan). Source: \\texttt{fragility.json}.",
         "tab:fragility",
         ["codec", "CV AUC", "mean act", "peak act", "rate"], rows)


def threshold_table():
    """Ratio-to-floor as a function of the survival threshold.

    Both numerator and denominator are recomputed at each threshold from
    the released per-feature arrays, so this is the same quantity the
    paper quotes at r>0.9, read at four other cutoffs.
    """
    ts = j("runs_m1/eval/threshold_sweep.json")
    thr = [str(t) for t in ts["thresholds"]]
    site_label = {"gpt2_l6": "GPT-2 L6", "pythia_l8": "Pythia L8",
                  "pythia_l16": "Pythia L16"}
    rows = []
    for site, s in ts["sites"].items():
        rows.append([rf"\midrule \textbf{{{site_label.get(site, site)}}}"]
                    + [""] * len(thr))
        rows.append(["floor (survival)"]
                    + [pct(s["floor"][t]["mean"]) for t in thr])
        for c, r in s["codecs"].items():
            if c not in ("fp16", "rot-int8", "rot-int4", "int4"):
                continue
            rows.append([c] + [f"{r[t]['ratio_to_floor']:.2f}$\\times$"
                               for t in thr])
    note = ""
    g = ts["sites"].get("gpt2_l6", {})
    if g:
        note = (f"Rows are a four-codec slice of a sweep covering 13 codecs at "
                f"GPT-2 and these four at each Pythia site. Ordering is "
                f"invariant across it at both Pythia sites. On GPT-2 the only inversions are among rot-int4, "
                f"int4-o16 and int6 (the last two not shown here), whose "
                f"largest inverting gap is "
                f"{g['max_inverting_gap_pp']:.2f}\\,pp, inside the tie "
                f"band. Na\\\"ive int4 is below floor at every threshold at "
                f"every site.")
    emit("Ratio to floor as a function of the survival threshold. The "
         "floor collapses faster than every above-floor treatment, so their "
         "ratios rise with the threshold; na\\\"ive int4's ratio falls, the "
         "floor collapsing more slowly than it does. The LEVEL quoted "
         "anywhere in this paper is a joint property of the codec and the "
         "$r>0.9$ cutoff. What is threshold-invariant is the ordering and "
         "the below-floor verdict, not the level. No retraining: "
         "recomputed from the released \\texttt{best\\_corr} arrays by "
         "\\texttt{scripts/threshold\\_sweep.py}. Source: "
         "\\texttt{threshold\\_sweep.json}.",
         "tab:threshold",
         ["", *[f"$r>{t}$" for t in thr]], rows, note=note)


def _redraw_sentence():
    """Redraw evidence for the two headline rungs, if the d1/d2 arms exist."""
    trip = {}
    for base in ("noise4", "sparseulp6"):
        v = []
        for t in (base, f"{base}d1", f"{base}d2"):
            d = j(f"runs_m1/eval/pair_fp32_s0__{t}_s0.json")
            if not d:
                return ""
            v.append(100 * d["survived"])
        trip[base] = v
    n, s = trip["noise4"], trip["sparseulp6"]
    return ("The fp16-scale noise and the millionth-of-values change "
            "were each re-run twice with fresh randomness: "
            f"{n[0]:.1f}/{n[1]:.1f}/{n[2]:.1f} and "
            f"{s[0]:.1f}/{s[1]:.1f}/{s[2]:.1f}\\%, half-ranges "
            f"{(max(n)-min(n))/2:.1f} and {(max(s)-min(s))/2:.1f}\\,pp, so "
            "neither headline number is an accident of one random draw; "
            "the table prints the original (seed-0 convention) runs. ")


def dose_response_table():
    """The saturation control: fp32 storage on both sides, only the
    perturbation differs, so nothing here is a codec effect."""
    fl = np.mean([j(f"runs_m1/eval/pair_fp32_s{a}__fp32_s{b}.json")["survived"]
                  for a, b in ((0, 1), (0, 2), (1, 2))])
    ceil = j("runs_m1/eval/pair_fp32_s0__fp32_s0_r1.json")
    # sorted by DECREASING rel-L2 below, so the per-decade slope column is
    # always a slope between adjacent rows and fp16 interleaves correctly
    doses = ["noise2", "noise3", "noise4", "noise5", "noise6", "noise7",
             "fp16", "sparseulp6", "sparseulp7", "sparseulp8", "sparseulp9",
             "oneulp"]
    doses = [t for t in doses if j(f"runs_m1/eval/frozen_{t}__fp32_s0.json")]
    # dense family sorts by rel-L2; the sparse family orders by COUNT,
    # since its rel-L2 does not order by count (one ulp on one large
    # value exceeds 77 ulps on small ones) and interleaving is misleading
    SPARSE_ORDER = ["sparseulp6", "sparseulp7", "sparseulp8",
                    "sparseulp9", "oneulp"]
    dense = [t for t in doses if t not in SPARSE_ORDER]
    dense.sort(key=lambda t: -j(f"runs_m1/eval/frozen_{t}__fp32_s0.json")
               ["direct"]["rel_l2_mean"])
    doses = dense + [t for t in SPARSE_ORDER if t in doses]
    n_dense = len(dense)
    counts = j("runs_m1/eval/sparse_counts_bracket.json") or {}

    def _n(c):
        v = counts.get(c, {}).get("total_touched")
        return f"{v:,}".replace(",", "{,}") if v else "?"
    got = []
    for c in doses:
        pair = j(f"runs_m1/eval/pair_fp32_s0__{c}_s0.json")
        froz = j(f"runs_m1/eval/frozen_{c}__fp32_s0.json")
        if not (pair and froz):
            continue
        label = (r"fp16 (codec)" if c == "fp16"
                 else r"1 ulp on $10^{-6}$ of entries ($\sim$15k)" if c == "sparseulp6"
                 else r"1 ulp on \textbf{one} scalar" if c == "oneulp"
                 else f"1 ulp on {_n(c)} scalars" if c.startswith("sparseulp")
                 else f"noise $10^{{-{c[-1]}}}$")
        got.append((label, froz["direct"]["rel_l2_mean"],
                    froz["mean_corr"], 100 * pair["survived"]))
    # Slope is per decade toward SMALLER perturbation, so it belongs to the
    # finer of each adjacent pair; `got` runs fine -> coarse.
    rows = [[r"zero (bit-exact)", "0", "1.000000",
             rf"\textbf{{{pct(ceil['survived']) if ceil else '100.0'}}}", "---"]]
    for i, (label, rl2, mc, surv) in enumerate(got):
        slope = "---"   # sparse rows carry no slope: count, not
        if i + 1 < n_dense:  # magnitude, is their variable
            nx = got[i + 1]
            slope = (f"{(surv - nx[3]) / (math.log10(nx[1]) - math.log10(rl2)):+.2f}")
        # the one-scalar rung is 99.9794%, which rounds to 100.0 and then
        # renders identically to the genuine bit-exact row above it --
        # blunting the very contrast this table exists to draw
        sv = f"{surv:.2f}" if surv > 99.9 else f"{surv:.1f}"
        rows.append([label, f"{rl2:.2e}", f"{mc:.6f}", sv, slope])
    rows.append([r"\midrule seed floor", "---", "---", pct(fl), "---"])
    emit("The noise and sparse-change experiments. Isotropic noise at an "
         "exact target relative $L_2$, stored in fp32 on both sides, so "
         "nothing is quantized and only the perturbation differs; each "
         "store passed its own identity check before use. Agreement "
         "is flat across three decades and stays far below the bit-exact "
         "ceiling even at $10^{-7}$, while the frozen SAE reads "
         "0.999985--1.000000 over the three decades below fp16, and 0.998551 at "
         "the largest noise level. Slope is per decade of relative $L_2$, "
         "toward smaller perturbation; it accelerates at this threshold but "
         "not at others, so we do not fit a knee (Table~\\ref{tab:doseslope}). "
         "Every sparse change moves each touched value by exactly one ulp, "
         "so these experiments differ only in count. The count "
         "response is a step, not a slope: one scalar retains 99.98\\% "
         "(though only 21.3\\% at $r>0.999$, and it moves the trained "
         "weights, max $|\\Delta|$ 0.19 on the decoder), while the four "
         "experiments from $n=77$ to $n=15{,}335$ are flat across two "
         "decades of count, so the entire transition sits in $1<n<77$ "
         "(each new experiment ran once; the locality control is reported "
         "in the text only). Sparse rows sort by count and carry no "
         "pp/decade: count, not magnitude, is their variable, and their "
         "rel-$L_2$ does not even order by count (one ulp on one large "
         "value exceeds 77 on small ones). A "
         "dense codec cannot go below fp32's relative "
         "$\\epsilon \\approx 6\\times10^{-8}$; a sparse one can. That is why we "
         "do not report a $10^{-8}$ dense noise level: that amount is not "
         "representable, so the store would not carry the perturbation we "
         "asked for (\\texttt{caption\\_numbers.json}). "
         + _redraw_sentence() +
         "Sources: "
         "\\texttt{pair/frozen} JSONs for \\texttt{noise\\{2,3,4,5,6,7\\}}, "
         "\\texttt{sparseulp\\{6,7,8,9\\}}, the \\texttt{d1}/\\texttt{d2} "
         "redraws, and \\texttt{sparse\\_counts\\_bracket.json}.",
         "tab:dose",
         ["perturbation", "rel-L2", "frozen $r$", "survived \\%",
          "pp/decade"], rows)


def dose_slope_table():
    """Threshold sensitivity of the dose ladder's slope. The acceleration
    that would license a knee is a property of the counting threshold,
    not of the process, so we report the sweep rather than the fit."""
    import numpy as _np
    rungs = []
    for tag in ("noise4", "noise5", "noise6", "noise7"):
        f = ROOT / "runs_m1" / "eval" / f"pair_fp32_s0__{tag}_s0.npz"
        fr = j(f"runs_m1/eval/frozen_{tag}__fp32_s0.json")
        if not f.exists() or not fr:
            return
        d = _np.load(f)
        bc = d["best_corr"]
        m = d["alive"].astype(bool) if "alive" in d else _np.ones_like(bc, bool)
        rungs.append((fr["direct"]["rel_l2_mean"], bc[m]))
    def gains(fn):
        return [(fn(rungs[i + 1][1]) - fn(rungs[i][1])) /
                (math.log10(rungs[i][0]) - math.log10(rungs[i + 1][0]))
                for i in range(len(rungs) - 1)]
    rows = []
    for t in (0.70, 0.80, 0.85, 0.90, 0.95, 0.99):
        g = gains(lambda a, t=t: 100 * _np.mean(a > t))
        shape = ("accelerating" if g[0] < g[1] < g[2]
                 else "decelerating" if g[0] > g[1] > g[2] else "non-monotone")
        lbl = rf"$r>{t:.2f}$" + (r" (ours)" if abs(t - 0.90) < 1e-9 else "")
        rows.append([lbl] + [f"{x:+.2f}" for x in g] + [shape])
    g = gains(lambda a: _np.mean(a))
    rows.append([r"\midrule mean $r$"] + [f"{x:+.4f}" for x in g]
                + ["flat (threshold-free)"])

    emit("Threshold sensitivity of the noise-response slope: gain in "
         "survival per decade of decreasing perturbation. The "
         "acceleration that would license extrapolating to a knee is "
         "present only near $r \\approx 0.80$--$0.90$, absent at 0.70 and "
         "0.95, reversed at 0.99, and absent from the threshold-free "
         "summary (mean best-match $r$, bottom row, in $r$ units not "
         "percentage points). We therefore report the measured flatness "
         "and decline to fit a shape between $10^{-7}$ and the "
         "millionth-of-values change. Recomputed from "
         "the released \\texttt{best\\_corr} arrays; no retraining.",
         "tab:doseslope",
         ["criterion", "$10^{-4}\\!\\to\\!10^{-5}$",
          "$10^{-5}\\!\\to\\!10^{-6}$", "$10^{-6}\\!\\to\\!10^{-7}$",
          "shape"], rows)


def dose_stratified_table():
    """The ladder on the seed-stable stratum: is saturation an artifact of
    counting features that carry no functional signal? It is not, and the
    asymmetry is larger there."""
    import numpy as _np
    E = ROOT / "runs_m1" / "eval"
    try:
        s1 = _np.load(E / "pair_fp32_s0__fp32_s1.npz")
        s2 = _np.load(E / "pair_fp32_s0__fp32_s2.npz")
    except FileNotFoundError:
        return
    alive = s1["alive"].astype(bool)
    stable = alive & (s1["best_corr"] > 0.9) & (s2["best_corr"] > 0.9)
    rows = []
    for tag, lab in (("noise7", r"noise $10^{-7}$"), ("noise6", r"noise $10^{-6}$"),
                     ("noise5", r"noise $10^{-5}$"), ("noise4", r"noise $10^{-4}$"),
                     ("noise3", r"noise $10^{-3}$"), ("noise2", r"noise $10^{-2}$"),
                     ("fp16", "fp16"), ("rot-int8", "rot-int8"),
                     ("rot-int4", "rot-int4"), ("int4", r"na\"ive int4")):
        f = E / f"pair_fp32_s0__{tag}_s0.npz"
        fr = j(f"runs_m1/eval/frozen_{tag}__fp32_s0.json")
        if not f.exists() or not fr:
            continue
        bc = _np.load(f)["best_corr"]
        rows.append([lab, f"{fr['direct']['rel_l2_mean']:.2e}",
                     f"{100*_np.mean(bc[alive]>0.9):.1f}",
                     f"{100*_np.mean(bc[stable]>0.9):.1f}"])
    emit("The noise experiments and the codecs on the seed-stable stratum "
         f"($n={int(stable.sum())}$, features reproduced in both other fp32 "
         "seeds), beside the all-alive column. The flatness is not an "
         "artifact of counting features that carry no functional signal: "
         "on the stable stratum the noise response is flatter still "
         "($0.35$\\,pp per decade against the codecs' $2.6$). The ratio of "
         "those two rates depends on which windows are compared (within "
         "the noise family alone the slope runs $1.8$ to $7.8$\\,pp per "
         "decade as the perturbation grows), so we quote no single "
         "asymmetry figure here or in the body. "
         "The retraining test discriminates better, not worse, on "
         "the population that carries the signal. Recomputed from the "
         "released \\texttt{best\\_corr} arrays; no retraining.",
         "tab:dosestrat",
         ["arm", "rel-L2", "all alive \\%", "seed-stable \\%"], rows)


def dose_margin_table():
    """The anti-chaos margin is itself threshold-dependent. Its SIGN is what
    the argument uses and that is robust; its magnitude is not, and the
    r>0.9 value the text quotes sits near this curve's maximum."""
    import numpy as _np
    E = ROOT / "runs_m1" / "eval"
    arms = [("$10^{-7}$ dense", "pair_fp32_s0__noise7_s0.npz"),
            ("sparse, 1 ulp on $10^{-6}$", "pair_fp32_s0__sparseulp6_s0.npz")]
    arms = [(l, E / f) for l, f in arms if (E / f).exists()]
    if not arms:
        return
    # three-pair floor, each pair on its own alive mask (a two-pair mean prints
    # 23.1 against the paper's canonical 23.0 and looks like a discrepancy)
    fl_arrs = [_np.load(E / f"pair_fp32_s0__fp32_s{k}.npz") for k in (1, 2)]
    third = E / "pair_fp32_s1__fp32_s2.npz"
    if third.exists():
        fl_arrs.append(_np.load(third))
    rows = []
    for name, path in arms:
        arr = _np.load(path)
        al = arr["alive"].astype(bool)
        rows.append([rf"\midrule \textbf{{{name}}}", "", "", "", ""])
        for t in (0.50, 0.70, 0.85, 0.90, 0.95, 0.99):
            a = 100 * _np.mean(arr["best_corr"][al] > t)
            # each floor pair must use ITS OWN alive mask: best_corr is
            # indexed by that pair's reference latents, so applying seed
            # 0's mask to pair_fp32_s1__fp32_s2 crosses index spaces
            # (overlap 1,960 of 4,860, i.e. chance) and yielded a floor
            # disagreeing with tab:threshold's for the same quantity
            fl = float(_np.mean([100 * _np.mean(
                x["best_corr"][x["alive"].astype(bool)] > t)
                for x in fl_arrs]))
            lbl = rf"$r>{t:.2f}$" + (r" (ours)" if abs(t-0.90) < 1e-9 else "")
            rows.append([lbl, f"{a:.1f}", f"{fl:.1f}", f"{a-fl:+.1f}", f"{100-a:.1f}"])
    emit("How far the two smallest perturbations sit above the seed floor, "
         "as a function of the counting threshold. The margin is positive "
         "everywhere (which is the whole of what the anti-chaos argument "
         "uses), but its magnitude is not a property of the process: it "
         "spans 11--43\\,pp for the smallest dense noise and 12--47 for "
         "the millionth-of-values change, "
         "and the value quoted at $r>0.9$ is near the "
         "maximum. The final column shows that the ``nearer the ceiling "
         "than the floor'' reading reverses at $r>0.95$. Recomputed "
         "from released \\texttt{best\\_corr} arrays; no retraining.",
         "tab:dosemargin",
         ["criterion", "perturbed \\%", "floor \\%", "margin (pp)",
          "below ceiling (pp)"], rows)


def paired_tests_table():
    """Within-seed matched-pairs tests. Every comparison in the paper scores
    two arms on the SAME alive features against the SAME reference, so the
    pairing is real and collapsing to two marginal percentages discards it."""
    d = j("runs_m1/eval/paired_feature_tests.json")
    if not d:
        return
    rows = []
    for k in sorted(d.get("forward_precision", {})):
        v = d["forward_precision"][k]
        rows.append([k.replace("seed", "seed "), f"{v['delta_pp']:+.2f}",
                     str(v["bf16_only"]), str(v["rot_int8_only"]),
                     f"{v['mcnemar_p']:.1e}"])
    if not rows:
        return
    emit("The forward-precision comparison as a matched-pairs design. At "
         "each seed both arms are scored on that seed's own alive set "
         "($4{,}860$/$4{,}892$/$4{,}878$; the filter keys to the reference "
         "seed's firing rates, so the count is not fixed across seeds) "
         "against the same fp32 reference, and the same-seed ceiling is "
         "bit-exact, so a same-seed difference is pure treatment effect. "
         "Discordant counts are features surviving under one arm only. "
         "Features are not independent, so these $p$-values are effect-size "
         "evidence within a run, not claims about a feature population: "
         "what carries the result is that the direction and significance "
         "replicate at all three seeds, where an across-seed sign test on "
         "$n=3$ could not go below 0.25. Source: "
         "\\texttt{paired\\_feature\\_tests.json} via "
         "\\texttt{scripts/paired\\_feature\\_tests.py}.",
         "tab:paired",
         ["", "$\\Delta$ (pp)", "bf16 fwd only", "rot-int8 only",
          "McNemar $p$"], rows)


def firing_rate_table():
    d = j("runs_m1/eval/firing_rate_sweep.json")
    if not d:
        return
    cuts = [str(c) for c in d["cuts"]]
    rows = []
    for site, per in d["sites"].items():
        rows.append([rf"\midrule \textbf{{{site.replace('_', chr(92)+'_')}}}"]
                    + [""] * len(cuts))
        rows.append(["n alive"] + [str(per[c]["n_alive"]) for c in cuts])
        rows.append(["floor \\%"] + [f"{per[c]['floor']:.1f}" for c in cuts])
        rows.append([f"{d['codec']} $\\times$floor"]
                    + [f"{per[c]['ratio_to_floor']:.2f}" for c in cuts])
    emit("The firing-rate cut, swept. \\S\\ref{sec:protocol} counts only "
         "alive features, those firing on more than $0.001$ of "
         "evaluation tokens, and that cut is \\textbf{the most favourable "
         "of the five swept here}: tightening it raises survival and the "
         "floor together and drives the ratio down monotonically, from "
         "$1.63$ to $1.13\\times$ on GPT-2 across the full range. We print "
         "all five rather than the two nearest our choice. The ratio is "
         "therefore not a threshold-free quantity and no value here should "
         "be quoted without its cut. This sweep covers rot-int4 only, so it "
         "bounds that codec's ratio and says nothing about ordering. Its "
         "floor row averages the two seed-0 pairs the per-feature arrays "
         "were saved for, not the three used elsewhere, so it reads "
         "$0.1$\\,pp high against Table~\\ref{tab:m1full}'s floor. "
         "Source: \\texttt{firing\\_rate\\_sweep.json}.",
         "tab:firingrate",
         [""] + [f"$>{c}$" for c in cuts], rows)


def steering_matched_table():
    d = j("runs_m1/eval/steering_matched_prompts.json")
    if not d:
        return
    rows = []
    for task in ("sst2", "ag_news"):
        A = d.get(f"stageA_{task}")
        B = d.get(f"stageB_{task}")
        if not A:
            continue
        for c in ("fp16", "rot-int8", "rot-int4", "int4", "pca256"):
            a = A["arms"].get(c)
            if not a:
                continue
            sw = [B[s]["arms"][c]["ratio"] for s in ("1.0", "2.0", "5.0", "10.0")]
            rows.append([task.replace("_", r"\_"), c,
                         f"{a['ratio_at_mean_floor']:.3f}",
                         f"{a['fails_x1_of_400']}/{a['fails_x2_of_400']}",
                         f"{min(sw):.2f}--{max(sw):.2f}"])
    emit("Causal steering rescored with TASK-MATCHED prompts, and swept "
         "over injection strength. The concern this answers: "
         "Table~\\ref{tab:steering} scored the ag\\_news "
         "World-vs-Sports direction with the eight sentiment "
         "prompts written for sst2, at a single strength, with no sweep. "
         "This rescores ag\\_news with eight neutral news-lead framings "
         "(\\texttt{The report said}, \\texttt{According to officials}, "
         "\\dots) and re-derives the 400-split floor under them; sst2's "
         "prompts were already task-matched and reproduce its published "
         "floor to seven digits, which is the pipeline check. "
         "\\textbf{Both verdicts survive.} Na\\\"ive int4 still fails every "
         "one of the 400 ag\\_news splits and 399 of 400 on sst2 "
         "($400$/$400$ under $\\times2$); PCA-256 still clears the mean "
         "floor on both, with an unchanged failure rate under $\\times2$ "
         "($58$ of 400 ag\\_news splits against the sentiment prompts' "
         "$51$). The sweep column gives the ratio's full range over "
         "strengths $1$, $2$, $5$ and $10$, each scored against a "
         "400-split floor re-derived at that strength, the same "
         "construction as the ratio column. Every verdict is "
         "strength-invariant, so the single-strength number was not "
         "load-bearing. Source: "
         "\\texttt{steering\\_matched\\_prompts.json} via "
         "\\texttt{scripts/steering\\_matched\\_prompts.py}.",
         "tab:steermatched",
         ["task", "codec", "ratio", "fails $\\times1$/$\\times2$",
          "sweep range"], rows)


def stability_definition_table():
    """Stable-stratum survival under four stability definitions."""
    defs = [("$r>0.8$", "runs_m1/eval/stability_fp32_s0_t0p8.json"),
            ("$r>0.9$", "runs_m1/eval/stability_fp32_s0_t0p9.json"),
            ("$r>0.95$", "runs_m1/eval/stability_fp32_s0_t0p95.json"),
            ("Hungarian", "runs_m1/eval/stability_hungarian.json")]
    cols = {}
    for name, path in defs:
        d = j(path)
        if not d:
            return
        rows = d["rows"]
        cols[name] = ({r["cand"]: r for r in rows} if isinstance(rows, list)
                      else rows)
    codecs = ["fp16_s0", "rot-int8_s0", "rot-int4_s0", "int4_s0"]
    table, vals, uvals = [], [], []
    for c in codecs:
        row = [c.replace("_s0", "")]
        for name, _ in defs:
            v = 100 * cols[name][c]["survived_stable"]
            u = 100 * cols[name][c]["survived_unstable"]
            vals.append((c, v)); uvals.append((c, u))
            row.append(f"{v:.1f} / {u:.1f}")
        table.append(row)
    unst = {}
    for site, path in (("L8", "runs_m4_L8/eval/stability_fp32_s0_t0p9.json"),
                       ("L16", "runs_m4_L16/eval/stability_fp32_s0_t0p9.json")):
        d = j(path)
        if d:
            for r in d["rows"]:
                if r["cand"] in ("rot-int8_s0", "int8_s0", "rot-int4_s0"):
                    unst[(site, r["cand"])] = 100 * r["survived_unstable"]
    pyth = (f" On Pythia the viable codecs keep {min(unst.values()):.1f} to "
            f"{max(unst.values()):.1f}\\% of the unstable features "
            "(\\texttt{runs\\_m4\\_L8}, \\texttt{runs\\_m4\\_L16} stability files)."
            if unst else "")
    lo = min(v for c, v in vals if c != "int4_s0")
    hi = max(v for c, v in vals if c != "int4_s0")
    ulo = min(u for c, u in uvals if c != "int4_s0")
    uhi = max(u for c, u in uvals if c != "int4_s0")
    emit("Survival among seed-stable / seed-unstable features under four "
         "definitions of stability: activation-space matching at three "
         "thresholds, and weight-space Hungarian matching on "
         "encoder$+$decoder cosine. Under every definition the viable codecs "
         "keep far more of the stable features than of the unstable ones "
         f"(stable {lo:.1f} to {hi:.1f}\\%, unstable {ulo:.1f} to "
         f"{uhi:.1f}\\%), and na\\\"ive int4 is lowest on both; the levels "
         "move with the definition, so a stratified percentage is only "
         "interpretable with its definition attached." + pyth +
         " GPT-2 L6, seed 0. "
         "Sources: \\texttt{stability\\_fp32\\_s0\\_t0p*.json}, "
         "\\texttt{stability\\_hungarian.json}.",
         "tab:stabdef", ["codec"] + [n + " \\%" for n, _ in defs], table)


def proxy_table():
    """Spearman rank agreement of cheap proxies with retrained agreement."""
    d = j("runs_m1/eval/proxy_ranking.json")
    if not d:
        return
    rows = []
    g = None
    for site, r in d["sites"].items():
        rows.append([site, str(r["n_conditions"]),
                     f"{r['rho_frozen_r_all']:.3f}", f"{r['rho_rel_l2_all']:.3f}",
                     f"{r['rho_cos_all']:.3f}"])
        if site.startswith("GPT-2"):
            g = r
    conds = ", ".join(g["codecs"]) if g else ""
    emit("Spearman rank correlation between three cheap readings of a "
         "codec and its retrained agreement, over every released "
         "condition at each site. The frozen-SAE mean $r$ ranks no better "
         "than plain relative $L_2$ distance or mean cosine, none of which "
         f"needs a trained SAE. The {g['n_conditions'] if g else ''} GPT-2 "
         f"conditions are {conds}. Source: \\texttt{{proxy\\_ranking.json}} "
         "via \\texttt{scripts/proxy\\_ranking.py}.",
         "tab:proxyrank",
         ["site", "$n$", "frozen $r$ $\\rho$", "rel-$L_2$ $\\rho$",
          "cosine $\\rho$"], rows)


def consensus_table():
    """Bias vs variance of codec damage on the fp32 consensus set."""
    d = j("runs_m1/eval/consensus_bias.json")
    if not d:
        return
    rows = []
    for name in ("rot-int8", "rot-int4", "int4"):
        c = d["codecs"].get(name)
        if not c:
            continue
        rows.append([name, f"{100*np.mean(c['loss_rate_per_seed']):.1f}",
                     f"{100*c['p_lost_s1_given_lost_s0']:.1f}",
                     f"{100*c['p_lost_s1_indep']:.1f}",
                     f"{100*c['frac_lost_all_three']:.2f}",
                     f"{100*c['frac_lost_all_three_indep']:.3f}"])
    r8 = d["codecs"]["rot-int8"]; i4 = d["codecs"]["int4"]
    emit("Is codec damage a shared bias or seed-like noise? On the fp32 "
         f"consensus set ($n={d['n_consensus']}$ features of fp32 seed 0 "
         "matched at $r>0.9$ in both other fp32 seeds), we ask whether the "
         "features a codec loses at seed 0 are also lost at seeds 1 and 2, "
         "using same-seed pairs (fp32 seed $i$ against the codec at seed "
         "$i$) so that only the codec differs, and the fp32 cross-seed "
         "matches to carry seed-0 features into seeds 1 and 2. "
         "``Independence'' is the co-loss rate if losses were unrelated "
         "across seeds. Losses are enriched above independence for every "
         "codec, so a shared component exists, but its size separates the "
         f"viable codecs from the failed one: rot-int8 loses only "
         f"{100*r8['frac_lost_all_three']:.2f}\\% of consensus features in "
         f"all three seeds, against {100*i4['frac_lost_all_three']:.1f}\\% "
         "for na\\\"ive int4. Read as a ratio to independence the enrichment "
         "is actually larger for the viable codecs, so the shared component "
         "is real for every codec; what separates them is its absolute size. "
         "A multi-seed protocol therefore recovers almost everything under "
         "a viable codec and cannot under na\\\"ive int4. "
         "No training: recomputed from released pair NPZs by "
         "\\texttt{scripts/consensus\\_bias.py} "
         "(\\texttt{consensus\\_bias.json}).",
         "tab:consensus",
         ["codec", "lost/seed \\%", "P(lost s1 $|$ lost s0) \\%",
          "indep.\\ \\%", "lost all 3 \\%", "indep.\\ \\%"], rows)


def rotation_ablation_table():
    """int4 -> std-int4 -> rot-int4: which operation repairs 4-bit?
    RotQuantCodec standardizes per channel and THEN rotates, so every
    "rotation repairs int4" number in this paper measures both at once.
    std-int4 is byte-identical to rot-int4 with the rotation set to the
    identity, so the two differ by exactly the rotation. Same-seed pairs
    (fp32_sK vs codec_sK) over every seed that exists; three seeds are
    shown as mean +- half the min-max range.
    """
    fl = 100 * np.mean([j(f"runs_m1/eval/pair_fp32_s{a}__fp32_s{b}.json")
                        ["survived"] for a, b in ((0, 1), (0, 2), (1, 2))])
    rows, seeds_of = [], {}
    for c, label in (("int4", r"na\"ive int4"),
                     ("std-int4", r"\textbf{std-int4} (standardize only)"),
                     ("rot-int4", r"rot-int4 (standardize $+$ rotate)")):
        svs = [100 * j(f"runs_m1/eval/pair_fp32_s{s}__{c}_s{s}.json")["survived"]
               for s in (0, 1, 2)
               if j(f"runs_m1/eval/pair_fp32_s{s}__{c}_s{s}.json")]
        fr = j(f"runs_m1/eval/frozen_{c}__fp32_s0.json")
        if not (svs and fr):
            continue
        seeds_of[c] = len(svs)
        mean = float(np.mean(svs))
        cell = (f"${mean:.1f} \\pm {(max(svs) - min(svs)) / 2:.1f}$"
                if len(svs) == 3 else f"{svs[0]:.1f}")
        bpt = theoretical_bytes_per_token(c, 768)
        rows.append([label, f"{bpt:.0f}",
                     f"{fr['direct']['rel_l2_mean']:.4f}",
                     f"{fr['mean_corr']:.5f}", cell,
                     f"{mean/fl:.2f}$\\times$"])
    if len(rows) < 3:
        return
    three = [c for c, k in seeds_of.items() if k == 3]
    seed_note = ("All three arms are three-seed means over same-seed pairs."
                 if len(three) == 3 else
                 "Cells with $\\pm$ are three-seed means over same-seed pairs; "
                 "bare cells are seed 0.")
    std = rows[1][4]; rot = rows[2][4]
    emit("Which step repairs 4-bit. std-int4 is byte-identical to rot-int4 "
         "with the rotation set to the identity, so the two differ by "
         f"exactly the rotation: standardization alone gives {std}\\%, "
         f"adding the rotation {rot}\\%, inside the tie band. "
         f"{seed_note} The ablation keeps the rotation matmul, so it "
         "isolates the rotation's effect, not its decode cost.",
         "tab:ablation",
         ["arm", "B/tok", "rel-$L_2$", "frozen $r$", "survived \\%",
          "$\\times$floor"],
         rows)

def steering_table():
    """The non-SAE, non-probe arm: causal steering vectors on two tasks."""
    import glob, os
    rows, floors = [], {}
    # Divide by the EIGHT-SPLIT MEAN floor, not the seed-0 draw stored in
    # each per-codec file: that floor spans 6-10x across splits, so the
    # single-draw ratios are noise and disagreed with the main text by up
    # to 56%. The numerator is a bias with no split dependence.
    spl = j("runs_m1/eval/steering_floor_splits.json") or {}
    _pca_fail = _pca_n = 0
    _pca_worst = 0.0
    _pf = spl.get("ag_news", {}).get("floors") or []
    _pn = j("runs_m1/eval/steering_ag_news_pca256.json")
    if _pf and _pn:
        _num = _pn["causal"]["l2_effect_fp32_vs_codec"]
        _r = [2 * _num / f for f in _pf]
        _pca_n = len(_r)
        _pca_fail = sum(1 for v in _r if v > 1)
        _pca_worst = max(_r)
    _spread = (max(_pf) / min(_pf)) if _pf else 0.0
    _sf = spl.get('sst2', {}).get('floors') or []
    _spread_lo = (max(_sf) / min(_sf)) if _sf else 0.0
    for task in ("sst2", "ag_news"):
        for f in sorted(glob.glob(str(ROOT / "runs_m1" / "eval" /
                                      f"steering_{task}_*.json"))):
            d = j(os.path.relpath(f, ROOT))
            if not d or "causal" not in d:
                continue
            floors[task] = (d["geometric"]["cos_floor_half_vs_half"],
                            d["causal"]["cos_effect_floor_half_vs_half"])
            mean_floor = spl.get(task, {}).get("mean")
            ratio = (d["causal"]["l2_effect_fp32_vs_codec"] / mean_floor
                     if mean_floor else d["causal"]["ratio_to_floor_l2"])
            rows.append((task, d["codec"],
                         d["geometric"]["cos_fp32_vs_codec"],
                         d["causal"]["cos_effect_fp32_vs_codec"],
                         ratio))
    if not rows:
        return
    order = {"fp16": 0, "rot-int8": 1, "rot-int4": 2, "pca256": 3, "int4": 4}
    rows.sort(key=lambda r: (r[0], order.get(r[1], 9)))
    out = []
    seen = None
    for task, c, g, cc, r in rows:
        if task != seen:
            out.append([rf"\midrule \textbf{{{task.replace(chr(95), chr(92)+chr(95))}}} (floor {floors[task][0]:.4f} / "
                        rf"{floors[task][1]:.4f})", "", "", ""])
            seen = task
        out.append([c, f"{g:.4f}", f"{cc:.4f}",
                    f"{r:.3f}" + (r"  \textbf{FAIL}" if r > 1 else "")])
    emit("The causal arm: a difference-in-means steering vector, injected "
         "into the residual stream at the probed layer; the readouts are "
         "its direction and the shift it induces in the model's output. "
         "The steering test is deterministic, so its same-process control is "
         "the estimator's own nuisance variation, two disjoint halves of "
         "the same fp32 data, at the largest split ($n=67{,}349$ on sst2, "
         "$60{,}000$ on ag\\_news; the floor is strongly $n$-dependent, "
         "\\texttt{caption\\_numbers.json}). A codec passes if it perturbs "
         "the causal effect no more than a data resample does. The floor "
         "differences two half samples while the numerator is a "
         "full-$n$ bias, so it runs exactly $2\\times$ a like-for-like "
         "referent; that is conservative only for a FAIL, so $\\times2$ is "
         "the correct convention. Effect/floor divides by the mean floor "
         f"over {_pca_n} resample splits "
         "(\\texttt{steering\\_floor\\_splits.json}); the floor spans "
         f"{_spread_lo:.0f}--{_spread:.0f}$\\times$ across splits, so read "
         "no digit past the first, and the header cosine floors are seed-0 "
         "draws, not this denominator. Under $\\times2$ PCA-256 does not "
         f"pass cleanly (ag\\_news exceeds 1 on {_pca_fail} of {_pca_n} "
         f"splits, worst {_pca_worst:.2f}: a pass with a failure rate), "
         "while na\\\"ive int4 fails every draw on both tasks, replicating "
         "the below-floor verdict on a test sharing no machinery "
         "with SAEs. PCA-256 clears the line on average while ranking "
         "second-worst on the SAE instrument, so the orderings do not "
         "agree. Table~\\ref{tab:steermatched} rescores ag\\_news with "
         "task-matched prompts and sweeps injection strength; every "
         "verdict survives. Source: \\texttt{steering\\_*.json} via "
         "\\texttt{scripts/run\\_steering.sh}.",
         "tab:steering",
         ["codec", "direction cos", "causal cos", "effect/floor"], out)


def _mhr(vals, dp=1, scale=100):
    """mean +- half the min-max range (the paper's three-run convention)."""
    v = [scale * x for x in vals]
    if len(v) == 1:
        return f"{v[0]:.{dp}f}"
    return f"${np.mean(v):.{dp}f} \\pm {(max(v) - min(v)) / 2:.{dp}f}$"


def subspace_table():
    """Held-out subspace recovery (scripts/subspace_recovery.py): the share
    of alive fp32 features whose activation a least-squares map from their
    m best-correlated partners explains at R^2 > 0.81 on held-out tokens.
    m = 1 is the retraining test's own count. Camera-ready addition."""
    groups = [
        ("identical bytes", [("fp32_s0", "fp32_s0_r1")]),
        (r"1 ulp on $10^{-6}$ of values",
         [("fp32_s0", "sparseulp6_s0"), ("fp32_s0", "sparseulp6d1_s0"),
          ("fp32_s0", "sparseulp6d2_s0")]),
        ("fp16", [("fp32_s0", "fp16_s0")]),
        ("rot-int4", [("fp32_s0", "rot-int4_s0")]),
        ("std-int4", [("fp32_s0", "std-int4_s0"), ("fp32_s1", "std-int4_s1"),
                      ("fp32_s2", "std-int4_s2")]),
        (r"na\"ive int4", [("fp32_s0", "int4_s0")]),
        ("reseed", [("fp32_s0", "fp32_s1"), ("fp32_s0", "fp32_s2"),
                    ("fp32_s1", "fp32_s2")]),
    ]
    rows, null_max = [], []
    for name, pairs in groups:
        ds = [j(f"runs_m1/eval/subspace_{a}__{b}.json") for a, b in pairs]
        if not all(ds):
            continue
        cells = [name] + [_mhr([d[f"S_{m}"] for d in ds]) for m in (1, 4, 8)]
        rec = [d["recovered_4_among_not_survived"] for d in ds]
        cells.append("---" if any(math.isnan(x) for x in rec) else _mhr(rec))
        null_max.append(max(d["S_4_null"] for d in ds))
        evs = [d["ev_set_level"]["64"]["ref_to_cand"] / d["ev_set_level"]["64"]["within_ref"]
               for d in ds if d["ev_set_level"]]
        cells.append(_mhr(evs, dp=2, scale=1) if evs else "---")
        rows.append(cells)
    if len(rows) < 7:
        return
    s = j("runs_m1/eval/subspace_summary.json")
    emit("Is a lost feature gone or re-cut? Share of alive fp32 features "
         "recovered on held-out tokens by a least-squares map from their $m$ "
         "best-correlated partners ($R^2 > 0.81$, so $m{=}1$ is the "
         "retraining count re-scored on these tokens). Recovering from four partners keeps std-int4 "
         f"{100 * s['gap_std_minus_reseed_S_4']:.1f}\\,pp above reseeding "
         f"(three seeds each). Lost: of the features lost at $m{{=}}1$, the "
         "share four partners recover, "
         f"{100 * s['recovered_4_among_lost_range_excl_int4'][0]:.0f}"
         f"--{100 * s['recovered_4_among_lost_range_excl_int4'][1]:.0f}\\% in "
         "every condition but na\\\"ive int4; four random partners recover "
         f"at most {100 * max(null_max):.2f}\\%. Set EV: the lost features' "
         "top-64 decoder subspace, applied to the retrained SAE's unmatched "
         "features, as a fraction of its within-SAE value~\\citep{reprosubspaces}. "
         "$\\pm$ rows are three runs.",
         "tab:subspace",
         ["condition", "$m{=}1$", "$m{=}4$", "$m{=}8$",
          "lost, $m{=}4$", "set EV"],
         rows)


def feature_ablation_table():
    """Causal check (scripts/feature_ablation.py): remove a sampled fp32
    feature's own contribution from GPT-2's residual stream at the site,
    then its best match's, on the same held-out rows; compare the two
    changes in next-token log-probabilities. Camera-ready addition."""
    d = j("runs_m1/eval/feature_ablation_fp32_s0.json")
    s = j("runs_m1/eval/feature_ablation_summary.json")
    if not (d and s):
        return
    names = {"fp32_s0_r1": "identical bytes",
             "sparseulp6_s0": r"1 ulp on $10^{-6}$ of values",
             "fp16_s0": "fp16", "std-int4_s0": "std-int4",
             "int4_s0": r"na\"ive int4", "fp32_s1": "reseed, seed 1",
             "fp32_s2": "reseed, seed 2"}
    rows = []
    for c in d["conds"]:
        v, sm = s["conds"][c], d["summary"][c]
        among = v["frac_effect_gt_0.9_among_survived"]
        rows.append([names.get(c, c), pct(v["frac_survived"]),
                     pct(v["frac_effect_gt_0.9"]),
                     pct(among) if among is not None else "---",
                     f"{v['median_effect_cos']:.3f}",
                     f"{v['median_dnext_r']:.3f}",
                     f"{v['median_ctx_overlap']:.2f}"])
    f3 = d["contrasts"]["F3_frac_gt_0.9_std_minus_reseed"]
    f2 = d["contrasts"]["F2_survived_median_std_minus_reseed"]
    emit(f"Do matched features do the same thing? {d['n_features']} sampled "
         "alive fp32 features, each removed from the residual stream on its "
         f"{d['rows_per_feature']} strongest held-out rows, then its best "
         "match in each retrained SAE on the same rows. Same effect: cosine "
         "above $0.9$ between the two changes in next-token "
         "log-probabilities; the last three columns are medians. "
         "std-int4 exceeds reseeding by "
         f"{100 * f3['estimate']:.1f}\\,pp (95\\% CI {100 * f3['ci95'][0]:.1f} "
         f"to {100 * f3['ci95'][1]:.1f}); among pairs matched at $r > 0.9$ "
         "its effects are as close as a reseed's (median difference "
         f"${f2['estimate']:+.3f}$, CI ${f2['ci95'][0]:+.3f}$ to "
         f"${f2['ci95'][1]:+.3f}$). Seed 0, one sample.",
         "tab:featablation",
         ["condition", "$r > 0.9$", "same effect", r"same $|\,r > 0.9$",
          "effect cos", r"$\Delta$logp $r$", "top-50 ctx"],
         rows)


def fresh_table():
    """3x budget, fresh text vs replay (scripts/run_fresh_b3x.sh, R1/R2;
    stable columns from scripts/fresh_stable.py). Camera-ready addition."""
    s = j("runs_m1/eval/fresh_summary.json")
    st = j("runs_m1/eval/fresh_stable.json")
    if not (s and st):
        return

    def cell(d):
        return f"${100 * d['mean']:.1f} \\pm {100 * d['half_range']:.1f}$"

    def seeds(reg, codec, key):
        v = [r[codec][key] for r in st["regimes"][reg]["per_seed"]]
        return _mhr(v)

    r1, r2 = s["R1"], s["R2"]
    rows = [
        [r"replay (20M, $\sim$5 passes)", "floor (fp32 reseed)", cell(r1["floor_replay"]),
         "1.00", "---", "---"],
        ["", "fp16", cell(r2["fp16_replay"]), f"{r2['fp16_ratio_replay']:.2f}",
         seeds("replay", "fp16_b3x", "stable"), seeds("replay", "fp16_b3x", "fixed774")],
        ["", "std-int4", cell(r2["std_int4_replay"]), f"{r2['ratio_replay']:.2f}",
         seeds("replay", "std-int4_b3x", "stable"), seeds("replay", "std-int4_b3x", "fixed774")],
        [r"\midrule fresh (98.5M, one pass)", "floor (fp32 reseed)", cell(r1["floor_fresh"]),
         "1.00", "---", "---"],
        ["", "std-int4", cell(r2["std_int4_fresh"]), f"{r2['ratio_fresh']:.2f}",
         seeds("fresh", "std-int4_freshb3x", "stable"),
         seeds("fresh", "std-int4_freshb3x", "fixed774")],
    ]
    emit("Fresh text against replay at the $3\\times$ budget (98.3M token-visits). "
         "Floors are three seed pairs; codec rows are same-seed pairs at three "
         "seeds, all $\\pm$ half the range. Stable: features matched above "
         "$r = 0.9$ in both other fp32 seeds of the same regime; fixed-774: the "
         "top 774 by cross-seed agreement, as in Table~\\ref{tab:budget}. The "
         f"fresh floor sits {100 * r1['fresh_minus_replay']:+.2f}\\,pp from the "
         "replay floor and std-int4 keeps the same lead in both regimes.",
         "tab:fresh",
         ["regime", "SAE", "agreement \\%", "$\\times$floor", "stable \\%", "fixed-774 \\%"],
         rows)


def equal_bytes_table():
    """Equal stored bytes at 3x (R3 of run_fresh_b3x.sh, cross-seed against
    fresh fp32), with the post-hoc resolution rows. Camera-ready addition."""
    s = j("runs_m1/eval/fresh_summary.json")
    if not s:
        return
    r3, lay = s["R3"], s["store"]
    res = r3["posthoc_resolution"]
    b = lambda codec: bytes_per_token("runs_m1", codec)
    n20 = 20_000_087 / 1e6

    def cell(d):
        return f"{100 * d['mean']:.1f} ({100 * d['min']:.1f}--{100 * d['max']:.1f})"

    fr = s["R1"]["floor_fresh"]
    rows = [
        ["fp32, fresh (reference)", f"{lay['master_fresh_tokens'] / 1e6:.1f}",
         f"{lay['master_fresh_tokens'] * 3072 / 1e9:.0f}", cell(fr), "1.00"],
        ["fp16, replayed", f"{n20:.1f}", f"{lay['fp16_payload_bytes'] / 1e9:.1f}",
         cell(r3["fp16_replay_20M"]), f"{r3['x16_over_floor']:.3f}"],
        ["std-int4, more text", f"{lay['mb_tokens'] / 1e6:.1f}",
         f"{lay['mb_payload_bytes'] / 1e9:.1f}", cell(r3["std_int4_mb"]),
         f"{r3['x4_over_floor']:.3f}"],
        [r"\midrule fp32, replayed$^\dagger$", f"{n20:.1f}", f"{20_000_087 * b('fp32') / 1e9:.1f}",
         cell(res["fp32_replay"]), f"{res['fp32_replay_over_floor']:.3f}"],
        [r"na\"ive int4, replayed$^\dagger$", f"{n20:.1f}", f"{20_000_087 * b('int4') / 1e9:.1f}",
         cell(res["naive_int4_replay"]), f"{res['naive_int4_replay']['mean'] / fr['mean']:.3f}"],
    ]
    emit("Equal stored bytes at $3\\times$: fp16 on the 20M-token cache against "
         "std-int4 on the first 79.0M tokens of the fresh store, in "
         f"{100 * r3['mb_bytes_over_fp16']:.2f}\\% of fp16's bytes. Agreement is "
         "cross-seed against fresh fp32 (six pairs per candidate, mean and "
         "range), so no candidate shares an initialization or data order with "
         "its reference; the reference row is fresh fp32 against itself across "
         "seeds (three pairs). The two "
         f"candidates differ by {100 * r3['x4_minus_x16']:+.2f}\\,pp: a tie. "
         "$^\\dagger$Post-hoc resolution check, not pre-registered (na\\\"ive int4: "
         "seed 0 only, two pairs).",
         "tab:equalbytes",
         ["candidate", "tokens (M)", "stored GB", "agreement \\%", "$\\times$ref"],
         rows)


def timing_table():
    """End-to-end training throughput on an A100 by format and decode path
    (scripts/modal_timing.py; readings via scripts/timing_summary.py)."""
    t = j("runs_m1/eval/timing_summary.json")
    if not t or "a100_local" not in t:
        return
    names = {"cpu": "CPU (paper)", "cpu-nomatmul": "CPU, no matmul", "gpu": "GPU"}
    loc = {r["codec"] + r["decode"]: r for r in t["a100_local"]["rows"]}
    vol = {r["codec"] + r["decode"]: r for r in t["a100_volume"]["rows"]}
    rows = []
    for codec in ("fp32", "fp16", "std-int4"):
        for dec in ("cpu", "cpu-nomatmul", "gpu"):
            k = codec + dec
            if k not in loc:
                continue
            a, v = loc[k], vol[k]
            rows.append([codec if dec == "cpu" else "", names[dec],
                         f"{a['tokens_per_s'] / 1e3:.1f}", f"{a['read_share']:.2f}",
                         f"{v['tokens_per_s'] / 1e3:.1f}", f"{v['read_share']:.2f}"])
    d = t["derived"]
    emit("Training throughput on an A100 (thousand tokens/s, full $1\\times$ run, "
         "mean of two fresh containers) by cache format and where the loader "
         "decodes; busy is the share of training time the loader spent "
         "reading, decoding and shuffling. Every decode path of a format "
         "trains bit-identical weights. Best std-int4 over best fp32: "
         f"{t['a100_local']['ratio_best_std_int4_over_best_fp32']:.2f} from "
         f"local disk, {t['a100_volume']['ratio_best_std_int4_over_best_fp32']:.2f} "
         "from the network volume, under the pre-registered $1.10$ for a "
         "material gain. Volume reads varied "
         f"{d['volume_read_MB_per_s_range'][0]:.0f}--{d['volume_read_MB_per_s_range'][1]:.0f}"
         "\\,MB/s between runs.",
         "tab:timing",
         ["format", "decode", "local", "busy", "volume", "busy"],
         rows)


def margins_table():
    """Frozen-SAE disagreement predicted from TopK margins and the codec's
    error covariance (scripts/geom_margins.py, scripts/geom_summary.py)."""
    m = j("runs_m1/eval/geom_margins.json")
    if not m or "readings" not in m:
        return
    c = m["codecs"]
    noise = ["fp16", "bf16", "int8", "int8-o8", "int6", "int6-o8", "rot-int8",
             "rot-int6", "rot-int4", "std-int4", "int4-o16"]
    sig = ["int4", "pca256", "rp256"]
    rows = []
    for k in noise + sig:
        v = c[k]
        name = k if k in noise or k != sig[0] else r"\midrule " + k
        cross = f"{100 * v['median_flip_share_bad']:.0f}" if k in sig else "--"
        rows.append([name, f"{v['mean_corr']:.4f}", f"{v['spearman_model']:.3f}",
                     f"{v['best_stat_abs']:.3f}", cross])
    emit("The frozen-SAE test explained by TopK margins (GPT-2 layer 6, the "
         "frozen test's own SAE and tokens). model: Spearman correlation, over "
         "alive features, between predicted and measured frozen disagreement "
         "$1-r$; the prediction uses only the second moment of the codec's error, "
         "estimated on held-out tokens, and the SAE's margins on clean "
         "activations. statistic: the largest $|\\rho|$ of log firing rate, log "
         "mean or log peak activation with $1-r$. crossing \\%: median share of a "
         "degraded or damaged feature's squared error that comes from tokens "
         "where it is active on one side only (--: not reported for the "
         "noise-like codecs, which have at most 6 degraded or damaged "
         "features each). Below the rule, codecs that "
         "remove part of the signal rather than add noise. Sources: "
         "\\texttt{geom\\_margins.json}, \\texttt{geom\\_summary.json}.",
         "tab:margins", ["codec", "frozen mean $r$", "model $\\rho$",
                         "statistic $|\\rho|$", "crossing \\%"], rows)


def stability_prediction_table():
    """Seed stability and codec survival predicted from one SAE
    (scripts/geom_identifiability.py)."""
    d = j("runs_m1/eval/geom_identifiability.json")
    if not d or "readings" not in d:
        return
    R = d["regimes"]
    names = {"fp32_s0": "$1\\times$, $k=32$ (fit)", "fp32_b3x_s0": "$3\\times$",
             "fp32_b10x_s0": "$10\\times$", "fp32_freshb3x_s0": "$3\\times$ fresh text",
             "fp32_k16_s0": "$k=16$", "fp32_k64_s0": "$k=64$"}
    rows = []
    for k, nm in names.items():
        v = R[k]
        if k == "fp32_s0":
            a = (v["cv_auc_stats"], v["cv_auc_geom"], v["cv_auc_both"])
        else:
            t = v["transfer_auc"]
            a = (t["stats"], t["geom"], t["both"])
        rows.append([nm, f"{100 * v['stable_frac']:.1f}"] + [f"{x:.3f}" for x in a])
    for i, (c, v) in enumerate(R["fp32_s0"]["codec_survival_auc"].items()):
        nm = c.replace("_s0", "") + " survival"
        rows.append([(r"\midrule " if i == 0 else "") + nm, f"{100 * v['survived_frac']:.1f}",
                     f"{v['stats']:.3f}", f"{v['geom']:.3f}", f"{v['both']:.3f}"])
    emit("Predicting from one SAE which features retraining recovers (GPT-2 "
         "layer 6, AUC). Above the rule the target is seed stability (best "
         "match $r>0.9$ in both other seeds); the logistic model is fit once on "
         "the $1\\times$, $k=32$ SAE (its row is cross-validated) and applied "
         "unchanged to the other regimes' own SAEs. Below the rule, the same "
         "seed-fit model predicts which features survive retraining on each "
         "codec. positive \\%: seed-stable or retained share of alive features. "
         "statistics: log firing rate, log mean and log peak activation. "
         "geometry: co-firing share with the best partner, decoder coherence, "
         "co-firing of the nearest decoder atom, TopK margin relative to the "
         "activation, encoder-decoder cosine. Predictors use held-out eval "
         "tokens. Source: \\texttt{geom\\_identifiability.json}.",
         "tab:stabpredict", ["target", "positive \\%", "statistics",
                             "geometry", "both"], rows)


def dynamics_table():
    """When a perturbation reaches training decides its cost
    (scripts/geom_dynamics.py; numbers via scripts/geom_summary.py)."""
    g = j("runs_m1/eval/geom_summary.json")
    if not g or "lens2_dynamics" not in g:
        return
    d = g["lens2_dynamics"]
    rows = []
    for t, v in d["impulse_1e-7_one_batch"].items():
        after = "--" if v["dist_2000_after"] is None else f"{v['dist_2000_after']:.4f}"
        rows.append([f"$10^{{-7}}$, one batch" if t == "0" else "", f"{int(t):,}".replace(",", "{,}"),
                     f"{v['agreement_pct']:.1f}", f"{v['final_W_dec_dist']:.4f}", after])
    rows.append([r"\midrule one ulp, one value", "112, 4{,}970",
                 f"{d['oneulp_replay_agreement_pct']:.2f}", f"{d['oneulp_replay_final_W_dec_dist']:.4f}", "--"])
    rows.append(["$10^{-7}$, every batch", "all",
                 f"{d['noise7_every_batch_agreement_pct']:.1f}", f"{d['noise7_every_batch_W_dec_dist']:.4f}", "--"])
    emit("In this run, when a perturbation reaches training decides its cost (GPT-2 layer 6, "
         "seed 0, one run per row). Each perturbed run is a copy of the "
         "reference run, which reproduces Table~\\ref{tab:m1full}'s fp32 seed-0 SAE "
         "bit for bit, that receives the change at the listed step(s) and "
         "trains on the same batches. agreement: the paper's retained "
         "percentage against the reference. distance: relative Frobenius "
         "distance of the decoder from the reference's, at the end and 2,000 "
         "steps after the change. The one-ulp row replays the one-scalar "
         "experiment (its altered value enters training twice; the weights "
         "are bit-identical to the reference at every logged step from 130 "
         "until the second entry; optimizer moments were not compared) "
         "and reproduces that SAE bit for bit. The last row is not a copy: it "
         "is the separately trained noise7 SAE of Table~\\ref{tab:dose}, where "
         "every batch carries the change, scored against the same reference. "
         "Sources: "
         "\\texttt{geom\\_summary.json}, \\texttt{pair\\_geomdyn\\_*.json}.",
         "tab:dynamics", ["change", "step", "agreement \\%", "distance", "+2{,}000 steps"],
         rows)


if __name__ == "__main__":
    main_m1_table()
    m4_tables()
    transfer_table()
    jumprelu_table()
    sweep_budget_tables()
    probes_table()
    firing_rate_table()
    sparsity_invariance_table()
    forward_precision_table()
    entropy_throughput_tables()
    threshold_table()
    dose_response_table()
    dose_stratified_table()
    dose_margin_table()
    steering_table()
    rotation_ablation_table()
    consensus_table()
    stability_definition_table()
    proxy_table()
    fragility_table()
    # camera-ready additions go last so existing table numbers do not move
    feature_ablation_table()
    subspace_table()
    fresh_table()
    equal_bytes_table()
    timing_table()
    margins_table()
    stability_prediction_table()
    dynamics_table()
    out = ROOT / "paper" / "appendix_tables.tex"
    out.write_text("\n".join(OUT))
    print(f"wrote {out} ({len(OUT)} lines)")
