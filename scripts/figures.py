"""Publication figures, generated from eval JSONs. Venue-neutral.

Usage: PYTHONPATH=src python3 scripts/figures.py
Outputs docs/figures/*.png (300 dpi).

Design rules (dataviz): one axis per chart, categorical hues in fixed
slot order (blue #2a78d6, orange #eb6834, aqua #1baf7a), neutral gray for
reference bands, direct labels over legends where series <= 4, recessive
grid, text in ink colors not series colors.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from acc.utils import read_json  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a8a", "#333333"

CODECS = ["fp16", "rot-int8", "int8", "rot-int4", "int4"]


def bits_per_dim(root, codec):
    """Payload bits per dimension for this root's codec store. Read from
    the store manifest when present; if the cache has been reclaimed
    (scripts/cleanup.sh), fall back to the codec's exact byte formula
    with d recovered from a trained SAE checkpoint. The two agree
    exactly for every codec here (tests/test_codecs.py asserts measured
    == theoretical), so the fallback is not an approximation.
    NEVER a constant: an internal audit found hardcoded GPT-2 sizes putting
    Pythia points at 2.7x-wrong x positions."""
    p = ROOT / root / "cache" / f"{codec}-train" / "manifest.json"
    if p.exists():
        m = read_json(p)
        return m["payload_bytes_per_token"] * 8.0 / m["d"]
    from acc.codecs import theoretical_bytes_per_token
    s = ROOT / root / "saes" / "fp32_s0" / "summary.json"
    if s.exists():                       # cheap path: no weights needed
        d = read_json(s).get("cfg", {}).get("d_in")
    else:
        import torch
        ck = ROOT / root / "saes" / "fp32_s0" / "sae.pt"
        if not ck.exists():
            return None
        d = torch.load(ck, map_location="cpu", weights_only=True)["cfg"]["d_in"]
    if not d:
        return None
    return theoretical_bytes_per_token(codec, d) * 8.0 / d


def survived(root, ref, cand):
    p = ROOT / root / "eval" / f"pair_{ref}__{cand}.json"
    return read_json(p)["survived"] if p.exists() else None


def floor(root, tag=""):
    """Mean over the three seed pairs; None unless ALL THREE exist —
    a partial floor must never be silently presented as triangulated."""
    pairs = [("fp32%s_s0" % tag, "fp32%s_s1" % tag),
             ("fp32%s_s0" % tag, "fp32%s_s2" % tag),
             ("fp32%s_s1" % tag, "fp32%s_s2" % tag)]
    vals = [survived(root, a, b) for a, b in pairs]
    if any(v is None for v in vals):
        return None
    return float(np.mean(vals))


def fig_ratio_to_floor():
    """Headline: survival as a multiple of the seed floor, across models."""
    models = [("GPT-2 small L6", "runs_m1", BLUE, "_s0"),
              ("Pythia-1.4B L8", "runs_m4_L8", ORANGE, "_s0"),
              ("Pythia-1.4B L16", "runs_m4_L16", AQUA, "_s0")]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.grid(True, color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(1.0, color=GRAY, lw=1.2, ls="--", zorder=1)
    ax.annotate("seed-noise floor", xy=(0.985, 1.0), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color="#5a5a5a")
    plotted = 0
    # ROTATED and NAIVE pair up at IDENTICAL bits/dim (rot-int4 vs int4 at
    # 4, rot-int8 vs int8 at 8), so a single line through all of them would
    # draw a vertical jump at one x and read as an error bar. The rotated
    # family carries the line; the naive counterpart is a hollow marker at
    # the same x, joined by a dotted riser. The riser IS the claim: one
    # rotation, identical bytes.
    ROT = {"rot-int4": "int4", "rot-int8": "int8"}
    for mi, (name, root, color, _) in enumerate(models):
        fl = floor(root)
        if fl is None:
            continue
        pt = {}
        for c in CODECS:
            s = survived(root, "fp32_s0", f"{c}_s0")
            bpd = bits_per_dim(root, c)
            if s is None or bpd is None:
                continue
            if c == "fp16" and root.startswith("runs_m4"):
                continue  # degenerate bit-identical ceiling, not a codec point
            pt[c] = (bpd, s / fl)
        line = sorted(pt[c] for c in ("rot-int4", "rot-int8", "fp16") if c in pt)
        if not line:
            continue
        xs, ys = zip(*line)
        ax.plot(xs, ys, "-o", color=color, lw=1.6, ms=7, zorder=3, label=name)
        for rot, naive in ROT.items():
            if rot not in pt or naive not in pt:
                continue
            (x, y_rot), (_, y_nv) = pt[rot], pt[naive]
            ax.plot([x, x], [y_nv, y_rot], color=color, lw=1.0, ls=":",
                    zorder=2)
            ax.scatter([x], [y_nv], s=58, facecolors="none", edgecolors=color,
                       linewidths=1.6, zorder=3)
        plotted += 1
    ax.scatter([], [], s=58, color="#555555", label="fp16 and rotated (rot-int$n$)")
    ax.scatter([], [], s=58, facecolors="none", edgecolors="#555555",
               linewidths=1.6, label="naïve per-token (int$n$)")
    ax.set_xscale("log", base=2)
    # every series shares the same three rates; tick them directly rather
    # than leaving 2^k gridlines that sit a hair off the measured payload
    # nominal bits, deduped: the per-token scale makes the measured payload
    # 4.02-4.04 bits/dim depending on d, which would double every tick
    rates = sorted({round(bits_per_dim(r, c))
                    for _, r, _, _ in models for c in ("rot-int4", "rot-int8")
                    if bits_per_dim(r, c)} | {16})
    ax.set_xticks(rates, minor=False)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([f"{v:.0f}" for v in rates])
    ax.set_xlim(3.4, 19.5)
    ax.set_xlabel("bits per dimension (measured payload)")
    ax.set_ylabel("feature survival, multiple of seed floor")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    ax.set_title("Codec ordering holds across model scale; levels do not",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "ratio_to_floor.png")
    print(f"ratio_to_floor.png ({plotted} models)")


def fig_saturation():
    """The paper's headline: agreement is discontinuous at bit-identity.
    The sparse rung (one ulp on 1e-6 of entries, 1.6e-12) sits five decades
    below anything the dense ladder reaches and still costs 30 points, so
    the drop is not a curve approaching a ceiling -- it is a step, then a
    shallow decline. Plotted at the left edge with a break, since a log
    axis cannot show both it and zero."""
    doses = ["noise7", "noise6", "noise5", "noise4", "noise3", "noise2"]
    xs, ys = [], []
    for t in doses:
        fr = read_json(ROOT / f"runs_m1/eval/frozen_{t}__fp32_s0.json")
        pr = read_json(ROOT / f"runs_m1/eval/pair_fp32_s0__{t}_s0.json")
        if fr is None or pr is None:
            return
        xs.append(fr["direct"]["rel_l2_mean"]); ys.append(100 * pr["survived"])
    codecs = []
    for c in ("fp16", "rot-int8", "rot-int4", "int4"):
        fr = read_json(ROOT / f"runs_m1/eval/frozen_{c}__fp32_s0.json")
        pr = read_json(ROOT / f"runs_m1/eval/pair_fp32_s0__{c}_s0.json")
        if fr and pr:
            codecs.append((c, fr["direct"]["rel_l2_mean"], 100 * pr["survived"]))
    fl = 100 * floor("runs_m1")
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=300)
    ax.grid(True, color="#e6e6e6", lw=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xscale("log")
    LO = 2e-8
    ax.axvspan(2e-8, 6e-8, color="#f0f0f0", zorder=0)
    ax.annotate("no DENSE codec\nbelow one ulp", xy=(3.4e-8, 86),
                ha="center", va="center", fontsize=7.5, color="#6a6a6a")
    # ceiling: bit-exact, at zero perturbation (drawn at the axis edge)
    ax.axhline(100, color=GRAY, lw=1.0, ls="--", zorder=1)
    ax.annotate("bit-exact ceiling (zero perturbation) = 100%",
                xy=(0.99, 100), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color="#5a5a5a")
    ax.axhline(fl, color=GRAY, lw=1.0, ls=":", zorder=1)
    ax.annotate(f"seed floor = {fl:.0f}%", xy=(0.99, fl),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=8, color="#5a5a5a")
    ax.plot(xs, ys, "-o", color=BLUE, lw=1.6, ms=5, zorder=3,
            label="calibrated noise (fp32 both sides)")
    ax.plot([c[1] for c in codecs], [c[2] for c in codecs], "s--",
            color=ORANGE, lw=1.2, ms=5, mfc="white", zorder=3,
            label="real codecs")
    for name, x, y in codecs:
        ax.annotate(name, xy=(x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=7.5, color="#8a4b10")
    ax.annotate("", xy=(xs[0], ys[0] + 3), xytext=(1e-4, ys[3] + 3),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.9))
    ax.annotate(f"3 decades, {ys[0]-ys[3]:.1f} pp", xy=(3e-6, ys[0] + 5),
                ha="center", fontsize=8, color=BLUE)
    ax.annotate("matched-magnitude noise\ntracks the codecs here",
                xy=(1.1e-3, 30), ha="center", fontsize=7.5, color="#4a4a4a")
    # one scalar in the whole corpus: same magnitude regime, no cost --
    # the comparison that shows the response is governed by count
    o_fr = ROOT / "runs_m1/eval/frozen_oneulp__fp32_s0.json"
    o_pr = ROOT / "runs_m1/eval/pair_fp32_s0__oneulp_s0.json"
    if o_fr.exists() and o_pr.exists():
        ox = read_json(o_fr)["direct"]["rel_l2_mean"]
        oy = 100 * read_json(o_pr)["survived"]
        ax.plot([ox], [oy], marker="*", ms=15, color=GRAY, zorder=5,
                mfc="white", mew=1.5)
        ax.annotate(f"1 ulp, ONE scalar\n{oy:.2f}%", xy=(ox, oy),
                    xytext=(4, -26), textcoords="offset points",
                    ha="left", fontsize=8, color="#444444")
    # the sparse rung: escapes the dense ulp bound
    sp_fr = ROOT / "runs_m1/eval/frozen_sparseulp6__fp32_s0.json"
    sp_pr = ROOT / "runs_m1/eval/pair_fp32_s0__sparseulp6_s0.json"
    if sp_fr.exists() and sp_pr.exists():
        sx = read_json(sp_fr)["direct"]["rel_l2_mean"]
        sy = 100 * read_json(sp_pr)["survived"]
        LO = min(sx, ox if o_fr.exists() else sx) / 4
        ax.plot([sx], [sy], marker="D", ms=7, color=BLUE, zorder=4,
                mfc="white", mew=1.8)
        ax.annotate(f"1 ulp on $10^{{-6}}$ of entries\n{sy:.1f}%",
                    xy=(sx, sy), xytext=(0, 13), textcoords="offset points",
                    ha="left", fontsize=8, color=BLUE)
        ax.plot([sx, xs[0]], [sy, ys[0]], ls=":", lw=1.2, color=BLUE, zorder=2)
    ax.set_xlim(LO, 1.0)
    ax.set_ylim(0, 108)
    ax.set_xlabel("perturbation size, mean relative $L_2$")
    ax.set_ylabel("features surviving a retrain (r > 0.9)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "saturation.png")
    plt.close(fig)
    print("wrote saturation.png")


def fig_frozen_transfer():
    """The headline: the frozen proxy orders codecs perfectly inside a
    site and carries no scale between them.

    This replaces the old frozen-vs-retrained dumbbell, which was drawn
    to show the thresholded readout saturating. That framing did not
    survive audit — the CONTINUOUS statistic ranks the codecs at
    rho = 1.000. Plotting frozen r against retrained survival shows both
    facts at once: every series is monotone (it ranks), the viable
    codecs crowd into the last 2% of the x-axis (no resolution where the
    decision is made), and the three series are vertically offset (the
    same reading means different things at different sites)."""
    sites = [("GPT-2 small L6", "runs_m1", BLUE),
             ("Pythia-1.4B L8", "runs_m4_L8", ORANGE),
             ("Pythia-1.4B L16", "runs_m4_L16", AQUA)]
    codecs = ["fp16", "rot-int8", "int8", "rot-int4", "int4"]
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=300)
    ax.grid(True, color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for name, root, color in sites:
        pts = []
        for c in codecs:
            f = ROOT / root / "eval" / f"frozen_{c}__fp32_s0.json"
            p_ = ROOT / root / "eval" / f"pair_fp32_s0__{c}_s0.json"
            if not (f.exists() and p_.exists()):
                continue
            fr = read_json(f)["mean_corr"]
            if fr >= 1.0:      # bit-lossless: log axis undefined, and it
                continue       # is a ceiling point, not a codec measurement
            pts.append((1 - fr, read_json(p_)["survived"], c))
        if not pts:
            continue
        pts.sort()
        xs, ys, labels = zip(*pts)
        ax.plot(xs, ys, "-o", color=color, lw=1.6, ms=6.5, zorder=3, label=name)
    # the matched-reading band: rot-int8 reads the same at all three sites
    r8 = []
    for _, root, _ in sites:
        f = ROOT / root / "eval" / "frozen_rot-int8__fp32_s0.json"
        p_ = ROOT / root / "eval" / "pair_fp32_s0__rot-int8_s0.json"
        if f.exists() and p_.exists():
            r8.append((1 - read_json(f)["mean_corr"], read_json(p_)["survived"]))
    if len(r8) == 3:
        x = sum(v[0] for v in r8) / 3
        lo, hi = min(v[1] for v in r8), max(v[1] for v in r8)
        ax.annotate("", xy=(x, lo), xytext=(x, hi),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
        ax.annotate(f"same frozen reading\n{100*(hi-lo):.1f} points apart",
                    xy=(x, (lo + hi) / 2), xytext=(13, 0),
                    textcoords="offset points", fontsize=8.5, color=INK,
                    va="center")
    for name, root, color in sites:
        fl = floor(root)
        if fl is not None:
            ax.axhline(fl, color=color, lw=0.9, ls=":", zorder=1, alpha=0.75)
    ax.annotate("each site's seed floor", xy=(0.985, 0.155),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=8, color="#5a5a5a")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("frozen-basis disagreement, $1-r$  (better $\\rightarrow$)")
    ax.set_ylabel("features surviving a retrain (r > 0.9)")
    ax.set_ylim(0, 0.68)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("The proxy ranks perfectly and fixes no level", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "frozen_transfer.png")
    print("frozen_transfer.png")


def fig_frozen_vs_retrained():
    """The construct-validity gap, as a dumbbell per codec (GPT-2)."""
    rows = []
    for c in ["fp16", "int8", "int6", "int4-o16", "int4"]:
        fz = ROOT / "runs_m1" / "eval" / f"frozen_{c}__fp32_s0.json"
        rt = survived("runs_m1", "fp32_s0", f"{c}_s0")
        if fz.exists() and rt is not None:
            rows.append((c, read_json(fz)["survived"], rt))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=300)
    ax.grid(True, axis="x", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ys = np.arange(len(rows))[::-1]
    for y, (c, f, r) in zip(ys, rows):
        ax.plot([r, f], [y, y], color="#cccccc", lw=2, zorder=2)
        ax.scatter([f], [y], s=52, color=GRAY, zorder=3)
        ax.scatter([r], [y], s=52, color=BLUE, zorder=3)
    ax.scatter([], [], s=52, color=GRAY, label="frozen-judge verdict")
    ax.scatter([], [], s=52, color=BLUE, label="retrained reality")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("fraction of features survived (r > 0.9)")
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("A precise judge, measuring the wrong construct", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "frozen_vs_retrained.png")
    print("frozen_vs_retrained.png")


def fig_stratified():
    """Survival among seed-stable vs seed-unstable features (GPT-2)."""
    st = read_json(ROOT / "runs_m1" / "eval" / "stability_fp32_s0_t0p9.json")
    rows = {r["cand"]: r for r in st["rows"]}
    codecs = ["fp16_s0", "rot-int8_s0", "rot-int4_s0", "int4_s0"]
    stable = [rows[c]["survived_stable"] for c in codecs]
    unstable = [rows[c]["survived_unstable"] for c in codecs]
    x = np.arange(len(codecs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=300)
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.bar(x - w / 2, stable, w, color=BLUE, zorder=3, label="seed-stable features")
    ax.bar(x + w / 2, unstable, w, color="#b8cfeb", zorder=3,
           label="seed-unstable features")
    for xi, v in zip(x - w / 2, stable):
        ax.annotate(f"{v:.0%}", xy=(xi, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_s0", "") for c in codecs], fontsize=9)
    ax.set_ylabel("survival under codec")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Codec damage concentrates on seed-unstable features", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "stratified.png")
    print("stratified.png")


def fig_budget_scaling():
    """Fixed-size stable set: rotated codecs improve with budget, fp16 is flat."""
    d = read_json(ROOT / "runs_m1" / "eval" / "budget_fixedset.json")
    se = d["binomial_se_pp"] / 100.0
    order = ["1x", "3x", "10x"]
    xs = [d["tiers"][k]["token_visits_M"] for k in order]
    # Camera-ready (10 Oct): same aspect as before (prints at the same size
    # at 0.75\linewidth) on a smaller canvas so the fonts print near 7 pt;
    # end labels take their line's colour and sit beside their own point
    # (fp16 and rot-int8 end 0.005 apart, so they are nudged apart).
    fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=300)
    ax.grid(True, color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for cname, color, dy in (("fp16", BLUE, -7), ("rot-int8", ORANGE, 6),
                             ("rot-int4", AQUA, 0)):
        ys = [d["tiers"][k][cname] for k in order]
        ax.errorbar(xs, ys, yerr=se, fmt="-o", color=color, lw=1.4, ms=5,
                    capsize=2.5, elinewidth=1.0, zorder=3)
        ax.annotate(cname, xy=(xs[-1], ys[-1]), xytext=(8, dy),
                    textcoords="offset points", fontsize=8, color=color,
                    va="center", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v}M" for v in xs])
    ax.minorticks_off()
    ax.set_xlim(26, 700)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("training budget (token-visits per SAE, log scale)", fontsize=8.5)
    ax.set_ylabel("survival among the top-%d\nstable features" % d["n_fixed"],
                  fontsize=8.5)
    ax.set_title("Fixed stable set: rotated codecs improve with budget, "
                 "fp16 is flat", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "budget_scaling.png")
    print("budget_scaling.png")


def fig_headline():
    """Fig. 1: the whole argument on one axis (GPT-2 L6, seed 0).
    Perturbation rungs (blue, matching fig_saturation's noise hue) and
    codecs (orange) on the common scale of retrained agreement, with the
    seed floor as the calibration line. The 100.0 ceiling row is the
    bit-exact same-seed retrain (sec:protocol), exact by construction;
    every other row is read from its pair/frozen JSONs."""
    def fmt_rel(v):
        if v == 0:
            return "0"
        if v < 1e-2:
            e = int(np.floor(np.log10(v)))
            m = f"{v / 10**e:.1f}".rstrip("0").rstrip(".")
            return rf"${m}{{\times}}10^{{{e}}}$"
        return f"{v:.2f}"

    rows = [("identical bytes (ceiling)", None, GRAY),
            ("one ulp on one scalar", "oneulp", BLUE),
            ("one ulp on a millionth", "sparseulp6", BLUE),
            ("fp16: every value rounds", "fp16", ORANGE),
            ("std-int4 (standardized)", "std-int4", ORANGE),
            ("int4 (naive per token)", "int4", ORANGE)]
    vals, rels = [], []
    for _, tag, _ in rows:
        if tag is None:
            vals.append(100.0)
            rels.append(0.0)
            continue
        pr = read_json(ROOT / f"runs_m1/eval/pair_fp32_s0__{tag}_s0.json")
        fr = read_json(ROOT / f"runs_m1/eval/frozen_{tag}__fp32_s0.json")
        if pr is None or fr is None:
            return
        vals.append(100 * pr["survived"])
        rels.append(fr["direct"]["rel_l2_mean"])
    fl = 100 * floor("runs_m1")

    # Camera-ready (10 Oct): printed at full column width (5.5 in) instead
    # of 0.4\linewidth, so the canvas is drawn at its printed size and the
    # fonts print at about 7 pt (the old 6 in canvas shrunk to 2.2 in
    # printed them near 3.7 pt). The colour legend moved to the caption.
    fig, ax = plt.subplots(figsize=(5.5, 1.22), dpi=300)
    ax.grid(True, axis="x", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ys = np.arange(len(rows))
    ax.barh(ys, vals, height=0.66, color=[c for _, _, c in rows], zorder=3)
    for y, v in zip(ys, vals):
        if v >= 99:
            lbl = "100.0%" if v == 100 else f"{v:.2f}%"
            ax.annotate(lbl, xy=(v - 0.8, y), va="center", ha="right",
                        fontsize=7, color="white")
        else:
            ax.annotate(f"{v:.1f}%", xy=(v + 0.8, y), va="center", ha="left",
                        fontsize=7, color=INK,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.85, pad=0.4))
    # rel-L2 column, right margin
    for y, r in zip(ys, rels):
        ax.annotate(fmt_rel(r), xy=(1.25, y), xycoords=("axes fraction", "data"),
                    va="center", ha="right", fontsize=7, color="#555555")
    ax.annotate("rel. $L_2$", xy=(1.25, -0.95), xycoords=("axes fraction", "data"),
                va="center", ha="right", fontsize=7, color="#555555",
                annotation_clip=False)
    ax.axvline(fl, color=GRAY, lw=0.9, ls="--", zorder=2)
    ax.annotate(f"seed floor {fl:.1f}% (reseed: zero bytes changed)",
                xy=(fl + 0.8, -0.95), ha="left", va="center", fontsize=6.5,
                color="#5a5a5a")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(5.5, -1.45)
    ax.set_xlim(0, 100)
    ax.tick_params(axis="x", labelsize=6.5, length=2, width=0.6, pad=1.5)
    ax.set_xlabel("features surviving a retrain (best match $r>0.9$), % of alive",
                  fontsize=7, labelpad=2)
    fig.subplots_adjust(left=0.255, right=0.80, top=0.995, bottom=0.235)
    fig.savefig(OUT / "headline.png")
    plt.close(fig)
    print("headline.png")


if __name__ == "__main__":
    fig_headline()
    fig_ratio_to_floor()
    fig_saturation()
    fig_frozen_transfer()
    fig_frozen_vs_retrained()
    fig_stratified()
    fig_budget_scaling()
