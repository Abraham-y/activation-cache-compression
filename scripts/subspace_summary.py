"""Collect scripts/subspace_recovery.py outputs into one table and apply
the pre-registered branches (A-E in that script's header).

Usage: python scripts/subspace_summary.py --root runs_m1
Writes <root>/eval/subspace_summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

ORDER = ["fp32_s0_r1", "sparseulp6_s0", "sparseulp6d1_s0", "sparseulp6d2_s0",
         "fp16_s0", "rot-int4_s0", "std-int4_s0", "int4_s0", "fp32_s1", "fp32_s2"]
STD_SEEDS = [("fp32_s0", "std-int4_s0"), ("fp32_s1", "std-int4_s1"),
             ("fp32_s2", "std-int4_s2")]
RESEEDS = [("fp32_s0", "fp32_s1"), ("fp32_s0", "fp32_s2"), ("fp32_s1", "fp32_s2")]
TIE = 0.02


def load(ev, ref, cand):
    p = ev / f"subspace_{ref}__{cand}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ms(vals):
    v = np.array(vals)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    args = ap.parse_args()
    ev = Path(args.root) / "eval"
    rows = {}
    print(f"{'cand (ref fp32_s0)':>18} {'S1':>6} {'S2':>6} {'S4':>6} {'S8':>6} "
          f"{'G4':>6} {'rec4|lost':>9} {'S4@.5':>6} {'nullS4':>7} {'EV64 x/w':>9} {'check':>6}")
    for c in ORDER:
        d = load(ev, "fp32_s0", c)
        if d is None:
            continue
        e = d["ev_set_level"].get("64")
        evr = e["ref_to_cand"] / e["within_ref"] if e else float("nan")
        rows[c] = {k: d[k] for k in ("S_1", "S_2", "S_4", "S_8", "G_4",
                                     "recovered_4_among_not_survived",
                                     "S_4_bar0.5", "S_4_null", "median_R2_4",
                                     "span_median_4", "span_median_4_null")}
        rows[c]["ev64_cross_over_within"] = evr
        rows[c]["ev64_random_over_within"] = (
            e["random_baseline"] / e["within_ref"] if e else float("nan"))
        rows[c]["check_pass"] = d["check"]["pass"]
        r = rows[c]
        print(f"{c:>18} {r['S_1']*100:6.1f} {r['S_2']*100:6.1f} {r['S_4']*100:6.1f} "
              f"{r['S_8']*100:6.1f} {r['G_4']*100:+6.1f} "
              f"{r['recovered_4_among_not_survived']*100:9.1f} {r['S_4_bar0.5']*100:6.1f} "
              f"{r['S_4_null']*100:7.2f} {evr:9.3f} {str(r['check_pass']):>6}")

    out = {"table": rows}
    std = [load(ev, *p) for p in STD_SEEDS]
    rs = [load(ev, *p) for p in RESEEDS]
    if all(std) and all(rs):
        for key in ("S_1", "S_4", "S_8"):
            out[f"std_int4_{key}"] = ms([d[key] for d in std])
            out[f"reseed_{key}"] = ms([d[key] for d in rs])
            out[f"gap_std_minus_reseed_{key}"] = (out[f"std_int4_{key}"][0]
                                                  - out[f"reseed_{key}"][0])
        print("\nthree seeds each (mean, sd):")
        for key in ("S_1", "S_4", "S_8"):
            a, b = out[f"std_int4_{key}"], out[f"reseed_{key}"]
            print(f"  {key}: std-int4 {a[0]*100:.1f} +- {a[1]*100:.1f}   "
                  f"reseed {b[0]*100:.1f} +- {b[1]*100:.1f}   "
                  f"gap {(a[0]-b[0])*100:+.1f} pp")

    # recoverable share of LOST features, every condition but the ceiling
    rec = {c: r["recovered_4_among_not_survived"] for c, r in rows.items()
           if c != "fp32_s0_r1"}
    non_int4 = [v for c, v in rec.items() if c != "int4_s0"]
    if non_int4:
        out["recovered_4_among_lost_range_excl_int4"] = [min(non_int4), max(non_int4)]
    ulp = [rows[c]["S_4"] for c in ("sparseulp6_s0", "sparseulp6d1_s0",
                                    "sparseulp6d2_s0") if c in rows]
    ulp1 = [rows[c]["S_1"] for c in ("sparseulp6_s0", "sparseulp6d1_s0",
                                     "sparseulp6d2_s0") if c in rows]
    if ulp:
        out["sparseulp6_S1_S4_mean"] = [float(np.mean(ulp1)), float(np.mean(ulp))]

    # pre-registered branches
    br = {}
    reseed_G = [rows[c]["G_4"] for c in ("fp32_s1", "fp32_s2") if c in rows]
    if reseed_G:
        g = float(np.mean(reseed_G))
        br["A"] = {"reseed_G4": g, "reading": "transfers" if g >= 0.20
                   else "does not transfer" if g < 0.05 else "partial"}
    ulp = [rows[c]["S_4"] for c in ("sparseulp6_s0", "sparseulp6d1_s0",
                                    "sparseulp6d2_s0") if c in rows]
    if ulp:
        s = float(np.mean(ulp))
        br["B"] = {"sparseulp6_S4_mean_over_draws": s,
                   "reading": "change of basis" if s >= 0.95
                   else "real change" if s < 0.85 else "indeterminate"}
    if "std-int4_s0" in rows and reseed_G:
        rs4 = float(np.mean([rows[c]["S_4"] for c in ("fp32_s1", "fp32_s2") if c in rows]))
        gap = rows["std-int4_s0"]["S_4"] - rs4
        br["C"] = {"std_minus_reseed_S4_seed0": gap,
                   "reading": "holds" if gap > TIE else "reverses" if gap < -TIE
                   else "indistinguishable"}
    if "std-int4_s0" in rows and "int4_s0" in rows:
        gap = rows["std-int4_s0"]["S_4"] - rows["int4_s0"]["S_4"]
        br["D"] = {"std_minus_int4_S4": gap, "pass": bool(gap > TIE)}
    br["E"] = {"max_null_S4": max(r["S_4_null"] for r in rows.values()),
               "pass": bool(max(r["S_4_null"] for r in rows.values()) < 0.01)}
    out["branches"] = br
    print("\nbranches:", json.dumps(br, indent=1))
    (ev / "subspace_summary.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {ev / 'subspace_summary.json'}")


if __name__ == "__main__":
    main()
