"""Every number the paper quotes from Appendix J (frozen-test margins,
scripts/geom_margins.py; seed stability, scripts/geom_identifiability.py;
training dynamics, scripts/geom_dynamics.py), computed in one place from
their outputs.

Usage: python scripts/geom_summary.py
Writes runs_m1/eval/geom_summary.json
"""
import json
from pathlib import Path

import numpy as np

EV = Path(__file__).resolve().parent.parent / "runs_m1" / "eval"
NOISE_LIKE = ["fp16", "bf16", "int8", "int8-o8", "int6", "int6-o8", "rot-int8",
              "rot-int6", "rot-int4", "std-int4", "int4-o16"]
SIGNAL = ["int4", "pca256", "rp256"]


def rng(v):
    v = [float(x) for x in v]
    return {"min": min(v), "max": max(v), "median": float(np.median(v))}


def _rho(ref, other):
    from scipy.stats import spearmanr
    z = np.load(EV / f"geom_ident_{ref}.npz")
    return float(spearmanr(z["margin_rel"], z[other]).statistic)


def _replay_course(early):
    rows = [(r["step"], r["runs"]["oneulp_replay"]["exactly_equal"])
            for r in early if "oneulp_replay" in r["runs"]]
    eq = [st for st, e in rows if e and st < 4970]
    return {"differs_at": [st for st, e in rows if not e and st < 4970],
            "bit_identical_again_from": min(eq), "bit_identical_until": max(eq),
            "next_difference": min(st for st, e in rows if not e and st > 4970)}


def main():
    m = json.loads((EV / "geom_margins.json").read_text())
    c = m["codecs"]
    lens1 = {
        "noise_like_spearman": rng(c[k]["spearman_model"] for k in NOISE_LIKE),
        "noise_like_spearman_hetero": rng(c[k]["spearman_model_hetero"] for k in NOISE_LIKE),
        "noise_like_best_stat_abs": rng(c[k]["best_stat_abs"] for k in NOISE_LIKE),
        "noise_like_wins": sum(abs(c[k]["spearman_model"]) > c[k]["best_stat_abs"] for k in NOISE_LIKE),
        "n_noise_like": len(NOISE_LIKE),
        "partial_peak_abs_median": float(np.median([abs(c[k]["partial_peak_given_model"]) for k in NOISE_LIKE])),
        "signal_spearman": {k: c[k]["spearman_model"] for k in SIGNAL},
        "signal_best_stat_abs": {k: c[k]["best_stat_abs"] for k in SIGNAL},
        "signal_crossing_share_bad": {k: c[k]["median_flip_share_bad"] for k in SIGNAL},
        "signal_n_bad": {k: c[k]["n_degraded_or_damaged"] for k in SIGNAL},
        "identity_max_abs_diff": max(c[k]["identity_max_abs_diff"] for k in c),
        "readings": m["readings"],
    }
    d = json.loads((EV / "geom_identifiability.json").read_text())
    R = d["regimes"]
    s0 = R["fp32_s0"]
    others = [k for k in R if k != "fp32_s0"]
    best = {k: max(v["single_auc_stable"], key=v["single_auc_stable"].get) for k, v in R.items()}
    lens3 = {
        "n_alive": s0["n_alive"], "stable_pct": 100 * s0["stable_frac"],
        "cv_auc": {"stats": s0["cv_auc_stats"], "geom": s0["cv_auc_geom"], "both": s0["cv_auc_both"]},
        "single_auc": s0["single_auc_stable"],
        "cofire_auc_for_unstable": 1 - s0["single_auc_stable"]["cofire_share"],
        # near-parallel atoms: no threshold was set for this one
        "coherence_auc_for_unstable": 1 - s0["single_auc_stable"]["coherence"],
        "geom_minus_stats": s0["cv_auc_geom"] - s0["cv_auc_stats"],
        "transfer": {key: rng(R[k]["transfer_auc"][key] for k in others)
                     for key in ("geom", "stats", "both")},
        "codec_survival": {key: rng(v[key] for v in s0["codec_survival_auc"].values())
                           for key in ("geom", "stats", "both")},
        "margin_best_single_in": sum(b == "margin_rel" for b in best.values()),
        "n_regimes": len(R),
        "best_single": best,
        # how closely the TopK margin tracks activation size, 1x k=32 SAE
        "margin_vs_log_mean_act_spearman": _rho("fp32_s0", "log_mean_act"),
        "margin_vs_log_peak_act_spearman": _rho("fp32_s0", "log_peak_act"),
        "readings": d["readings"],
    }
    J = lambda n: json.loads((EV / n).read_text())
    late, early, p1 = J("geom_dynamics_late.json"), J("geom_dynamics_early.json"), J("geom_dynamics_phase1.json")
    pair = lambda c: J(f"pair_geomdyn_A_s0__geomdyn_{c}_s0.json")["survived"]
    logs = {f: [json.loads(l) for l in open(EV / f"geom_dynamics_{f}.jsonl")]
            for f in ("phase1", "late", "early")}

    def dist(f, run, step):
        for r in logs[f]:
            if r["step"] == step and run in r["runs"]:
                return r["runs"][run]["W_dec"]

    t0s = [0, 4000, 6000, 7000, 7500, 7800, 7950]
    impulses = {}
    for t in t0s:
        f = "early" if t == 0 else "late"
        impulses[str(t)] = {"agreement_pct": 100 * pair(f"d7_t{t}"),
                            "final_W_dec_dist": dist(f, f"d7_t{t}", 8000),
                            "dist_2000_after": dist(f, f"d7_t{t}", t + 2000) if t + 2000 <= 8000 else None}
    hk = {h["step"]: h for h in early["hooks"] + late["hooks"]}
    i4 = early["I4_d7_t0_equals_phase1_dose7"]
    wd = J("weight_divergence.json")["arms"]
    lens2 = {
        "I1_reproduces_fp32_s0": p1["I1_reproduces_fp32_s0"],
        "I1b_replay_reproduces_oneulp_s0": early["I1b_replay_reproduces_oneulp_s0"],
        "I3": [late["I3_late_A_reproduces_phase1_A"], early["early_A_reproduces_phase1_A"]],
        "I4_logged_steps": i4["n_steps"], "I4_all_equal": i4["all_equal"],
        # where the step-0 rerun's logged distances differ from phase 1's
        "I4_mismatch": [
            {"step": st, "W_dec": [a["W_dec"], b["W_dec"]], "W_enc_equal": a["W_enc"] == b["W_enc"]}
            for st, a, b in (
                (r["step"], r["runs"]["dose7"], q["runs"]["d7_t0"])
                for r in logs["phase1"] for q in logs["early"]
                if r["step"] == q["step"] and "dose7" in r["runs"] and "d7_t0" in q["runs"])
            if a["W_dec"] != b["W_dec"] or a["W_enc"] != b["W_enc"]],
        "absorption_changed_pct": {"step0": 100 * hk[0]["frac_changed"],
                                   "step4000": 100 * hk[4000]["frac_changed"]},
        # one_step_check: does the state differ ONE step after each entry?
        "oneulp_entries_change_state_after_one_step": [hk[112]["changes_state"], hk[4970]["changes_state"]],
        # what the first entry did afterwards (replay fork vs the reference)
        "oneulp_replay_logged": _replay_course(logs["early"]),
        "impulse_1e-7_one_batch": impulses,
        # A3 (ii): |agree(4000) - agree(0)|, against the pre-set 5 pp
        "A3_ii_gap_pp": impulses["4000"]["agreement_pct"] - impulses["0"]["agreement_pct"],
        "oneulp_replay_agreement_pct": 100 * pair("oneulp_replay"),
        "oneulp_replay_final_W_dec_dist": dist("early", "oneulp_replay", 8000),
        "noise7_every_batch_agreement_pct": 100 * J("pair_fp32_s0__noise7_s0.json")["survived"],
        "noise7_every_batch_W_dec_dist": wd["noise7_s0"]["W_dec"]["rel_frobenius"],
        "train_hours": sum(logs[f][-1]["elapsed_s"] for f in logs) / 3600,
    }
    out = {"lens1_frozen_margins": lens1, "lens3_stability": lens3, "lens2_dynamics": lens2}
    (EV / "geom_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
