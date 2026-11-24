"""TopK cells and margins, for the frozen-SAE test (Appendix J).

PRE-REGISTERED READINGS (fixed 11 Oct 2026, before this script existed):
  H1a  Over the 11 noise-like codecs (fp16, bf16, int8, int8-o8, int6,
       int6-o8, rot-int8, rot-int6, rot-int4, std-int4, int4-o16): median
       Spearman of predicted vs measured 1-r over alive features >= 0.5 ->
       "margins plus encoder-projected error explain frozen damage"; < 0.3
       -> the Gaussian margin model does not explain it; between -> partial.
       Secondary: a heteroscedastic variant scaling Sigma_e by ||e_t||^2.
  H1b  The model beats the best single fragility statistic (|Spearman| of
       log rate, log mean act, log peak act with 1-r) in >= 9 of the 11.
  H1c  Share of each feature's squared activation error from tokens where
       it is active on exactly one side. For int4, pca256, rp256, among
       degraded or damaged features: median share >= 0.5 -> "wall crossings
       dominate"; <= 0.3 -> "in-cell distortion dominates"; else mixed.
  H1d  Peak activation is subsumed: median over the 11 of |partial
       Spearman(log peak act, 1-r | model prediction)| < 0.1.
  Expectation, not a test: for int4/pca256/rp256 the Gaussian model is worse.
OUTCOME (11 Oct): H1a pass (median 0.980), H1b pass (11/11), H1c mixed
(0.84 / 0.46 / 0.57), H1d fail (0.113).

A TopK SAE is piecewise affine: inside a polyhedral cell (bounded by the
ties z_i = z_j) the active set is fixed and each active feature moves
linearly with its pre-activation. A codec error e moves the pre-activations
by W_enc^T e. A feature's frozen activation then changes continuously
inside the cell and jumps where the token crosses a wall: it drops out when
it falls below the token's runner-up (the (k+1)-th pre-activation), and a
silent feature drops in when it rises past the token's weakest active one.

Margin model (H1a), second-order codec statistics only: e ~ N(0, M_e) with
M_e = E[e e^T] estimated on HELD-OUT eval tokens 500k-550k, so
  in-cell variance        s_i^2       = w_i' M_e w_i
  wall-crossing variance  sigma_ij^2  = (w_i - w_j)' M_e (w_i - w_j)
  P(drop out of i | t)    = Phi(-(z_i - z_runner) / sigma_i,runner)
  P(drop in of i | t)     = Phi(-(z_weakest - z_i) / sigma_i,weakest)
  predicted error energy  V_i = mean_t[ active: (1-P_out) s_i^2 + P_out a_i^2 ;
                                        silent: P_in max(z_weakest, 0)^2 ]
and the prediction is V_i / Var(a_i) (monotone in predicted 1 - r). The
heteroscedastic variant scales M_e by each token's ||e_t||^2 / E||e||^2.
H1c decomposes the ACTUAL squared activation error into tokens where the
feature is active on both sides (in-cell) or on exactly one (a crossing).
Identity check: the per-feature Pearson r reproduced here must equal
frozen_<codec>__fp32_s0.npz to 1e-9, or the script exits 1.

Usage: PYTHONPATH=src python scripts/geom_margins.py [codec ...]
Writes runs_m1/eval/geom_margins_<codec>.npz and geom_margins.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import geom_common as G  # noqa: E402
from acc.metrics import StreamingPaired  # noqa: E402

NOISE_LIKE = ["fp16", "bf16", "int8", "int8-o8", "int6", "int6-o8", "rot-int8",
              "rot-int6", "rot-int4", "std-int4", "int4-o16"]
SIGNAL = ["int4", "pca256", "rp256"]
REF = "fp32_s0"
CAL = (500_000, 550_000)


def paired_range(codec, lo, hi, chunk=8192):
    """Aligned (fp32, codec) eval chunks over [lo, hi), shard by shard and
    chunked from each shard's start exactly as cmd_eval_frozen does."""
    ga = G.iter_range("master-eval", lo, hi, chunk)
    gb = G.iter_range(f"{codec}-eval", lo, hi, chunk)
    for (sa, xa), (sb, xb) in zip(ga, gb):
        assert sa == sb and xa.shape == xb.shape
        yield xa, xb


@torch.no_grad()
def error_moment(codec, dev):
    M = torch.zeros(768, 768, dtype=torch.float64)
    n, e2 = 0, 0.0
    for xa, xb in paired_range(codec, *CAL):
        e = torch.from_numpy(xb - xa).double()
        M += e.T @ e
        e2 += float((e * e).sum())
        n += e.shape[0]
    return (M / n).float().to(dev), e2 / n


def dense(vals, idx, d):
    z = torch.zeros(vals.shape[0], d, device=vals.device)
    z.scatter_(1, idx, vals)
    return z


@torch.no_grad()
def run_codec(codec, s, dev):
    d, k = s.d_sae, s.k
    M, e2_mean = error_moment(codec, dev)
    W = s.W_enc.detach()                       # [768, d_sae]
    Q = W.T @ (M @ W)                          # [d_sae, d_sae]
    q = torch.diagonal(Q).clone()
    paired = StreamingPaired(d)
    acc = {n: np.zeros(d) for n in ("V", "Vh", "sse_cont", "sse_flip",
                                    "n_out", "n_in", "sum_a", "sum_a2", "fire")}
    peak = np.zeros(d)
    N = 0
    for xa, xb in paired_range(codec, *G.SCORE):
        xa_t = torch.from_numpy(xa).to(dev)
        xb_t = torch.from_numpy(xb).to(dev)
        za = dense(*s.feature_acts(xa_t), d)
        zb = dense(*s.feature_acts(xb_t), d)
        paired.update(za.cpu().numpy(), zb.cpu().numpy())      # == cmd_eval_frozen
        err = zb - za
        act_a, act_b = za > 0, zb > 0
        acc["sse_cont"] += ((err * err) * (act_a & act_b)).sum(0).cpu().double().numpy()
        acc["sse_flip"] += ((err * err) * (act_a ^ act_b)).sum(0).cpu().double().numpy()
        acc["n_out"] += (act_a & ~act_b).sum(0).cpu().double().numpy()
        acc["n_in"] += (~act_a & act_b).sum(0).cpu().double().numpy()
        acc["sum_a"] += za.sum(0).cpu().double().numpy()
        acc["sum_a2"] += (za * za).sum(0).cpu().double().numpy()
        acc["fire"] += act_a.sum(0).cpu().double().numpy()
        peak = np.maximum(peak, za.max(0).values.cpu().numpy())
        g_all = ((xb_t - xa_t) ** 2).sum(1) / e2_mean            # heteroscedastic scale
        for h0 in range(0, xa_t.shape[0], 4096):
            x = xa_t[h0:h0 + 4096]
            g = g_all[h0:h0 + 4096]
            n = x.shape[0]
            z = s.pre_acts(x)
            v, idx = torch.topk(z, k + 1, dim=-1)
            S, r, w = idx[:, :k], idx[:, k], idx[:, k - 1]
            zS, zr, zw = v[:, :k], v[:, k], v[:, k - 1]
            aS = zS.clamp(min=0)
            sig_out = (q[S] + q[r][:, None] - 2 * Q[S, r[:, None]]).clamp(min=1e-20)
            Qw = Q.index_select(1, w).T                          # [n, d]
            sig_in = (q[None, :] + q[w][:, None] - 2 * Qw).clamp(min=1e-20)
            margin_out = zS - zr[:, None]
            margin_in = zw[:, None] - z
            mask_in = torch.ones(n, d, device=dev, dtype=torch.bool)
            mask_in.scatter_(1, S, False)
            enter = zw.clamp(min=0) ** 2
            for key, scale in (("V", torch.ones_like(g)), ("Vh", g)):
                p_out = torch.special.ndtr(-margin_out / (sig_out * scale[:, None]).sqrt())
                p_in = torch.special.ndtr(-margin_in / (sig_in * scale[:, None]).sqrt())
                p_in = p_in * mask_in
                contrib_S = ((1 - p_out) * q[S] * scale[:, None] + p_out * aS * aS) * (aS > 0)
                tot = torch.zeros(d, device=dev)
                tot.scatter_add_(0, S.reshape(-1), contrib_S.reshape(-1))
                tot += (p_in * enter[:, None]).sum(0)
                acc[key] += tot.cpu().double().numpy()
        N += xa.shape[0]
    corr = paired.pearson()
    var_a = acc["sum_a2"] / N - (acc["sum_a"] / N) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_act = acc["sum_a"] / np.maximum(acc["fire"], 1)
        flip_share = acc["sse_flip"] / (acc["sse_flip"] + acc["sse_cont"])
    return dict(corr=corr, var_a=var_a, pred=acc["V"] / N / var_a,
                pred_h=acc["Vh"] / N / var_a, flip_share=flip_share,
                rate=acc["fire"] / N, mean_act=mean_act, peak=peak,
                out_rate=acc["n_out"] / np.maximum(acc["fire"], 1),
                in_count=acc["n_in"], e2_mean=e2_mean, n_tokens=N)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    c = np.corrcoef([rx, ry, rz])
    rxy, rxz, ryz = c[0, 1], c[0, 2], c[1, 2]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def main():
    codecs = sys.argv[1:] or NOISE_LIKE + SIGNAL
    dev = G.device()
    s, _ = G.sae(REF, dev)
    path = G.EVAL / "geom_margins.json"
    out = json.loads(path.read_text()) if path.exists() else {"codecs": {}}
    for c in codecs:
        r = run_codec(c, s, dev)
        ref = np.load(G.EVAL / f"frozen_{c}__{REF}.npz")
        alive = ref["alive"]
        diff = np.nanmax(np.abs(r["corr"][alive] - ref["corr"][alive]))
        if not diff <= 1e-9:
            sys.exit(f"IDENTITY CHECK FAILED [{c}]: max |r - frozen r| = {diff:.3e}")
        np.savez(G.EVAL / f"geom_margins_{c}.npz", alive=alive,
                 **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
        a = alive & np.isfinite(r["corr"])
        y = 1 - r["corr"][a]
        stats = {"log_rate": np.log(r["rate"][a] + 1e-9),
                 "log_mean_act": np.log(r["mean_act"][a] + 1e-9),
                 "log_peak_act": np.log(r["peak"][a] + 1e-9)}
        sp_stats = {n: float(spearmanr(v, y).statistic) for n, v in stats.items()}
        bad = a & (r["corr"] <= 0.9)
        row = {"identity_max_abs_diff": float(diff), "n_alive": int(a.sum()),
               "mean_corr": float(np.mean(r["corr"][a])),
               "spearman_model": float(spearmanr(r["pred"][a], y).statistic),
               "spearman_model_hetero": float(spearmanr(r["pred_h"][a], y).statistic),
               "spearman_stats": sp_stats,
               "best_stat_abs": max(abs(v) for v in sp_stats.values()),
               "partial_peak_given_model": float(partial_spearman(
                   stats["log_peak_act"], y, r["pred"][a])),
               "median_flip_share_all": float(np.nanmedian(r["flip_share"][a])),
               "n_degraded_or_damaged": int(bad.sum()),
               "median_flip_share_bad": (float(np.nanmedian(r["flip_share"][bad]))
                                         if bad.any() else None),
               "median_out_rate": float(np.median(r["out_rate"][a])),
               "e2_mean_cal": r["e2_mean"]}
        out["codecs"][c] = row
        path.write_text(json.dumps(out, indent=2))
        print(c, json.dumps(row), flush=True)
    cs = out["codecs"]
    nl = [c for c in NOISE_LIKE if c in cs]
    if len(nl) == len(NOISE_LIKE) and all(c in cs for c in SIGNAL):
        med = float(np.median([cs[c]["spearman_model"] for c in nl]))
        wins = sum(abs(cs[c]["spearman_model"]) > cs[c]["best_stat_abs"] for c in nl)
        part = float(np.median([abs(cs[c]["partial_peak_given_model"]) for c in nl]))
        fs = {c: cs[c]["median_flip_share_bad"] for c in SIGNAL}
        out["readings"] = {
            "H1a_median_spearman_noise_like": med,
            "H1a": ("margins + encoder-projected error explain frozen damage" if med >= 0.5
                    else "Gaussian margin model does not explain it" if med < 0.3 else "partial"),
            "H1a_hetero_median": float(np.median([cs[c]["spearman_model_hetero"] for c in nl])),
            "H1b_model_beats_best_stat": f"{wins}/{len(nl)}",
            "H1b": "pass" if wins >= 9 else "fail",
            "H1c_median_flip_share_bad": fs,
            "H1c": ("wall crossings dominate" if all(v is not None and v >= 0.5 for v in fs.values())
                    else "in-cell distortion dominates" if all(v is not None and v <= 0.3 for v in fs.values())
                    else "mixed"),
            "H1d_median_abs_partial_peak": part,
            "H1d": "pass" if part < 0.1 else "fail",
            "signal_codecs_spearman": {c: cs[c]["spearman_model"] for c in SIGNAL}}
        path.write_text(json.dumps(out, indent=2))
        print(json.dumps(out["readings"], indent=1))


if __name__ == "__main__":
    main()
