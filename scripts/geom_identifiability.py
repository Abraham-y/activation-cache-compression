"""Can ONE SAE's geometry predict which of its features are seed-stable?
(Appendix J.)

PRE-REGISTERED READINGS (fixed 11 Oct 2026, before this script existed):
  H3a  The co-firing share D_i alone predicts INstability with AUC >= 0.60
       (higher D, less stable). Below 0.55 -> co-firing degeneracy is not
       the mechanism.
  H3b  Geometry-only CV AUC >= stats-only CV AUC + 0.05.
  H3c  Coefficients fit on fp32_s0 (1x, k=32), applied to each regime's own
       s0 SAE and stability labels (b3x, b10x, freshb3x, k16, k64): AUC >=
       0.65 in each. JumpReLU excluded (no TopK margin).
  H3d  The same seed-fit model predicts which features survive retraining
       on fp16, rot-int4 and std-int4 with AUC >= 0.65.
OUTCOME (11 Oct): H3a fail (0.437, slightly the wrong way), H3b fail
(0.829 vs 0.846), H3c pass (0.696-0.842), H3d pass (0.743-0.762).

Identifiability results for sparse dictionary learning need incoherent
atoms and diverse supports. Two failure modes leave the loss flat:
  co-firing: atoms i, j that mostly fire on the same tokens can be rotated
    within their span with compensating codes at almost no cost;
  merge/split: near-parallel atoms that fire on different tokens can be
    merged into one or split differently.
A TopK feature that usually wins its slot by a thin margin is a third:
which tokens it gets is decided by the competition, not by the data.
So instability should be visible in a single SAE. Predictors are computed
on held-out eval tokens 500k-1M; the stability labels use the first 500k.

Usage: PYTHONPATH=src python scripts/geom_identifiability.py
Writes runs_m1/eval/geom_identifiability.json and geom_ident_<ref>.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import geom_common as G  # noqa: E402

STATS = ["log_rate", "log_mean_act", "log_peak_act"]
GEOM = ["cofire_share", "coherence", "nn_jaccard", "margin_rel", "encdec_cos"]
REGIMES = {  # ref -> the two other seeds that define its stability labels
    "fp32_s0": ["fp32_s1", "fp32_s2"],
    "fp32_b3x_s0": ["fp32_b3x_s1", "fp32_b3x_s2"],
    "fp32_b10x_s0": ["fp32_b10x_s1", "fp32_b10x_s2"],
    "fp32_freshb3x_s0": ["fp32_freshb3x_s1", "fp32_freshb3x_s2"],
    "fp32_k16_s0": ["fp32_k16_s1", "fp32_k16_s2"],
    "fp32_k64_s0": ["fp32_k64_s1", "fp32_k64_s2"],
}
CODECS = ["fp16_s0", "rot-int4_s0", "std-int4_s0"]


@torch.no_grad()
def predictors(ref, alive):
    dev = G.device()
    s, _ = G.sae(ref, dev)
    k = s.k
    ai = np.flatnonzero(alive)
    col = -np.ones(s.d_sae, dtype=np.int64)
    col[ai] = np.arange(ai.size)
    m = ai.size
    n_fire = np.zeros(m); s1 = np.zeros(m); s2 = np.zeros(m); peak = np.zeros(m)
    C2 = sp.csr_matrix((m, m)); CC = sp.csr_matrix((m, m))
    mf, mm, ma = [], [], []
    N = 0
    for _, x in G.iter_range("master-eval", *G.HELDOUT, chunk=16384):
        pre = s.pre_acts(torch.from_numpy(x).to(dev))
        v, idx = torch.topk(pre, k + 1, dim=-1)
        runner = v[:, k:k + 1]
        a = v[:, :k].clamp(min=0).cpu().numpy()
        marg = (v[:, :k] - runner).cpu().numpy()
        idx = idx[:, :k].cpu().numpy()
        n = x.shape[0]; N += n
        c = col[idx]
        keep = (c >= 0) & (a > 0)
        rows = np.repeat(np.arange(n), k).reshape(n, k)[keep]
        cc, aa, mg = c[keep], a[keep], marg[keep]
        np.add.at(n_fire, cc, 1); np.add.at(s1, cc, aa); np.add.at(s2, cc, aa * aa)
        np.maximum.at(peak, cc, aa)
        mf.append(cc); mm.append(mg); ma.append(aa)
        A2 = sp.csr_matrix((aa.astype(np.float64) ** 2, (rows, cc)), shape=(n, m))
        B = sp.csr_matrix((np.ones_like(aa, dtype=np.float64), (rows, cc)), shape=(n, m))
        C2 = C2 + (A2.T @ B).tocsr()
        CC = CC + (B.T @ B).tocsr()
    C2 = C2.toarray(); CC = CC.toarray()
    mf = np.concatenate(mf); mm = np.concatenate(mm); ma = np.concatenate(ma)
    order = np.argsort(mf, kind="stable")
    mf, mm, ma = mf[order], mm[order], ma[order]
    bounds = np.searchsorted(mf, np.arange(m + 1))
    med_m = np.zeros(m); med_a = np.zeros(m)
    for i in range(m):
        a0, b0 = bounds[i], bounds[i + 1]
        if b0 > a0:
            med_m[i] = np.median(mm[a0:b0]); med_a[i] = np.median(ma[a0:b0])
    diag2 = np.diag(C2).copy()
    off2 = C2.copy(); np.fill_diagonal(off2, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cofire = np.where(diag2 > 0, off2.max(1) / diag2, 0.0)
    Wd = s.W_dec.detach().cpu().numpy()[ai].astype(np.float64)
    Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
    Gm = Wd @ Wd.T
    np.fill_diagonal(Gm, 0)
    nn = np.abs(Gm).argmax(1)
    coh = np.abs(Gm)[np.arange(m), nn]
    dg = np.diag(CC)
    inter = CC[np.arange(m), nn]
    union = dg + dg[nn] - inter
    jac = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    We = s.W_enc.detach().cpu().numpy()[:, ai].T.astype(np.float64)
    encdec = (We * Wd).sum(1) / np.linalg.norm(We, axis=1)
    eps = 1e-12
    feats = {
        "log_rate": np.log(n_fire / N + 1e-7),
        "log_mean_act": np.log(np.where(n_fire > 0, s1 / np.maximum(n_fire, 1), 0) + 1e-6),
        "log_peak_act": np.log(peak + 1e-6),
        "cofire_share": cofire,
        "coherence": coh,
        "nn_jaccard": jac,
        "margin_rel": np.where(med_a > 0, med_m / (med_a + eps), 0.0),
        "encdec_cos": encdec,
    }
    silent = int((n_fire == 0).sum())
    return ai, feats, silent, N


def X(feats, names):
    return np.column_stack([feats[n] for n in names])


def cv_auc(Xm, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(model, Xm, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main():
    out = {"regimes": {}}
    fitted = {}
    for ref, others in REGIMES.items():
        alive, stable = G.stability(ref, others)
        ai, feats, silent, N = predictors(ref, alive)
        y = stable[ai].astype(int)
        np.savez(G.EVAL / f"geom_ident_{ref}.npz", alive_idx=ai, stable=y,
                 **feats)
        r = {"n_alive": int(ai.size), "stable_frac": float(y.mean()),
             "silent_on_heldout": silent, "heldout_tokens": N,
             "single_auc_stable": {n: float(roc_auc_score(y, f))
                                   for n, f in feats.items()}}
        if ref == "fp32_s0":
            r["cv_auc_stats"] = cv_auc(X(feats, STATS), y)
            r["cv_auc_geom"] = cv_auc(X(feats, GEOM), y)
            r["cv_auc_both"] = cv_auc(X(feats, STATS + GEOM), y)
            for names, key in ((GEOM, "geom"), (STATS, "stats"), (STATS + GEOM, "both")):
                mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                mdl.fit(X(feats, names), y)
                fitted[key] = (names, mdl)
            lr = fitted["geom"][1][-1]
            r["geom_coef"] = dict(zip(GEOM, map(float, lr.coef_[0])))
            r["codec_survival_auc"] = {}
            for c in CODECS:
                a2, surv = G.survived(ref, c)
                assert np.array_equal(a2, alive)
                ys = surv[ai].astype(int)
                r["codec_survival_auc"][c] = {
                    key: float(roc_auc_score(ys, mdl.predict_proba(X(feats, names))[:, 1]))
                    for key, (names, mdl) in fitted.items()}
                r["codec_survival_auc"][c]["survived_frac"] = float(ys.mean())
        else:
            r["transfer_auc"] = {
                key: float(roc_auc_score(y, mdl.predict_proba(X(feats, names))[:, 1]))
                for key, (names, mdl) in fitted.items()}
        out["regimes"][ref] = r
        print(ref, json.dumps({k: v for k, v in r.items() if k != "single_auc_stable"}),
              flush=True)
    s0 = out["regimes"]["fp32_s0"]
    a_unstable = 1 - s0["single_auc_stable"]["cofire_share"]
    tr = {k: v["transfer_auc"]["geom"] for k, v in out["regimes"].items() if k != "fp32_s0"}
    cod = {c: v["geom"] for c, v in s0["codec_survival_auc"].items()}
    out["readings"] = {
        "H3a_cofire_auc_for_unstable": a_unstable,
        "H3a": ("pass" if a_unstable >= 0.60 else
                "co-firing is not the mechanism" if a_unstable < 0.55 else "weak"),
        "H3b_geom_minus_stats": s0["cv_auc_geom"] - s0["cv_auc_stats"],
        "H3b": "pass" if s0["cv_auc_geom"] >= s0["cv_auc_stats"] + 0.05 else "fail",
        "H3c_transfer_auc_geom": tr,
        "H3c": "pass" if all(v >= 0.65 for v in tr.values()) else "fail",
        "H3d_codec_survival_auc_geom": cod,
        "H3d": "pass" if all(v >= 0.65 for v in cod.values()) else "fail",
    }
    (G.EVAL / "geom_identifiability.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["readings"], indent=1))


if __name__ == "__main__":
    main()
