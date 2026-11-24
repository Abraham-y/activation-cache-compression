"""SAE training as a dynamical system (Appendix J, "When features are
decided").

PRE-REGISTERED (fixed 11 Oct 2026, before this script existed):
  I1   run A's final weights equal runs_m1/saes/fp32_s0/sae.pt bit for bit
       (else report, and continue with I2 only).
  I2   an unperturbed clone from step 4000 equals A exactly for 300 steps.
  H2a  after a one-ulp impulse at t0 = 0 the relative Frobenius distance of
       W_dec from A grows log-linearly (R^2 >= 0.9 over >= 1.5 decades)
       before levelling off; a jump to the plateau within 100 steps -> "no
       amplification phase"; log-log better by R^2 > 0.05 -> "power law".
  H2b  dense impulses at t0 = 0 of relative L2 1e-7, 1e-5, 1e-3: the step
       at which each first reaches 1e-3 shifts by ln(F)/lambda (+-50%).
  H2c  T_sat from the one-ulp curve; impulses at t0 = 8000 - {4,2,1,.5,.25}
       x T_sat: agreement >= 90% when 8000 - t0 <= T_sat/4, and within 5 pp
       of t0 = 0 when 8000 - t0 >= 2 T_sat.
DEVIATION (11 Oct, after phase 1, before any amended run): the one-ulp
impulse at step 0 (batch row 0, channel 57) was absorbed by rounding (the
clone stayed bit-identical), so H2a-H2c, which all read off that curve,
could not be evaluated and phase 2 was cancelled. Exploratory, seen before
the amendment: dense impulses moved W_dec within 10 steps, then grew
roughly linearly with no exponential phase. AMENDED, fixed before running
(late / early below):
  A1   share of 200 random single-scalar one-ulp impulses that change the
       state after one step, at steps 0 and 4000 (a measurement).
  A2   the oneulp scalar enters training at step 112 (batch row 1884) and
       step 4970 (row 2407), channel 57; replaying both must reproduce
       runs_m1/saes/oneulp_s0 bit for bit (I1b); which one propagates?
  A3   the dense 1e-7 impulse at t0 in {0, 4000, 6000, 7000, 7500, 7800,
       7950}: (i) agreement at 7950 >= 90%; (ii) |agree(4000) - agree(0)|
       <= 5 pp -> "saturated within 4,000 steps", else "damage keeps
       growing with remaining steps"; (iii) |agree(0) - 66.0| <= 5 pp
       (noise7, the same dose on every batch) -> "one perturbed batch costs
       as much as perturbing all of them".
  I4   the t0 = 0 fork equals phase-1 dose7 at every logged step to 2,500.
OUTCOME (12 Oct; A2 corrected 3 Nov): I1, I1b, I2, I3 pass. A1: 58.0% /
58.5% change the state after one step. A2: the step-112 entry changes the
state after one step but is undone (the replay is bit-identical to A from
step 130 until 4970); only the step-4970 entry carries through, so the
oneulp 99.98% is the late-impulse branch. A3: 64.7, 99.9, then 100% -> (i)
pass, (ii) the "keeps growing" branch (35 pp), (iii) pass (64.7 vs 66.0).
I4: equal at 89 of 90 logged steps; at step 160 the logged W_dec distance
differs while W_enc and every later value are equal, a glitch in the
logged norm, not in training (geom_summary.json "I4_mismatch").
NOTE (added 3 Nov): the A2 line above shortens the amended plan as written
in the lab notebook on 11 Oct, before the late / early passes ran, which
continued: "Reading: which occurrence propagates. If only the step-4970
one does, the paper's 99.98% is a late-impulse effect." Equality in
states_equal is of the SAE weights and dead-feature counters only; Adam's
moment estimates are not compared. At step 0 the batch mean also sets
b_dec (as in acc.sae.train_sae), so a step-0 impulse also perturbs that
initial value.

If training amplifies a perturbation like e * exp(lambda t) until it
saturates, every perturbation big enough to saturate within the budget
ends in the same place: the step-shaped dose response of the paper (one
ulp on a millionth of values costs as much as fp16). This measures the
growth directly. Run A is the paper's fp32 seed-0 run, step for step (the
loop below mirrors acc.sae.train_sae for arch=topk; I1 checks it
reproduces runs_m1/saes/fp32_s0 bit for bit). Perturbed runs are clones
of A that receive ONE impulse at step t0 and then train in lockstep with
A on the same batches, so their distance from A is measured live.

  phase1: A from step 0 to 8000, plus
          ulp_t0      one ulp up on batch row 0, channel 57 at step 0
          dose{E}     dense relative-L2 10^-E noise on the whole batch at
                      step 0, E in 7, 5, 3; 2,500 steps
          null4000    an UNPERTURBED clone from step 4000 for 300 steps;
                      must equal A exactly (I2)
          A's full state is saved every 500 steps for phase 2.
  phase2: T_sat from ulp_t0's curve; impulses at t0 = 8000 - {4,2,1,.5,.25}
          x T_sat (rounded to 50), forked in lockstep from A restarted at
          the last saved state before the earliest t0 (A's restart must
          reproduce phase 1's saved state at the next 500 boundary: I3).
Final checkpoints of A and of every full-length fork go to
runs_m1/saes/geomdyn_*_s0 so the paper's eval-pair scores them.

Usage: PYTHONPATH=src python scripts/geom_dynamics.py phase1|late|early
(phase2 is the cancelled original; scripts/run_geom_dynamics.sh runs the rest)
Writes runs_m1/eval/geom_dynamics_*.json(l)
"""
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import geom_common as G  # noqa: E402
from acc.sae import SAEConfig, TopKSAE  # noqa: E402
from acc.store import BatchLoader, Store  # noqa: E402

import os
TEST = os.environ.get("GEOMDYN_TEST")      # smoke test: tiny run, eval cache, scratch outputs
STEPS = 120 if TEST else 8000
STATE_EVERY = 50 if TEST else 500
LOG_EVERY = 10 if TEST else 50
PROBE_EVERY = 50 if TEST else 500
STATE_DIR = Path(TEST) / "states" if TEST else G.ROOT / "runs_m1" / "geomdyn_states"
OUT = Path(TEST) if TEST else G.EVAL
CKPT = Path(TEST) / "saes" if TEST else G.SAES
TRAIN_STORE = "master-eval" if TEST else "master-train"
ULP_ROW, ULP_COL = 0, 57


def cfg_m1():
    # the m1 preset exactly as cmd_train builds it
    return SAEConfig(d_in=768, d_sae=12288, k=32, aux_k=512, dead_tokens=2_000_000,
                     lr=3e-4, batch_size=4096, steps=STEPS, log_every=100, arch="topk")


class Run:
    """One SAE + Adam + dead-tracking, stepped exactly as train_sae does."""

    def __init__(self, name, cfg, dev, sae=None, opt_state=None, tsf=None):
        self.name, self.cfg, self.dev = name, cfg, dev
        self.sae = sae if sae is not None else TopKSAE(cfg.d_in, cfg.d_sae, cfg.k, seed=0).to(dev)
        self.opt = torch.optim.Adam(self.sae.parameters(), lr=cfg.lr)
        if opt_state is not None:
            self.opt.load_state_dict(opt_state)
        self.tsf = (tsf.clone() if tsf is not None
                    else torch.zeros(cfg.d_sae, dtype=torch.long, device=dev))

    def clone(self, name):
        sae = copy.deepcopy(self.sae)
        r = Run(name, self.cfg, self.dev, sae=sae, tsf=self.tsf)
        r.opt.load_state_dict(copy.deepcopy(self.opt.state_dict()))
        return r

    def step(self, step, x):
        cfg, sae = self.cfg, self.sae
        if step == 0:
            with torch.no_grad():
                sae.b_dec.copy_(x.mean(dim=0))
        recon, z, pre = sae(x)
        mse = (recon - x).pow(2).sum(dim=1).mean()
        fired = z.detach().gt(0).any(dim=0)
        self.tsf += cfg.batch_size
        self.tsf[fired] = 0
        dead = self.tsf > cfg.dead_tokens
        n_dead = int(dead.sum())
        loss = mse
        if n_dead > 0 and cfg.aux_coef > 0:
            err = x - recon.detach()
            k_aux = min(cfg.aux_k, n_dead)
            pre_dead = pre.masked_fill(~dead.unsqueeze(0), float("-inf"))
            aux_vals, aux_idx = torch.topk(pre_dead, k_aux, dim=-1)
            aux_vals = aux_vals.clamp(min=0.0)
            z_aux = torch.zeros_like(z)
            z_aux.scatter_(1, aux_idx, aux_vals)
            recon_aux = z_aux @ sae.W_dec
            aux_mse = (recon_aux - err).pow(2).sum(dim=1).mean()
            if torch.isfinite(aux_mse):
                loss = loss + cfg.aux_coef * aux_mse
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.project_decoder_grads()
        self.opt.step()
        sae.renorm_decoder()
        return float(mse.detach())

    def save_state(self, path):
        torch.save({"sae": {k: v.cpu() for k, v in self.sae.state_dict().items()},
                    "opt": self.opt.state_dict(), "tsf": self.tsf.cpu()}, path)

    @classmethod
    def load_state(cls, name, cfg, dev, path):
        st = torch.load(path, map_location="cpu", weights_only=False)
        sae = TopKSAE(cfg.d_in, cfg.d_sae, cfg.k, seed=0)
        sae.load_state_dict(st["sae"])
        sae = sae.to(dev)
        r = cls(name, cfg, dev, sae=sae, tsf=st["tsf"].to(dev))
        r.opt.load_state_dict(st["opt"])
        return r

    def save_ckpt(self, tag, extra):
        d = CKPT / f"geomdyn_{tag}_s0"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in self.sae.state_dict().items()},
                    "cfg": asdict(self.cfg), "seed": 0, "init_seed": None,
                    "codec": "fp32", "geomdyn": extra}, d / "sae.pt")
        (d / "summary.json").write_text(json.dumps({"codec": "fp32", "seed": 0,
                                                    "cfg": asdict(self.cfg), "geomdyn": extra}))


def batch_stream(dev, skip=0):
    """The paper's fp32 seed-0 batch stream (cmd_train's loader), optionally
    fast-forwarded on the CPU before batches move to the device."""
    store = Store(G.CACHE / TRAIN_STORE)
    loader = BatchLoader(store, 4096, seed=0, device="cpu", decode="cpu")
    it = loader.batches(epochs=None, prefetch=1)   # same stream as prefetch=2, less RAM
    for _ in range(skip):
        next(it)
    for x in it:
        yield x.to(dev)


def perturb_ulp(x):
    x = x.clone()
    v = x[ULP_ROW, ULP_COL:ULP_COL + 1].cpu().numpy()
    x[ULP_ROW, ULP_COL] = float(np.nextafter(v, np.float32(np.inf), dtype=np.float32)[0])
    return x


def perturb_dense(x, rel, seed):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(x.shape, generator=g).to(x.device)
    u = u / u.norm(dim=1, keepdim=True) * x.norm(dim=1, keepdim=True) * rel
    return x + u


def rel_dist(a, b):
    out = {}
    for k in ("W_dec", "W_enc"):
        wa, wb = getattr(a.sae, k).detach(), getattr(b.sae, k).detach()
        out[k] = float((wb - wa).norm() / wa.norm())
    return out


def states_equal(a, b):
    sa, sb = a.sae.state_dict(), b.sae.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa) and torch.equal(a.tsf, b.tsf)


@torch.no_grad()
def probe_agreement(a, b, xp):
    """Same-index activation correlation on held-out probe tokens: the share
    of A's features firing on the probe whose same-index twin in B
    correlates > 0.9 (clones share indices until they diverge)."""
    za = torch.zeros(xp.shape[0], a.sae.d_sae, device=xp.device)
    zb = torch.zeros_like(za)
    va, ia = a.sae.feature_acts(xp); za.scatter_(1, ia, va)
    vb, ib = b.sae.feature_acts(xp); zb.scatter_(1, ib, vb)
    alive = (za > 0).float().mean(0) > 1e-3
    za, zb = za[:, alive], zb[:, alive]
    za = za - za.mean(0); zb = zb - zb.mean(0)
    r = (za * zb).sum(0) / (za.norm(dim=0) * zb.norm(dim=0)).clamp(min=1e-12)
    return float((r > 0.9).float().mean())


def sd_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def probe_tokens(dev):
    xs = [x for _, x in G.iter_range("master-eval", *G.HELDOUT, chunk=16384)]
    return torch.from_numpy(np.concatenate(xs)[:8192]).to(dev)


def lockstep(A, forks, start, stop, batches, log_path, xp, plan, ckpt_tags):
    """Step A and its forks from `start` to `stop`. plan: list of
    (t0, name, perturb_fn, stop_at) clones of A created at t0."""
    t_start = time.perf_counter()
    for step in range(start, stop):
        x = next(batches)
        for t0, name, fn, until in plan:
            if t0 == step:
                forks[name] = (A.clone(name), fn, t0, until)
        A.step(step, x)
        for name, (r, fn, t0, until) in list(forks.items()):
            if step >= until:
                continue
            r.step(step, fn(x) if (step == t0 and fn is not None) else x)
        if (step + 1) % STATE_EVERY == 0 and log_path.name.startswith("geom_dynamics_phase1"):
            A.save_state(STATE_DIR / f"A_{step + 1}.pt")
        # every 10 steps for the first 500 after any fork's impulse, else LOG_EVERY
        fine = any(0 < step + 1 - t0 <= 500 for (_, _, t0, _) in forks.values())
        if (step + 1) % LOG_EVERY == 0 or step + 1 == stop or (fine and (step + 1) % 10 == 0):
            rec = {"step": step + 1, "elapsed_s": time.perf_counter() - t_start, "runs": {}}
            for name, (r, fn, t0, until) in forks.items():
                if step >= until or step < t0:
                    continue
                d = rel_dist(A, r)
                if (step + 1) % PROBE_EVERY == 0:
                    d["probe_same_index_agree"] = probe_agreement(A, r, xp)
                if fn is None:
                    d["exactly_equal"] = states_equal(A, r)
                rec["runs"][name] = d
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        for name, (r, fn, t0, until) in list(forks.items()):
            if step + 1 == until:
                if fn is None:   # I2: unperturbed clone must equal A exactly
                    ok = states_equal(A, r)
                    print(f"I2 {name}: exactly equal to A after {until - t0} steps: {ok}", flush=True)
                    if not ok:
                        sys.exit("I2 FAILED: an unperturbed clone diverged from A")
                elif name in ckpt_tags:
                    r.save_ckpt(name, {"t0": t0, "stop": until})
                del forks[name]
                if r.dev == "mps":
                    torch.mps.empty_cache()


def phase1():
    dev = G.device()
    torch.manual_seed(0)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT / "geom_dynamics_phase1.jsonl"
    if log.exists():
        log.unlink()
    cfg = cfg_m1()
    A = Run("A", cfg, dev)
    xp = probe_tokens(dev)
    plan = [(0, "ulp_t0", perturb_ulp, STEPS),
            (0, "dose7", lambda x: perturb_dense(x, 1e-7, 7), 100 if TEST else 2500),
            (0, "dose5", lambda x: perturb_dense(x, 1e-5, 5), 100 if TEST else 2500),
            (0, "dose3", lambda x: perturb_dense(x, 1e-3, 3), 100 if TEST else 2500),
            (40, "null40", None, 70) if TEST else (4000, "null4000", None, 4300)]
    forks = {}
    lockstep(A, forks, 0, STEPS, batch_stream(dev), log, xp, plan, {"ulp_t0"})
    A.save_ckpt("A", {"t0": None})
    if TEST:
        print("TEST phase1 done", flush=True)
        return
    ref = torch.load(G.SAES / "fp32_s0" / "sae.pt", map_location="cpu", weights_only=True)["state_dict"]
    mine = {k: v.cpu() for k, v in A.sae.state_dict().items()}
    i1 = all(torch.equal(ref[k], mine[k]) for k in ref)
    res = {"I1_reproduces_fp32_s0": i1, "sha_A": sd_sha(mine), "sha_fp32_s0": sd_sha(ref)}
    if not i1:
        res["I1_rel_W_dec_vs_fp32_s0"] = float((mine["W_dec"] - ref["W_dec"]).norm() / ref["W_dec"].norm())
    (OUT / "geom_dynamics_phase1.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res), flush=True)


def curve(log, name, key="W_dec"):
    st, d = [], []
    for line in open(log):
        r = json.loads(line)
        if name in r["runs"]:
            st.append(r["step"]); d.append(r["runs"][name][key])
    return np.array(st), np.array(d)


def t_sat():
    st, d = curve(OUT / "geom_dynamics_phase1.jsonl", "ulp_t0")
    final = d[st == STEPS][0]
    ok = (d > 0) & (np.log10(np.maximum(d, 1e-30)) >= np.log10(final) - math.log10(3))
    return int(st[np.argmax(ok)]), float(final)


def phase2():
    dev = G.device()
    T, final = t_sat()
    t0s = sorted({int(round((STEPS - f * T) / 50) * 50) for f in (4, 2, 1, 0.5, 0.25)})
    t0s = [t for t in t0s if 0 < t < STEPS]
    start = max(s for s in range(0, STEPS, STATE_EVERY) if s <= min(t0s))
    cfg = cfg_m1()
    A = (Run.load_state("A", cfg, dev, STATE_DIR / f"A_{start}.pt") if start > 0
         else Run("A", cfg, dev))
    xp = probe_tokens(dev)
    log = OUT / "geom_dynamics_phase2.jsonl"
    if log.exists():
        log.unlink()
    plan = [(t, f"ulp_t{t}", perturb_ulp, STEPS) for t in t0s]
    forks = {}
    nxt = start + STATE_EVERY
    meta = {"T_sat": T, "final_ulp_t0_dist": final, "t0s": t0s, "restart_from": start}
    print(json.dumps(meta), flush=True)
    # I3: A restarted from its saved state reproduces phase 1 at the next boundary
    batches = batch_stream(dev, skip=start)
    lockstep(A, forks, start, nxt, batches, log, xp, plan,
             {n for _, n, _, _ in plan})
    saved = Run.load_state("A_saved", cfg, dev, STATE_DIR / f"A_{nxt}.pt")
    meta["I3_restart_reproduces_phase1"] = states_equal(A, saved)
    del saved
    if not meta["I3_restart_reproduces_phase1"]:
        sys.exit("I3 FAILED: restarted A differs from phase 1")
    lockstep(A, forks, nxt, STEPS, batches, log, xp, plan,
             {n for _, n, _, _ in plan})
    (OUT / "geom_dynamics_phase2.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


# ---------------------------------------------------------------------------
# Amended plan (DEVIATION in the header): the pre-registered one-ulp
# impulse at step 0 was absorbed by rounding.

def perturb_ulp_at(row, col):
    def fn(x):
        x = x.clone()
        v = x[row, col:col + 1].cpu().numpy()
        x[row, col] = float(np.nextafter(v, np.float32(np.inf), dtype=np.float32)[0])
        return x
    return fn


D7 = lambda x: perturb_dense(x, 1e-7, 7)   # == phase-1 dose7 (same seed)


def absorption_scan(A, x, step, n=200, seed=0):
    """Share of random single-scalar one-ulp impulses that change the state
    after one step (A itself is not advanced)."""
    rng = np.random.default_rng((seed, step))
    rows = rng.integers(0, x.shape[0], n); cols = rng.integers(0, x.shape[1], n)
    ref = A.clone("ref"); ref.step(step, x)
    changed = []
    for r, c in zip(rows, cols):
        b = A.clone("probe"); b.step(step, perturb_ulp_at(int(r), int(c))(x))
        changed.append(not states_equal(ref, b))
        del b
    del ref
    return {"step": step, "n": n, "frac_changed": float(np.mean(changed)),
            "positions": [[int(r), int(c), bool(ch)] for r, c, ch in zip(rows, cols, changed)]}


def one_step_check(A, x, step, row, col):
    ref = A.clone("ref"); ref.step(step, x)
    b = A.clone("probe"); b.step(step, perturb_ulp_at(row, col)(x))
    return {"step": step, "row": row, "col": col, "changes_state": not states_equal(ref, b)}


def multi_lockstep(A, start, stop, batches, log_path, xp, plan, hooks):
    """plan: name -> {"impulses": {step: fn}, "until": step}; a fork is cloned
    from A at its first impulse step. hooks: step -> callable(A, x) run
    before that step (A not advanced)."""
    forks = {}
    t_start = time.perf_counter()
    first = {n: min(v["impulses"]) for n, v in plan.items()}
    out = {}
    for step in range(start, stop):
        x = next(batches)
        if step in hooks:
            out.setdefault("hooks", []).append(hooks[step](A, x))
            print(json.dumps(out["hooks"][-1])[:300], flush=True)
        for n, t0 in first.items():
            if t0 == step:
                forks[n] = A.clone(n)
        A.step(step, x)
        for n, r in forks.items():
            fn = plan[n]["impulses"].get(step)
            r.step(step, fn(x) if fn is not None else x)
        fine = any(0 < step + 1 - first[n] <= 500 for n in forks)
        if (step + 1) % LOG_EVERY == 0 or step + 1 == stop or (fine and (step + 1) % 10 == 0):
            rec = {"step": step + 1, "elapsed_s": time.perf_counter() - t_start, "runs": {}}
            for n, r in forks.items():
                d = rel_dist(A, r)
                d["exactly_equal"] = states_equal(A, r)
                if (step + 1) % PROBE_EVERY == 0:
                    d["probe_same_index_agree"] = probe_agreement(A, r, xp)
                rec["runs"][n] = d
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        for n in list(forks):
            if step + 1 == plan[n]["until"]:
                forks[n].save_ckpt(n, {"impulses": sorted(plan[n]["impulses"]), "stop": step + 1})
                del forks[n]
                if dev_is_mps(A):
                    torch.mps.empty_cache()
    return out


def dev_is_mps(A):
    return str(A.dev).startswith("mps")


def late():
    dev = G.device()
    cfg = cfg_m1()
    A = Run.load_state("A", cfg, dev, STATE_DIR / "A_4000.pt")
    xp = probe_tokens(dev)
    log = OUT / "geom_dynamics_late.jsonl"
    if log.exists():
        log.unlink()
    plan = {f"d7_t{t}": {"impulses": {t: D7}, "until": STEPS}
            for t in (4000, 6000, 7000, 7500, 7800, 7950)}
    hooks = {4000: lambda A, x: absorption_scan(A, x, 4000),
             4970: lambda A, x: one_step_check(A, x, 4970, 2407, 57)}
    res = multi_lockstep(A, 4000, STEPS, batch_stream(dev, skip=4000), log, xp, plan, hooks)
    ref = torch.load(G.SAES / "geomdyn_A_s0" / "sae.pt", map_location="cpu", weights_only=True)["state_dict"]
    mine = {k: v.cpu() for k, v in A.sae.state_dict().items()}
    res["I3_late_A_reproduces_phase1_A"] = all(torch.equal(ref[k], mine[k]) for k in ref)
    (OUT / "geom_dynamics_late.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "hooks"}), flush=True)


def early():
    dev = G.device()
    cfg = cfg_m1()
    A = Run("A", cfg, dev)
    xp = probe_tokens(dev)
    log = OUT / "geom_dynamics_early.jsonl"
    if log.exists():
        log.unlink()
    plan = {"d7_t0": {"impulses": {0: D7}, "until": STEPS},
            "oneulp_replay": {"impulses": {112: perturb_ulp_at(1884, 57),
                                           4970: perturb_ulp_at(2407, 57)}, "until": STEPS}}
    hooks = {0: lambda A, x: absorption_scan(A, x, 0),
             112: lambda A, x: one_step_check(A, x, 112, 1884, 57)}
    res = multi_lockstep(A, 0, STEPS, batch_stream(dev), log, xp, plan, hooks)
    def sd(name):
        return torch.load(G.SAES / name / "sae.pt", map_location="cpu", weights_only=True)["state_dict"]
    a, o, rp = sd("geomdyn_A_s0"), sd("oneulp_s0"), sd("geomdyn_oneulp_replay_s0")
    mine = {k: v.cpu() for k, v in A.sae.state_dict().items()}
    res["early_A_reproduces_phase1_A"] = all(torch.equal(a[k], mine[k]) for k in a)
    res["I1b_replay_reproduces_oneulp_s0"] = all(torch.equal(o[k], rp[k]) for k in o)
    res["replay_rel_W_dec_vs_oneulp_s0"] = float((rp["W_dec"] - o["W_dec"]).norm() / o["W_dec"].norm())
    # I4: d7_t0 equals phase-1 dose7 at every shared logged step <= 2500
    p1 = {r["step"]: r["runs"]["dose7"]["W_dec"] for r in map(json.loads, open(OUT / "geom_dynamics_phase1.jsonl"))
          if "dose7" in r["runs"]}
    e = {r["step"]: r["runs"]["d7_t0"]["W_dec"] for r in map(json.loads, open(log)) if "d7_t0" in r["runs"]}
    shared = sorted(set(p1) & set(e))
    res["I4_d7_t0_equals_phase1_dose7"] = {"n_steps": len(shared),
                                           "all_equal": all(p1[s] == e[s] for s in shared)}
    (OUT / "geom_dynamics_early.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "hooks"}), flush=True)


if __name__ == "__main__":
    {"phase1": phase1, "phase2": phase2, "late": late, "early": early}[sys.argv[1]]()
