"""Calibrate the ReLU SAE's L1 coefficient to hit a target mean L0 on the
fp32 store, so the ReLU architecture comparison runs at matched sparsity
with the TopK SAEs (k=32). The chosen coefficient is then FIXED across all
codecs — same hyperparameters, only the bytes differ.

Short probe trainings at log-spaced coefficients, log-log interpolation to
the target, one confirmation run. Writes eval/l1_calibration.json.

Usage: PYTHONPATH=src python3 scripts/calibrate_l1.py --root runs_m1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import numpy as np  # noqa: E402

from acc.cli import PRESETS, cache_dir  # noqa: E402
from acc.sae import SAEConfig, train_sae  # noqa: E402
from acc.store import BatchLoader, Store  # noqa: E402
from acc.utils import get_device, write_json  # noqa: E402


def probe_l0(root, preset, coef, steps, device, scratch, arch="relu"):
    p = PRESETS[preset]
    store = Store(cache_dir(root, "fp32", "train"))
    cfg = SAEConfig(
        d_in=store.d, d_sae=p["d_sae"], k=p["k"], arch=arch, l1_coef=coef,
        aux_coef=0.0, dead_tokens=p["dead_tokens"], lr=p["lr"],
        batch_size=p["batch_size"], steps=steps, log_every=max(steps // 4, 1),
    )
    loader = BatchLoader(store, cfg.batch_size, seed=0, device=device)
    out = Path(scratch) / f"l1probe_{coef:g}"
    train_sae(loader, cfg, seed=0, device=device, out_dir=out,
              codec_name="fp32")
    logs = [json.loads(l) for l in open(out / "train_log.jsonl")]
    return logs[-1]["l0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_m1")
    ap.add_argument("--preset", default="m1")
    ap.add_argument("--target-l0", type=float, default=32.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--arch", default="relu", choices=["relu", "jumprelu"])
    ap.add_argument("--coefs", nargs="+", type=float, default=None)
    ap.add_argument("--max-probes", type=int, default=6)
    args = ap.parse_args()
    if args.coefs is None:
        args.coefs = [10.0, 100.0] if args.arch == "relu" else [0.3, 3.0]
    device = get_device()
    scratch = Path(args.root) / f"l1_calib_{args.arch}"

    probes = []
    for c in args.coefs:
        l0 = probe_l0(args.root, args.preset, c, args.steps, device, scratch,
                      args.arch)
        probes.append((c, l0))
        print(f"l1={c:g}: L0={l0:.1f}")
    # L0 falls with coef; extend the sweep until the target is bracketed
    while len(probes) < args.max_probes:
        l0s_now = [p[1] for p in probes]
        if min(l0s_now) <= args.target_l0 <= max(l0s_now):
            break
        cs_now = [p[0] for p in probes]
        c = (max(cs_now) * 10.0 if min(l0s_now) > args.target_l0
             else min(cs_now) / 10.0)
        l0 = probe_l0(args.root, args.preset, c, args.steps, device, scratch,
                      args.arch)
        probes.append((c, l0))
        print(f"l1={c:g}: L0={l0:.1f} (extending bracket)")

    cs = np.array([p[0] for p in probes])
    l0s = np.array([p[1] for p in probes]).clip(min=1.0)  # log-safe
    order = np.argsort(l0s)
    cs, l0s = cs[order], l0s[order]
    t = args.target_l0

    if l0s[0] <= t <= l0s[-1]:
        # interpolate between the bracketing pair — never beyond the probes
        i = int(np.clip(np.searchsorted(l0s, t) - 1, 0, len(cs) - 2))
        lc = np.log(cs[i]) + (np.log(t) - np.log(l0s[i])) * (
            np.log(cs[i + 1]) - np.log(cs[i])
        ) / (np.log(l0s[i + 1]) - np.log(l0s[i]))
        chosen = float(np.exp(lc))
        intended_l0 = t
        mode = "interpolated"
    else:
        # target unreachable (L1's L0 often plateaus): take the probe whose
        # L0 is log-closest to target; among near-ties (10%) prefer the
        # SMALLEST coef — same L0 with less shrinkage damage
        dist = np.abs(np.log(l0s) - np.log(t))
        near = dist <= dist.min() + np.log(1.1)
        chosen = float(cs[near].min())
        intended_l0 = float(l0s[near][np.argmin(cs[near])])
        mode = "nearest_achievable"
        print(f"WARNING: target L0={t} not achievable "
              f"(probed range {l0s.min():.0f}-{l0s.max():.0f}); "
              f"using nearest L0~{intended_l0:.0f}")

    l0_confirm = probe_l0(args.root, args.preset, chosen, args.steps,
                          device, scratch, args.arch)
    print(f"chosen l1={chosen:g} [{mode}]: L0={l0_confirm:.1f} "
          f"(intended {intended_l0:.1f})")
    ok = (l0_confirm >= 1.0
          and intended_l0 / 2.5 <= l0_confirm <= intended_l0 * 2.5)
    write_json(Path(args.root) / "eval" / f"l1_calibration_{args.arch}.json", {
        "arch": args.arch,
        "probes": probes, "chosen_coef": chosen, "mode": mode,
        "l0_at_chosen": l0_confirm, "intended_l0": intended_l0,
        "target_l0": t, "probe_steps": args.steps, "valid": bool(ok),
        # explicit, machine-readable: did we actually hit the TARGET, or
        # settle for the nearest achievable L0? Consumers must surface this.
        "target_achieved": bool(ok and t / 2.5 <= l0_confirm <= t * 2.5),
    })
    if not ok:
        sys.exit(f"CALIBRATION FAILED: confirm L0={l0_confirm:.1f} not "
                 f"within 2.5x of intended {intended_l0:.1f}")


if __name__ == "__main__":
    main()
