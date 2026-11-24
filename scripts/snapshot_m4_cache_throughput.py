"""Recover the Pythia cache-build throughput into a first-class artifact.

The forward (cache-build) rate for the M4 sites lived in each master
store's manifest under `throughput`. cleanup.sh reclaimed those caches,
and the manifests went with them, so the only surviving record of
2,400 / 2,146 tok/s is the driver log — which the paper then cites
without an artifact behind it. verify_paper_numbers.py flags exactly
this.

This writes the numbers back out as JSON with their provenance stated:
they are log-derived, not manifest-derived, and the log's tok/s figures
are already rounded to whole tokens per second. That is weaker
provenance than every other number in the paper and the artifact says so
rather than hiding it.

The lesson for next time is in cleanup.sh's direction: snapshot the
manifests BEFORE reclaiming a cache (appendix_tables.py already looks
for a `cache_manifests_snapshot/` directory for this reason).

Usage: python scripts/snapshot_m4_cache_throughput.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "runs_m4.log"
# the driver runs layer 8 first, then layer 16, into separate roots
ORDER = ["runs_m4_L8", "runs_m4_L16"]

PAT = re.compile(
    r"cached master-(?P<split>train|eval): (?P<n>\d+) tokens, "
    r"(?P<bpt>\d+) B/tok, (?P<rate>\d+) tok/s, (?P<sec>[\d.]+)s")


def main():
    if not LOG.exists():
        sys.exit(f"{LOG} not found; nothing to recover")
    hits = [m.groupdict() for m in PAT.finditer(LOG.read_text())]
    # the log carries exactly [L8 train, L8 eval, L16 train, L16 eval]
    if len(hits) != 2 * len(ORDER):
        sys.exit(f"expected {2*len(ORDER)} 'cached master-*' lines, found "
                 f"{len(hits)} — check the log before trusting this")
    per_root: dict[str, dict] = {}
    for root, pair in zip(ORDER, [hits[i:i + 2] for i in range(0, len(hits), 2)]):
        if [h["split"] for h in pair] != ["train", "eval"]:
            sys.exit(f"{root}: expected train then eval, got "
                     f"{[h['split'] for h in pair]}")
        per_root[root] = {h["split"]: {
            "n_tokens": int(h["n"]),
            "bytes_per_token": int(h["bpt"]),
            "tokens_per_s": int(h["rate"]),
            "seconds": float(h["sec"]),
        } for h in pair}
    for root, splits in per_root.items():
        out = ROOT / root / "eval" / "cache_throughput.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "site": root,
            "model_dtype": "bf16",
            "note": ("Forward/cache-build throughput, store write included. "
                     "PROVENANCE IS WEAKER THAN THE REST OF THE PAPER: the "
                     "master manifests that originally carried this were "
                     "reclaimed by cleanup.sh, so these are parsed from "
                     "runs_m4.log, where tok/s is already rounded to whole "
                     "tokens per second. Regenerate with "
                     "scripts/snapshot_m4_cache_throughput.py."),
            "source": "runs_m4.log",
            **splits,
        }, indent=2) + "\n")
        print(f"wrote {out}: train {splits['train']['tokens_per_s']} tok/s, "
              f"eval {splits['eval']['tokens_per_s']} tok/s")


if __name__ == "__main__":
    main()
