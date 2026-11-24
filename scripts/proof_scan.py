"""Proofing scan for prose-edit damage that a word-level regex cannot see.

Written after a de-dashing pass corrupted seven passages that a
\\b(\\w+)\\s+\\1\\b scan reported clean. That regex structurally cannot
detect: repeated multi-word PHRASES ("at most at most"), repeated
hyphenated TOKENS ("GPT-2 GPT-2", "PCA-256 PCA-256"), adjacent duplicate
MATH ("$4.74$$4.74$"), intra-word SPLICES ("nanaive"), or interleaved
restatements ("permissive since permissive, since").
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLICE_OK = {"bibitem", "newblock"}   # real LaTeX macros, not splices
LATEX_TOKENS = {"checkmark", "textbf", "times", "emph", "dagger",
                "midrule", "toprule", "bottomrule", "multicolumn"}


def scan(text: str) -> list[tuple[str, str]]:
    hits = []
    # 1. repeated n-grams, n = 1..4, over word-ish tokens incl. hyphens/digits
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", text)
    for n in range(1, 5):
        for i in range(len(toks) - 2 * n):
            a, b = toks[i:i + n], toks[i + n:i + 2 * n]
            if a != b or not any(len(w) > 1 for w in a):
                continue
            # numbers legitimately repeat in tables and in per-site triples
            # (e.g. "$1.19$/$1.19$/$1.15$"), and LaTeX commands repeat in
            # every table row; neither is prose damage
            if all(re.fullmatch(r"[\d.]+", w) for w in a):
                continue
            if any(w in LATEX_TOKENS for w in a):
                continue
            hits.append(("repeated phrase", " ".join(a + b)))
    # 2. adjacent identical math spans
    for m in re.finditer(r"(\$[^$]{1,30}\$)\1", text):
        hits.append(("duplicate math", m.group(0)))
    # 3. intra-word splice: a word containing a doubled 2-4 char prefix
    for m in re.finditer(r"\b([A-Za-z]{2,4})\1([A-Za-z]{2,})", text):
        if m.group(0) not in SPLICE_OK:
            hits.append(("possible splice", m.group(0)))
    # 4. punctuation damage
    for pat, lab in ((r",\s*,", "double comma"), (r":\s*:", "double colon"),
                     (r"\(\s*\)", "empty parens"), (r"\s+[,.;:]", "space before punct")):
        for m in re.finditer(pat, text):
            hits.append((lab, repr(m.group(0))))
    return hits


def main() -> int:
    bad = 0
    for f in (ROOT / "paper" / "main.tex", ROOT / "paper" / "appendix_tables.tex"):
        t = f.read_text()
        # full-line LaTeX comments are not prose (e.g. authorship tags);
        # scanning them produces false "repeated phrase" hits
        t = re.sub(r"(?m)^\s*%.*$", "", t)
        t = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", t, flags=re.S)
        hits = scan(t)
        # de-duplicate; splices produce known false positives on real words
        seen, out = set(), []
        for kind, s in hits:
            if (kind, s) in seen:
                continue
            seen.add((kind, s))
            out.append((kind, s))
        if out:
            print(f"{f.name}: {len(out)} candidate(s)")
            for kind, s in out:
                print(f"  [{kind}] {s}")
            bad += len(out)
        else:
            print(f"{f.name}: clean")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
