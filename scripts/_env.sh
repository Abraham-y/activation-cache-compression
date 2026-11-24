# Shared interpreter resolution. Sourced by every scripts/run_*.sh.
#
# Why this exists: the drivers used to hardcode `python3`. Where the
# project's dependencies live in a conda env, `python3` resolves to that
# env only inside an activated interactive shell — under cron, nohup, a
# fresh non-login shell, or a CI runner it can resolve to a bare system
# Python with no torch. The failure then surfaces at the first import,
# which for run_m1.sh is AFTER an hour of caching. Resolve once, up
# front, and fail immediately with an actionable message instead.
#
# Override with PYTHON=/path/to/python. An explicit PYTHON is
# AUTHORITATIVE: if it cannot import the deps this fails loudly rather
# than silently falling back to some other interpreter, because a
# silently-ignored override is worse to debug than the original bug.
_acc_can_import() {
  command -v "$1" > /dev/null 2>&1 && \
    "$1" -c "import torch, transformers, numpy, scipy" > /dev/null 2>&1
}
PY=""
if [ -n "${PYTHON:-}" ]; then
  if _acc_can_import "$PYTHON"; then
    PY="$PYTHON"
  else
    echo "FATAL: PYTHON=$PYTHON cannot import torch+transformers+numpy+scipy." >&2
    echo "       Point PYTHON at the project environment, or unset it to search PATH." >&2
    exit 1
  fi
else
  for _cand in python3 python; do
    if _acc_can_import "$_cand"; then PY="$_cand"; break; fi
  done
  unset _cand
fi
if [ -z "$PY" ]; then
  echo "FATAL: no interpreter on PATH can import torch+transformers+numpy+scipy." >&2
  echo "       Activate the project environment, or set PYTHON=/path/to/python." >&2
  exit 1
fi
export PY
echo "[$(date '+%F %T')] interpreter: $("$PY" -c 'import sys; print(sys.executable)')"
