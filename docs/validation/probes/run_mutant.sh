#!/usr/bin/env bash
# usage: run_mutant.sh <mutation> <probe.py> [args...]   prints probe rc on the mutant
set -u
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
PY="$REPO/.venv/bin/python"; M=$1; shift; PROBE=$1; shift
before=$(cd "$REPO" && find src -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum)
T=$(mktemp -d); cp -r "$REPO/src" "$T/src"; find "$T" -name __pycache__ -prune -exec rm -rf {} +
"$PY" "$HERE/mutations.py" "$T" "$M" || { echo "MUTATION_APPLY_FAILED"; exit 99; }
PYTHONPATH="$T/src" PYTHONDONTWRITEBYTECODE=1 "$PY" -B -c "import llm_lookahead_probe as m; assert m.__file__.startswith('$T'), m.__file__" || { echo "MUTANT_NOT_LOADED"; exit 98; }
PYTHONPATH="$T/src" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$PY" -B "$HERE/$PROBE" "$@"; rc=$?
rm -rf "$T"
after=$(cd "$REPO" && find src -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum)
[ "$before" = "$after" ] && echo "src checksum unchanged" || echo "SRC CHANGED!"
echo "mutant=$M probe=$PROBE falsifier_rc=$rc"
