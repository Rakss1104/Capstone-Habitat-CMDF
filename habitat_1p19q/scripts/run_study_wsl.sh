#!/usr/bin/env bash
# Runs the full study on the real data inside WSL. Usage: scripts/run_study_wsl.sh [extra CLI args]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HABITAT_VENV:-$HOME/venvs/habitat2}"
DATA="${GLIOMA_DATA_ROOT:-$HERE/../1P19Q -Dataset}"
OUT="${GLIOMA_OUTPUT_DIR:-$HERE/outputs}"
mkdir -p "$OUT"
cd "$HERE"
"$VENV/bin/python" -m habitat1p19q --data-root "$DATA" --output-dir "$OUT" --mode real "$@" 2>&1 | tee "$OUT/run.log"
