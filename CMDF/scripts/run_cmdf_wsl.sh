#!/usr/bin/env bash
# Runs the CMDF study inside WSL with the habitat study's environment. Usage: scripts/run_cmdf_wsl.sh [extra CLI args]
# Needs the habitat study to have been run first: its patch-feature cache is reused, not recomputed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HABITAT_VENV:-$HOME/venvs/habitat2}"
export GLIOMA_DATA_ROOT="${GLIOMA_DATA_ROOT:-$HERE/../1P19Q -Dataset}"
export HABITAT_OUTPUT_DIR="${HABITAT_OUTPUT_DIR:-$HERE/../habitat_1p19q/outputs}"
export CMDF_OUTPUT_DIR="${CMDF_OUTPUT_DIR:-$HERE/outputs}"
mkdir -p "$CMDF_OUTPUT_DIR"
cd "$HERE"
"$VENV/bin/python" -m cmdf1p19q "$@" 2>&1 | tee "$CMDF_OUTPUT_DIR/run.log"
