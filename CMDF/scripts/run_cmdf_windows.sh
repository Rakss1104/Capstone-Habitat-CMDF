#!/usr/bin/env bash
# Runs the CMDF study with a Windows Python environment (Git Bash). Needs the habitat study's patch-feature cache and,
# because PyRadiomics has no Windows build, an existing voxel cache or WSL for stage 2. Usage: scripts/run_cmdf_windows.sh [args]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${HABITAT_WIN_PY:-$HOME/venvs/habitat_win/Scripts/python.exe}"
mkdir -p "$HERE/outputs"
cd "$HERE"
PYTHONIOENCODING=utf-8 "$PY" -m cmdf1p19q "$@" 2>&1 | tee -a "$HERE/outputs/run.log"
