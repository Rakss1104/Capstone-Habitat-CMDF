#!/usr/bin/env bash
# Builds a Linux virtual environment for the pipeline (PyRadiomics has no Windows wheel).
# uv supplies a standalone CPython that ships its own headers, so no system packages are needed.
set -euo pipefail
VENV="${HABITAT_VENV:-$HOME/venvs/habitat2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOT="$HOME/venvs/uv-bootstrap"
[ -x "$BOOT/bin/uv" ] || { python3 -m venv "$BOOT" && "$BOOT/bin/pip" install -q uv; }
UV="$BOOT/bin/uv"
"$UV" venv -q --allow-existing --python 3.12 --python-preference only-managed "$VENV"
VIRTUAL_ENV="$VENV" "$UV" pip install -q -r "$HERE/requirements.txt"
VIRTUAL_ENV="$VENV" "$UV" pip install -q -e "$HERE"
"$VENV/bin/python" -c "import platform, radiomics, numpy, pandas, sklearn; print('python', platform.python_version(), '| radiomics', radiomics.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| sklearn', sklearn.__version__)"
