"""Command-line entry point: `python -m habitat1p19q --data-root ... --output-dir ...`."""
from __future__ import annotations

import argparse
import json
import logging
import warnings

from .config import build_config
from .pipeline import run_study


def _parse_override(text):
    key, _, raw = text.partition("=")
    if not key or not _:
        raise argparse.ArgumentTypeError(f"--set expects key=value, got {text!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def main(argv=None):
    p = argparse.ArgumentParser(prog="habitat1p19q", description=__doc__)
    p.add_argument("--data-root", help="Folder with the two metadata files and the imaging folders (env GLIOMA_DATA_ROOT)")
    p.add_argument("--output-dir", help="Where tables, figures, caches and the manifest go (env GLIOMA_OUTPUT_DIR)")
    p.add_argument("--mode", choices=["real", "synthetic_smoke_test"], help="Imaging mode (env GLIOMA_IMAGING_MODE)")
    p.add_argument("--n-jobs", type=int, help="Worker processes for extraction and null tests")
    p.add_argument("--set", dest="overrides", action="append", type=_parse_override, default=[],
                   metavar="KEY=VALUE", help="Override any config key, dotted for nesting, JSON values "
                                             "(e.g. --set cv.n_repeats=5 --set n_label_permutations=100)")
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    overrides = dict(args.overrides)
    if args.n_jobs:
        overrides["n_jobs"] = args.n_jobs
    cfg = build_config(args.data_root, args.output_dir, args.mode, overrides)
    run_study(cfg)


if __name__ == "__main__":
    main()
