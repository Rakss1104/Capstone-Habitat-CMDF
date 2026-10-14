"""Command line: python -m cmdf1p19q [--data-root D] [--habitat-output H] [--output-dir O] [--stages 1,2,...]"""
from __future__ import annotations

import argparse

from .config import build_cmdf_config
from .study import CMDFStudy


def main(argv=None):
    p = argparse.ArgumentParser(prog="cmdf1p19q", description=__doc__)
    p.add_argument("--data-root", help="Folder with the metadata files and imaging (env GLIOMA_DATA_ROOT)")
    p.add_argument("--habitat-output", help="habitat_1p19q output folder holding the patch-feature cache (env HABITAT_OUTPUT_DIR)")
    p.add_argument("--output-dir", help="Where this study writes its outputs (env CMDF_OUTPUT_DIR; default CMDF/outputs)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="Override a CMDF setting")
    p.add_argument("--stages", help="Comma-separated stage numbers to run (default: all; later stages need earlier ones)")
    p.add_argument("--resume", action="store_true", help="Continue after the last checkpointed stage")
    a = p.parse_args(argv)
    overrides = dict(kv.split("=", 1) for kv in a.overrides)
    hcfg, ccfg = build_cmdf_config(a.data_root, a.habitat_output, a.output_dir, overrides)
    CMDFStudy(hcfg, ccfg).run(stages=a.stages.split(",") if a.stages else None, resume=a.resume)


if __name__ == "__main__":
    main()
