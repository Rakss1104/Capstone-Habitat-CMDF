"""CMDF settings. Cohort, imaging and model settings are inherited from habitat1p19q so both studies share them.

Paths: the data root (env GLIOMA_DATA_ROOT), the habitat study's output folder that holds the cached patch features
(env HABITAT_OUTPUT_DIR) and this study's output folder (env CMDF_OUTPUT_DIR).
"""
from __future__ import annotations

import copy
import itertools
import os
from pathlib import Path

from habitat1p19q.config import build_config

CMDF_ROOT = Path(__file__).resolve().parents[1]

MODS = ["T1", "T1CE", "T2", "FLAIR"]
PAIRS = list(itertools.combinations(MODS, 2))          # 6 sequence pairs, fixed order

CMDF_DEFAULTS = {
    "pairing_null_reps": 100,
    "label_permutations": 200,
    "naive_permutations": 50,
    "naive_top_k": 30,
    "naive_seed": 42,                       # the original notebooks' RANDOM_SEED
    "mismatch_threshold": 1.0,              # z_T2 - z_FLAIR above this counts as mismatch
    "core_depth_norm": 0.5,
    "rim_depth_mm": 3.0,
    "min_region_voxels": 50,
    "spearman_subsample": 200_000,
    "min_patches_robustness": 4,
    "voxel_n_jobs": 3,                      # each worker holds four N4-corrected volumes (~1.5 GB)
    "cv_n_jobs": max(1, min(6, (os.cpu_count() or 2) - 1)),
    # Null tests and the naive arm run whole replicates per worker; 6 workers ran the host low on memory once.
    "null_n_jobs": max(1, min(4, (os.cpu_count() or 2) - 1)),
}


def build_cmdf_config(data_root=None, habitat_output=None, output_dir=None, overrides=None):
    """Return (habitat_cfg, cmdf_cfg). habitat_cfg points at the habitat outputs so the feature cache is reused."""
    data_root = data_root or os.environ.get("GLIOMA_DATA_ROOT", CMDF_ROOT.parent / "1P19Q -Dataset")
    habitat_output = habitat_output or os.environ.get("HABITAT_OUTPUT_DIR", CMDF_ROOT.parent / "habitat_1p19q" / "outputs")
    hcfg = build_config(data_root=str(data_root), output_dir=str(habitat_output), imaging_mode="real")
    ccfg = copy.deepcopy(CMDF_DEFAULTS)
    ccfg["output_dir"] = Path(output_dir or os.environ.get("CMDF_OUTPUT_DIR", CMDF_ROOT / "outputs")).expanduser()
    for k, v in (overrides or {}).items():
        if k not in ccfg:
            raise KeyError(f"Unknown CMDF setting: {k}")
        ccfg[k] = type(ccfg[k])(v) if not isinstance(ccfg[k], Path) else Path(v)
    return hcfg, ccfg
