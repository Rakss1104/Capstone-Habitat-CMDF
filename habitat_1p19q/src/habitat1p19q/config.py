"""Study configuration.

Every setting lives in one nested dict, as in the notebook. Paths come from arguments or the environment variables
GLIOMA_DATA_ROOT, GLIOMA_OUTPUT_DIR and GLIOMA_IMAGING_MODE, so no personal path is hard-coded.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

DEFAULT_CONFIG = {
    "seed": 20261012,
    "data_root": "data",
    "output_dir": "outputs",
    "imaging_mode": "real",                 # "real" | "synthetic_smoke_test"
    "utsw_metadata": "UTSW_Glioma_Metadata.tsv",
    "ucsf_metadata": "UCSF-PDGM-metadata_v5.csv",
    # Imaging layout (from iteration-2 cell 13). {pid} = canonical ID; UCSF also tries 3/4-digit variants.
    "imaging": {
        "UTSW": {"root": "UTSW_processed",
                 "files": {"T1": "T1.nii.gz", "T1CE": "T1_GD.nii.gz", "T2": "T2.nii.gz",
                           "FLAIR": "FLAIR.nii.gz", "SEG": "segmentation.nii.gz"}},
        "UCSF": {"root": "UCSF/images",
                 "files": {"T1": "T1.nii.gz", "T1CE": "T1_GD.nii.gz", "T2": "T2.nii.gz",
                           "FLAIR": "FLAIR.nii.gz", "SEG": "seg_mask.nii.gz"}},
    },
    "modalities": ["T1", "T1CE", "T2", "FLAIR"],
    "tumour_labels": None,                  # None = every non-zero segmentation label (whole abnormality)
    "n4_bias_correction": True,
    "target_spacing": (1.0, 1.0, 1.0),
    "patch_size": 32, "patch_stride": 16, "min_tumour_fraction": 0.25,
    "bin_width": 0.25,
    "texture_classes": ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"],
    "n_jobs": max(1, (os.cpu_count() or 2) - 1),
    # Extraction workers each hold four full volumes plus N4/PyRadiomics buffers (~2 GB); 7 workers exhausted an
    # 11 GB WSL VM, so extraction gets its own, lower cap.
    "extraction_n_jobs": 3,
    "habitat": {"pca_variance": 0.95, "k_grid": [3, 4, 5, 6, 7, 8], "stability_bootstraps": 10,
                "kmeans_n_init": 10, "selection_n_init": 3, "min_habitat_share": 0.02,
                "stability_tolerance": 0.02, "fixed_k": None, "silhouette_sample": 3000},
    "spatial": {"min_patches": 4, "n_inner_null": 200},
    "cv": {"n_splits": 5, "n_repeats": 10, "inner_splits": 5},
    "logreg_C_grid": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    "rf": {"n_estimators": 500, "max_depth": 4, "min_samples_leaf": 5, "class_weight": "balanced_subsample"},
    "n_bootstrap": 2000,
    "n_label_permutations": 200,
    "n_spatial_null": 100,
    "min_train_events_loco": 20,
    "k_sensitivity": [4, 5, 6, 7, 8],
    "harmonisation": None,                  # None | "per_cohort_zscore" (sensitivity only)
    "primary": {"analysis_set": "UTSW", "test": "M4", "reference": "M3", "estimator": "logreg"},
    "colors": {"non-codeleted": "#2a78d6", "codeleted": "#eb6834", "UTSW": "#6250d6", "UCSF": "#1baf7a",
               "models": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]},
}

# Small, fast settings that exercise every code path. They are NOT study settings.
SMOKE_OVERRIDES = {
    "patch_size": 16, "patch_stride": 8, "n4_bias_correction": False,
    "n_bootstrap": 200, "n_label_permutations": 10, "n_spatial_null": 5, "k_sensitivity": [3, 4],
    "cv": {"n_splits": 3, "n_repeats": 2, "inner_splits": 3},
    "min_train_events_loco": 5,
}
SMOKE_HABITAT = {"k_grid": [2, 3, 4], "stability_bootstraps": 3, "kmeans_n_init": 4, "selection_n_init": 2}

# Settings that change extracted features; a change invalidates the per-patient cache.
FINGERPRINT_KEYS = ["modalities", "tumour_labels", "n4_bias_correction", "target_spacing", "patch_size", "patch_stride",
                    "min_tumour_fraction", "bin_width", "texture_classes", "imaging_mode"]


def build_config(data_root=None, output_dir=None, imaging_mode=None, overrides=None):
    """Return a fully resolved config: defaults <- environment <- explicit arguments <- overrides."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["data_root"] = data_root or os.environ.get("GLIOMA_DATA_ROOT", cfg["data_root"])
    cfg["output_dir"] = output_dir or os.environ.get("GLIOMA_OUTPUT_DIR", cfg["output_dir"])
    cfg["imaging_mode"] = imaging_mode or os.environ.get("GLIOMA_IMAGING_MODE", cfg["imaging_mode"])
    if cfg["imaging_mode"] not in {"real", "synthetic_smoke_test"}:
        raise ValueError(f"imaging_mode must be 'real' or 'synthetic_smoke_test', got {cfg['imaging_mode']!r}")
    if cfg["imaging_mode"] == "synthetic_smoke_test":
        cfg.update(copy.deepcopy(SMOKE_OVERRIDES))
        cfg["n_jobs"] = min(4, cfg["n_jobs"])
        cfg["habitat"].update(SMOKE_HABITAT)
        cfg["spatial"]["n_inner_null"] = 50
    for key, value in (overrides or {}).items():
        _set_nested(cfg, key, value)
    cfg["data_root"] = Path(cfg["data_root"]).expanduser()
    cfg["output_dir"] = Path(cfg["output_dir"]).expanduser()
    return cfg


def _set_nested(cfg, dotted_key, value):
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"Unknown config key: {dotted_key}")
    node[parts[-1]] = value


def is_smoke(cfg):
    return cfg["imaging_mode"] == "synthetic_smoke_test"


def config_fingerprint(cfg):
    blob = json.dumps({k: cfg[k] for k in FINGERPRINT_KEYS}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def prepare_output_dirs(cfg):
    out = cfg["output_dir"]
    for sub in ["tables", "figures", "cache", "features"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out
