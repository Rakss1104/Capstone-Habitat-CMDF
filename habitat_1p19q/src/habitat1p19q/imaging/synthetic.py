"""Synthetic smoke-test data (software verification only; used when imaging_mode == "synthetic_smoke_test")."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_cohort(root, cfg, n_utsw=30, n_ucsf=18, shape=(60, 60, 60), seed=7):
    """Small skull-stripped-looking volumes with an ellipsoid tumour made of two compartments of equal volume.

    The inner sphere (r <= 0.794) holds half the volume. Non-codeleted: profile X inside, Y outside; codeleted:
    swapped. Voxel composition is the same in both classes; only the radial arrangement differs. This exercises the
    habitat, depth and spatial-metric code on data with real within-tumour structure.
    """
    import SimpleITK as sitk

    rng = np.random.default_rng(seed)
    rows = []
    zz, yy, xx = np.indices(shape)
    c = np.array(shape) / 2
    brain = ((zz - c[0]) / 27) ** 2 + ((yy - c[1]) / 26) ** 2 + ((xx - c[2]) / 25) ** 2 <= 1
    prof_x = {"T1": -0.5, "T1CE": -0.2, "T2": 1.6, "FLAIR": -0.4}
    prof_y = {"T1": -0.3, "T1CE": 0.3, "T2": 0.8, "FLAIR": 1.2}
    for coh, n, prev in [("UTSW", n_utsw, 0.45), ("UCSF", n_ucsf, 0.2)]:
        n_pos = max(3, int(round(n * prev)))
        labels = np.array([1] * n_pos + [0] * (n - n_pos))
        rng.shuffle(labels)
        for i in range(n):
            pid = f"SYN-{coh}-{i + 1:03d}"
            lab = int(labels[i])
            small = (i == 0)                     # one tiny tumour per cohort: exercises the no-patch path
            radii = rng.uniform(3, 4, 3) if small else rng.uniform(12, 16, 3)
            centre = c + rng.uniform(-4, 4, 3)
            r = np.sqrt(((zz - centre[0]) / radii[0]) ** 2 + ((yy - centre[1]) / radii[1]) ** 2 + ((xx - centre[2]) / radii[2]) ** 2)
            tumour = (r <= 1) & brain
            inner = (r <= 0.794) & tumour
            outer = tumour & ~inner
            p_in, p_out = (prof_y, prof_x) if lab else (prof_x, prof_y)
            vols = {}
            for mod in cfg["modalities"]:
                v = np.where(brain, 1.0 + 0.08 * rng.standard_normal(shape), 0.0)
                v = np.where(inner, v + p_in[mod], v)
                v = np.where(outer, v + p_out[mod], v)
                vols[mod] = np.where(brain, np.clip(v, 0.05, None), 0.0).astype(np.float32)
            seg = np.zeros(shape, np.uint8)
            seg[outer] = 2
            seg[inner] = 1
            files = cfg["imaging"][coh]["files"]
            d = Path(root) / cfg["imaging"][coh]["root"] / pid
            d.mkdir(parents=True, exist_ok=True)
            for mod, arr in vols.items():
                if coh == "UCSF" and i == 1 and mod == "FLAIR":
                    continue                     # one missing modality: exercises the missing-file path
                im = sitk.GetImageFromArray(arr)
                im.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(im, str(d / files[mod]))
            im = sitk.GetImageFromArray(seg)
            im.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(im, str(d / files["SEG"]))
            rows.append({"cohort": coh, "source_id": pid, "patient_id": pid, "label": lab,
                         "age": float(rng.integers(25, 70)), "sex": rng.choice(["M", "F"]), "grade": np.nan,
                         "field_strength_T": np.nan, "seg_provenance": "synthetic", "diagnosis": "synthetic",
                         "codel_status": "codeleted" if lab else "non-codeleted", "idh_status": "mutant",
                         "biopsy_prior_to_imaging": False})
    return pd.DataFrame(rows)
