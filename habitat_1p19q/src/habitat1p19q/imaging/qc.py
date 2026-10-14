"""MRI, segmentation and file quality control (notebook section 7)."""
from __future__ import annotations

import json
import re
import shutil

import numpy as np
import pandas as pd

from ..config import is_smoke
from .synthetic import make_synthetic_cohort

try:
    import SimpleITK as sitk
    HAVE_SITK = True
except ImportError:  # the cohort audit runs without imaging libraries
    sitk = None
    HAVE_SITK = False


def select_study_patients(cfg, cohort, rep):
    """Return (study table, imaging root). In smoke mode a synthetic cohort replaces the real one."""
    if is_smoke(cfg):
        assert HAVE_SITK, "SimpleITK is required for the smoke test"
        root = cfg["output_dir"] / "smoke_test_data"
        if root.exists():
            shutil.rmtree(root)
        study = make_synthetic_cohort(root, cfg)
        rep.banner()
        rep.say(f"Synthetic cohort written to {root}: {len(study)} fake patients")
    else:
        # Primary-label patients plus the UCSF IDH-mutant patients whose label is resolvable only by the WHO 2021
        # integrated diagnosis (relative co-deletion / unknown). The latter are imaged and extracted so the label
        # sensitivity analysis can use them, but in_primary=False keeps them out of every primary analysis.
        sens = cohort.sens
        extras = sens[sens["cohort"].eq("UCSF") & sens["codel_status"].isin(["relative_codeletion", "unknown"])
                      & sens["label_who2021_resolved"].notna()].copy()
        extras["label"] = extras["label_who2021_resolved"].astype(int)
        study = pd.concat([cohort.eligible.assign(in_primary=True), extras.assign(in_primary=False)], ignore_index=True)
        root = cfg["data_root"]
    if "in_primary" not in study:
        study["in_primary"] = True
    assert study["patient_id"].is_unique
    rep.show(study.groupby(["cohort", "in_primary", "label"]).size().unstack(fill_value=0),
             f"Patients entering imaging QC: {len(study)} | by cohort, primary-set membership and label:")
    return study, root


def candidate_dirs(row, cfg, imaging_root):
    coh = row["cohort"]
    root = imaging_root / cfg["imaging"][coh]["root"]
    names = [row["source_id"], row["patient_id"]]
    m = re.match(r"^UCSF-PDGM-(\d+)$", str(row["patient_id"]))
    if coh == "UCSF" and m:
        n = int(m.group(1))
        names = [f"UCSF-PDGM-{n:03d}", f"UCSF-PDGM-{n:04d}", f"UCSF-PDGM-{n:04d}_nifti"] + names
    return [root / nm for nm in dict.fromkeys(names)]


def locate_case(row, cfg, imaging_root):
    files = cfg["imaging"][row["cohort"]]["files"]
    for d in candidate_dirs(row, cfg, imaging_root):
        if d.is_dir():
            paths = {k: d / v for k, v in files.items()}
            missing = [k for k, p in paths.items() if not p.is_file()]
            return {"subject_dir": str(d), "paths": {k: str(p) for k, p in paths.items()}, "missing": missing}
    return {"subject_dir": None, "paths": {}, "missing": ["subject folder"]}


def same_geometry(a, b, tol=1e-3):
    return (a.GetSize() == b.GetSize() and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol) and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol))


def qc_case(row, cfg, imaging_root):
    out = {"patient_id": row["patient_id"], "cohort": row["cohort"], "label": row["label"]}
    loc = locate_case(row, cfg, imaging_root)
    out.update(subject_dir=loc["subject_dir"], missing=", ".join(loc["missing"]))
    if loc["missing"]:
        out["qc_status"] = "missing_files"
        return out
    try:
        seg = sitk.ReadImage(loc["paths"]["SEG"])
        arr = sitk.GetArrayViewFromImage(seg)
        out["seg_labels"] = " ".join(str(int(v)) for v in np.unique(arr))
        out["spacing"] = " x ".join(f"{s:.2f}" for s in seg.GetSpacing())
        out["size"] = " x ".join(str(s) for s in seg.GetSize())
        tum = np.isin(arr, cfg["tumour_labels"]) if cfg["tumour_labels"] else arr > 0
        out["tumour_volume_ml"] = float(tum.sum() * np.prod(seg.GetSpacing()) / 1000.0)
        mism = []
        for mod in cfg["modalities"]:
            r = sitk.ImageFileReader()
            r.SetFileName(loc["paths"][mod])
            r.ReadImageInformation()
            if not (r.GetSize() == seg.GetSize() and np.allclose(r.GetSpacing(), seg.GetSpacing(), atol=1e-3)
                    and np.allclose(r.GetOrigin(), seg.GetOrigin(), atol=1e-3)
                    and np.allclose(r.GetDirection(), seg.GetDirection(), atol=1e-3)):
                mism.append(mod)
        out["geometry_mismatch"] = ", ".join(mism)
        out["qc_status"] = "no_tumour_voxels" if out["tumour_volume_ml"] == 0 else "ok"
    except Exception as exc:
        out["qc_status"] = f"unreadable: {type(exc).__name__}: {exc}"
    out["paths"] = json.dumps(loc["paths"])
    return out


def run_qc(study, cfg, imaging_root, rep):
    """Return (qc table, imaging_available)."""
    root_exists = any((imaging_root / cfg["imaging"][c]["root"]).is_dir() for c in ["UTSW", "UCSF"])
    if not HAVE_SITK:
        rep.say("NOT EVALUATED: SimpleITK is not installed (see requirements.txt).")
        qc = pd.DataFrame()
    elif not root_exists:
        rep.say("NOT EVALUATED: no imaging folder found. Looked for:")
        for c in ["UTSW", "UCSF"]:
            rep.say(f"   {imaging_root / cfg['imaging'][c]['root']}")
        qc = pd.DataFrame()
    else:
        rep.banner()
        qc = pd.DataFrame([qc_case(r, cfg, imaging_root) for _, r in study.iterrows()])
        rep.show(qc["qc_status"].value_counts().rename("patients"), "Imaging QC status:")
        rep.show(qc.groupby(["cohort", "label", "qc_status"]).size().unstack(fill_value=0),
                 "QC status by cohort and label (patients):")
        ok = qc[qc["qc_status"].eq("ok")]
        if len(ok):
            rep.show(ok.groupby("cohort")["seg_labels"].value_counts().rename("patients"), "Segmentation label sets by cohort:")
        if (qc.get("geometry_mismatch", pd.Series(dtype=str)).fillna("") != "").any():
            rep.show(qc.loc[qc["geometry_mismatch"].fillna("") != "", ["patient_id", "cohort", "geometry_mismatch"]],
                     "Cases whose modalities do not share the segmentation grid (resampled onto it at extraction):")
        rep.save_table(qc.drop(columns=["paths"], errors="ignore"), "07_imaging_qc")
    available = (not qc.empty) and qc["qc_status"].eq("ok").any()
    rep.record_check("Imaging available for at least one eligible patient", available,
                     "imaging sections will run" if available else "imaging sections will report NOT EVALUATED",
                     status=None if available else "NOT AVAILABLE")
    return qc, bool(available)
