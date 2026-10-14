"""Patch construction and radiomics feature extraction (notebook section 8).

- Shape features are computed once per patch (`shape__*`); intensity/texture per modality (`<MOD>__<feature>`).
- Shape features are excluded from habitat clustering and kept only as descriptors.
- A Euclidean distance transform of the tumour mask gives each patch an anatomical depth (mm to tumour boundary).
- Whole-tumour radiomics (`wt_*`) and tumour volume are extracted for every readable patient, even without patches.
- Optional N4 bias-field correction before brain z-scoring.
- Per-patient caching keyed by a configuration fingerprint.
"""
from __future__ import annotations

import itertools
import json
import logging
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import distance_transform_edt

from ..config import config_fingerprint
from ..cohort import CLASS_NAME
from .qc import same_geometry

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
try:
    from radiomics import featureextractor
    HAVE_RADIOMICS = True
except ImportError:
    featureextractor = None
    HAVE_RADIOMICS = False

PATCH_META = ["patient_id", "cohort", "patch_idx", "z0", "y0", "x0", "n_tumour_vox", "tumour_fraction",
              "depth_mean_mm", "depth_norm"]


def n4_correct(img, mask):
    shrink = 4
    small = sitk.Shrink(img, [shrink] * img.GetDimension())
    small_mask = sitk.Shrink(mask, [shrink] * img.GetDimension())
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.SetMaximumNumberOfIterations([50, 50, 30])
    corr.Execute(small, small_mask)
    log_bias = corr.GetLogBiasFieldAsImage(img)
    return sitk.Cast(img / sitk.Exp(log_bias), sitk.sitkFloat32)


def resample(img, reference=None, spacing=None, interpolator=None):
    f = sitk.ResampleImageFilter()
    f.SetInterpolator(interpolator if interpolator is not None else sitk.sitkLinear)
    f.SetDefaultPixelValue(0)
    if reference is not None:
        f.SetReferenceImage(reference)
    else:
        in_sp, in_sz = np.array(img.GetSpacing()), np.array(img.GetSize())
        out_sp = np.array(spacing, float)
        f.SetOutputSpacing(out_sp.tolist())
        f.SetSize([int(s) for s in np.maximum(np.round(in_sz * in_sp / out_sp), 1)])
        f.SetOutputOrigin(img.GetOrigin())
        f.SetOutputDirection(img.GetDirection())
    return f.Execute(img)


def load_case(paths, cfg):
    """Return the tumour mask and brain-z-scored modality arrays in (z, y, x) order, spacing (z, y, x), and notes."""
    notes = []
    seg = sitk.ReadImage(paths["SEG"])
    if not np.allclose(seg.GetSpacing(), cfg["target_spacing"], atol=1e-3):
        seg = resample(seg, spacing=cfg["target_spacing"], interpolator=sitk.sitkNearestNeighbor)
        notes.append("segmentation resampled to target spacing")
    seg_arr = sitk.GetArrayFromImage(seg)
    mask = np.isin(seg_arr, cfg["tumour_labels"]) if cfg["tumour_labels"] else seg_arr > 0
    imgs = {}
    for mod in cfg["modalities"]:
        img = sitk.ReadImage(paths[mod], sitk.sitkFloat32)
        if not same_geometry(img, seg):
            img = resample(img, reference=seg)
            notes.append(f"{mod} resampled onto segmentation grid")
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        brain = arr != 0
        if brain.sum() < 1000:
            raise ValueError(f"{mod}: fewer than 1000 non-zero voxels; is the image skull-stripped and non-empty?")
        if cfg["n4_bias_correction"]:
            bm = sitk.GetImageFromArray(brain.astype(np.uint8))
            bm.CopyInformation(img)
            arr = sitk.GetArrayFromImage(n4_correct(img, bm)).astype(np.float32)
            arr[~brain] = 0
        mu, sd = float(arr[brain].mean()), float(arr[brain].std())
        if sd < 1e-8:
            raise ValueError(f"{mod}: no intensity variation inside the brain")
        imgs[mod] = np.where(brain, (arr - mu) / sd, 0).astype(np.float32)
    spacing_zyx = tuple(reversed(seg.GetSpacing()))
    return mask, imgs, spacing_zyx, notes


def axis_starts(lo, hi, size, dim, stride):
    if dim < size:
        raise ValueError(f"volume axis ({dim}) smaller than patch size ({size})")
    if hi - lo <= size:
        starts = [int(round((lo + hi) / 2 - size / 2))]
    else:
        starts = list(range(lo, hi - size + 1, stride))
        if starts[-1] + size < hi:
            starts.append(hi - size)
    return sorted({min(max(s, 0), dim - size) for s in starts})


def generate_patches(mask, depth, cfg):
    P = cfg["patch_size"]
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    axes = [axis_starts(int(lo[a]), int(hi[a]), P, mask.shape[a], cfg["patch_stride"]) for a in range(3)]
    n_vox = P ** 3
    kept, n_cand = [], 0
    for s in itertools.product(*axes):
        n_cand += 1
        sl = tuple(slice(a, a + P) for a in s)
        win = mask[sl]
        n_t = int(win.sum())
        if n_t / n_vox >= cfg["min_tumour_fraction"] - 1e-12:
            kept.append({"z0": s[0], "y0": s[1], "x0": s[2], "n_tumour_vox": n_t, "tumour_fraction": n_t / n_vox,
                         "depth_mean_mm": float(depth[sl][win].mean())})
    return kept, n_cand


def make_extractors(cfg):
    settings = dict(binWidth=cfg["bin_width"], normalize=False, resampledPixelSpacing=None, force2D=False,
                    distances=[1], symmetricalGLCM=True, voxelArrayShift=0, label=1, additionalInfo=False,
                    geometryTolerance=1e-3)

    def make(classes):
        ext = featureextractor.RadiomicsFeatureExtractor(**settings)
        ext.disableAllFeatures()
        for c in classes:
            ext.enableFeatureClassByName(c)
        ext.disableAllImageTypes()
        ext.enableImageTypeByName("Original")
        return ext
    return make(cfg["texture_classes"]), make(["shape"])


def to_sitk(arr, spacing_zyx):
    im = sitk.GetImageFromArray(arr)
    im.SetSpacing(tuple(float(s) for s in reversed(spacing_zyx)))
    return im


def run_ext(ext, img_arr, mask_arr, spacing, prefix):
    res = ext.execute(to_sitk(img_arr, spacing), to_sitk(mask_arr.astype(np.uint8), spacing), label=1)
    # Some PyRadiomics features (e.g. GLCM MCC) can come back as complex numbers with ~0 imaginary part
    return {f"{prefix}{k}": float(np.real(v)) for k, v in res.items() if not k.startswith("diagnostics_")}


def extract_patient(row_dict, paths, cfg):
    """One patient: patch features + whole-tumour features + volume. Writes a cache file and returns a status dict."""
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    pid, coh = row_dict["patient_id"], row_dict["cohort"]
    fp = config_fingerprint(cfg)
    cdir = cfg["output_dir"] / "features" / "per_patient"
    cdir.mkdir(parents=True, exist_ok=True)
    csv_path, st_path = cdir / f"{pid}.csv", cdir / f"{pid}.json"
    if st_path.is_file():
        st = json.loads(st_path.read_text())
        if st.get("fingerprint") == fp and (st["status"] != "ok" or csv_path.is_file()):
            return st
    t0 = time.time()
    st = {"patient_id": pid, "cohort": coh, "status": "ok", "message": "", "fingerprint": fp, "n_candidates": 0,
          "n_patches": 0, "n_failed_patches": 0, "tumour_volume_ml": np.nan, "max_depth_mm": np.nan, "wt": {}}
    try:
        mask, imgs, sp, notes = load_case(paths, cfg)
        st["message"] = "; ".join(notes)
        st["tumour_volume_ml"] = float(mask.sum() * np.prod(sp) / 1000.0)
        if mask.sum() == 0:
            raise ValueError("no tumour voxels")
        depth = distance_transform_edt(mask, sampling=sp).astype(np.float32)
        st["max_depth_mm"] = float(depth.max())
        tex_ext, shape_ext = make_extractors(cfg)
        mods = cfg["modalities"]
        # whole-tumour features (cropped to the bounding box + 2 voxels)
        idx = np.argwhere(mask)
        lo = np.maximum(idx.min(0) - 2, 0)
        hi = np.minimum(idx.max(0) + 3, mask.shape)
        bb = tuple(slice(a, b) for a, b in zip(lo, hi))
        wt = run_ext(shape_ext, imgs[mods[0]][bb], mask[bb], sp, "wt_shape__")
        for mod in mods:
            wt.update(run_ext(tex_ext, imgs[mod][bb], mask[bb], sp, f"wt_{mod}__"))
        st["wt"] = wt
        patches, n_cand = generate_patches(mask, depth, cfg)
        st["n_candidates"] = n_cand
        rows = []
        P = cfg["patch_size"]
        for k, p in enumerate(patches):
            sl = tuple(slice(p[a], p[a] + P) for a in ["z0", "y0", "x0"])
            try:
                feats = run_ext(shape_ext, imgs[mods[0]][sl], mask[sl], sp, "shape__")
                for mod in mods:
                    feats.update(run_ext(tex_ext, imgs[mod][sl], mask[sl], sp, f"{mod}__"))
            except Exception:
                st["n_failed_patches"] += 1
                continue
            rows.append({"patient_id": pid, "cohort": coh, "patch_idx": k, **p,
                         "depth_norm": p["depth_mean_mm"] / st["max_depth_mm"], **feats})
        st["n_patches"] = len(rows)
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        else:
            st["status"] = "no_patches"
            st["message"] += f"; 0/{n_cand} candidate patches reached tumour fraction {cfg['min_tumour_fraction']}"
    except Exception as exc:
        st["status"] = "error"
        st["message"] = f"{type(exc).__name__}: {exc}"
    st["seconds"] = round(time.time() - t0, 1)
    st_path.write_text(json.dumps(st, default=float))
    return st


def extract_all(study, qc, cfg, rep):
    """Run extraction for every QC-passing patient (parallel, cached). Returns (patient_img, patches)."""
    if not HAVE_RADIOMICS:
        raise ImportError("PyRadiomics is required for feature extraction; install it from requirements.txt")
    from joblib import Parallel, delayed

    rep.banner()
    todo = qc[qc["qc_status"].eq("ok")]
    by_id = study.set_index("patient_id")
    jobs = [(by_id.loc[r["patient_id"]].to_dict() | {"patient_id": r["patient_id"]}, json.loads(r["paths"]))
            for _, r in todo.iterrows()]
    t0 = time.time()
    n_jobs = min(cfg["n_jobs"], cfg["extraction_n_jobs"])
    if n_jobs > 1:
        statuses = Parallel(n_jobs=n_jobs, verbose=10)(delayed(extract_patient)(rd, p, cfg) for rd, p in jobs)
    else:
        statuses = [extract_patient(rd, p, cfg) for rd, p in jobs]
    rep.say(f"Extraction finished for {len(statuses)} patients in {time.time() - t0:.0f} s (cached patients are instant).")
    patient_img = pd.DataFrame([{k: v for k, v in s.items() if k != "wt"} for s in statuses])
    wt = pd.DataFrame([s["wt"] for s in statuses], index=[s["patient_id"] for s in statuses])
    patient_img = patient_img.set_index("patient_id").join(wt)
    frames = [pd.read_csv(cfg["output_dir"] / "features" / "per_patient" / f"{s['patient_id']}.csv")
              for s in statuses if s["status"] == "ok"]
    patches = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rep.show(patient_img["status"].value_counts().rename("patients"), "Extraction status:")
    if (patient_img["status"] == "error").any():
        rep.show(patient_img.loc[patient_img["status"].eq("error"), ["cohort", "message"]], "Extraction errors:")
    rep.say(f"Patches: {len(patches)} from {patches['patient_id'].nunique() if len(patches) else 0} patients")
    return patient_img, patches


def exclusion_accounting(study, qc, patient_img, patches, cfg, rep):
    """Feature-naming checks and exclusion accounting by class. Returns the habitat-evaluable patient IDs."""
    rep.banner()
    mods = cfg["modalities"]
    shape_cols = [c for c in patches.columns if c.startswith("shape__")]
    n_per_mod = {m: len([c for c in patches.columns if c.startswith(f"{m}__")]) for m in mods}
    rep.record_check("Shape features stored once per patch (not per modality)",
                     not any(c.startswith(f"{m}__") and "_shape_" in c for m in mods for c in patches.columns),
                     f"{len(shape_cols)} shape columns")
    rep.record_check("Same number of features for every modality", len(set(n_per_mod.values())) == 1, json.dumps(n_per_mod))
    rep.record_check("No duplicated patch rows", not patches.duplicated(["patient_id", "patch_idx"]).any())

    acc = (study.set_index("patient_id")[["cohort", "label", "in_primary"]]
           .join(qc.set_index("patient_id")[["qc_status"]])
           .join(patient_img[["status", "n_patches", "tumour_volume_ml"]]))
    acc["outcome"] = np.select([acc["qc_status"].ne("ok"), acc["status"].ne("ok")],
                               ["excluded: imaging QC", "excluded: no valid patch / extraction error"], "habitat-evaluable")
    evaluable_ids = acc.index[acc["outcome"].eq("habitat-evaluable")].tolist()   # includes label-sensitivity extras
    acc = acc[acc["in_primary"].astype(bool)]                                     # accounting on the primary label set
    exclusion = pd.crosstab([acc["cohort"], acc["outcome"]], acc["label"].map(CLASS_NAME))
    rep.show(exclusion, "Exclusions by cohort and class (primary label set):")
    rep.save_table(exclusion, "08_exclusions_by_class")
    exc_any = acc["outcome"].ne("habitat-evaluable")
    tab = pd.crosstab(exc_any, acc["label"]).reindex(index=[True, False], columns=[1, 0], fill_value=0)
    if tab.shape == (2, 2):
        p_exc = stats.fisher_exact(tab.values)[1]
        rep.say(f"Excluded: {tab.loc[True, 1]}/{tab[1].sum()} codeleted vs {tab.loc[True, 0]}/{tab[0].sum()} "
                f"non-codeleted (Fisher p = {p_exc:.3g})")
    vol = acc.dropna(subset=["tumour_volume_ml"])
    if len(vol):
        rep.show(vol.groupby(["outcome", "label"])["tumour_volume_ml"].describe()[["count", "25%", "50%", "75%"]],
                 "Tumour volume (mL) by evaluability and class:", round_to=1)
    return evaluable_ids
