"""Voxel-level CMDF and the quantitative T2–FLAIR mismatch block (PROTOCOL.md section 3, blocks V and MM).

Images are loaded with the habitat study's `load_case` (N4, brain z-score, segmentation grid), so voxel values are on
the same scale as the patch radiomics. Results are cached per patient, keyed by a fingerprint of every setting used.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import distance_transform_edt

from habitat1p19q.config import config_fingerprint
from habitat1p19q.imaging.extraction import load_case

from .config import PAIRS
from .features import stable_int

VOXEL_KEYS = ["mismatch_threshold", "core_depth_norm", "rim_depth_mm", "min_region_voxels", "spearman_subsample"]


def voxel_fingerprint(hcfg, ccfg):
    blob = json.dumps({"habitat": config_fingerprint(hcfg), **{k: ccfg[k] for k in VOXEL_KEYS}, "pairs": PAIRS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def voxel_columns():
    cols = [f"V_{a}-{b}__{s}" for a, b in PAIRS for s in ["d_mean", "d_sd", "d_p10", "d_p50", "d_p90", "pearson", "spearman"]]
    return cols + ["MM__mismatch_frac", "MM__flair_core_minus_rim", "MM__t2_core_minus_rim"]


def voxel_features(mask, imgs, spacing, ccfg, pid):
    depth = distance_transform_edt(mask, sampling=spacing)
    dmax = float(depth.max())
    vals = {m: imgs[m][mask].astype(np.float64) for m in imgs}
    n = int(mask.sum())
    rng = np.random.default_rng(stable_int(pid))
    sub = rng.choice(n, min(n, ccfg["spearman_subsample"]), replace=False)
    f = {}
    for a, b in PAIRS:
        d = vals[a] - vals[b]
        p10, p50, p90 = np.percentile(d, [10, 50, 90])
        f.update({f"V_{a}-{b}__d_mean": d.mean(), f"V_{a}-{b}__d_sd": d.std(ddof=1), f"V_{a}-{b}__d_p10": p10,
                  f"V_{a}-{b}__d_p50": p50, f"V_{a}-{b}__d_p90": p90,
                  f"V_{a}-{b}__pearson": float(np.corrcoef(vals[a], vals[b])[0, 1]),
                  f"V_{a}-{b}__spearman": float(stats.spearmanr(vals[a][sub], vals[b][sub]).statistic)})
    t2, fl = vals["T2"], vals["FLAIR"]
    f["MM__mismatch_frac"] = float(np.mean(t2 - fl > ccfg["mismatch_threshold"]))
    dep = depth[mask]
    core, rim = dep / dmax >= ccfg["core_depth_norm"], dep <= ccfg["rim_depth_mm"]
    ok = core.sum() >= ccfg["min_region_voxels"] and rim.sum() >= ccfg["min_region_voxels"]
    f["MM__flair_core_minus_rim"] = float(fl[core].mean() - fl[rim].mean()) if ok else np.nan
    f["MM__t2_core_minus_rim"] = float(t2[core].mean() - t2[rim].mean()) if ok else np.nan
    return f, {"n_voxels": n, "max_depth_mm": dmax, "n_core": int(core.sum()), "n_rim": int(rim.sum())}


def voxel_patient(pid, paths, hcfg, ccfg, cache_dir):
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    fp = voxel_fingerprint(hcfg, ccfg)
    path = cache_dir / f"{pid}.json"
    if path.is_file():
        st = json.loads(path.read_text())
        if st.get("fingerprint") == fp:
            return st
    t0 = time.time()
    st = {"patient_id": pid, "status": "ok", "message": "", "fingerprint": fp, "features": {}}
    try:
        mask, imgs, sp, notes = load_case(paths, hcfg)
        if mask.sum() == 0:
            raise ValueError("no tumour voxels")
        st["features"], info = voxel_features(mask, imgs, sp, ccfg, pid)
        st.update(info, message="; ".join(notes))
    except Exception as exc:
        st["status"], st["message"] = "error", f"{type(exc).__name__}: {exc}"
    st["seconds"] = round(time.time() - t0, 1)
    path.write_text(json.dumps(st, default=float))
    return st


def extract_voxel_all(qc, ids, hcfg, ccfg, rep):
    """Voxel CMDF for `ids` (parallel, cached). Returns a DataFrame indexed by patient_id."""
    from joblib import Parallel, delayed
    cache_dir = ccfg["output_dir"] / "features" / "voxel"
    cache_dir.mkdir(parents=True, exist_ok=True)
    q = qc.set_index("patient_id")
    jobs = [(p, json.loads(q.loc[p, "paths"])) for p in ids]
    t0 = time.time()
    n_jobs = ccfg["voxel_n_jobs"]
    if n_jobs > 1:
        sts = Parallel(n_jobs=n_jobs, verbose=10)(delayed(voxel_patient)(p, pa, hcfg, ccfg, cache_dir) for p, pa in jobs)
    else:
        sts = [voxel_patient(p, pa, hcfg, ccfg, cache_dir) for p, pa in jobs]
    rep.say(f"Voxel CMDF finished for {len(sts)} patients in {time.time() - t0:.0f} s (cached patients are instant).")
    status = pd.DataFrame([{k: v for k, v in s.items() if k != "features"} for s in sts]).set_index("patient_id")
    rep.show(status["status"].value_counts().rename("patients"), "Voxel CMDF status:")
    if status["status"].ne("ok").any():
        rep.show(status.loc[status["status"].ne("ok"), ["message"]], "Voxel CMDF errors:")
    V = pd.DataFrame({s["patient_id"]: s["features"] for s in sts if s["status"] == "ok"}).T
    return V.reindex(columns=voxel_columns()), status
