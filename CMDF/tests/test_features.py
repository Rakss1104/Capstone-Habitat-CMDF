import numpy as np
import pandas as pd
import pytest

from cmdf1p19q.config import MODS, PAIRS
from cmdf1p19q.features import (CMDFBuilder, cos_block, d_block, fit_sequence_scaling, model_columns, p_block,
                                pair_permuted, patient_arrays, texture_names)
from cmdf1p19q.voxel import voxel_columns, voxel_features

NAMES = ["f1", "f2", "f3"]


def toy_patches(n_pat=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pat):
        for k in range(3 + i):
            r = {"patient_id": f"P{i}", "cohort": "UTSW", "patch_idx": k}
            for m in MODS:
                for n in NAMES:
                    r[f"{m}__{n}"] = rng.normal()
            rows.append(r)
    return pd.DataFrame(rows)


def toy_base(ids):
    return pd.DataFrame({"cohort": "UTSW", "label": [i % 2 for i in range(len(ids))], "age": 50.0, "sex_M": 1.0,
                         "cohort_UCSF": 0.0, "in_primary": True}, index=ids)


def test_shapes_and_delta_definition():
    pt = toy_patches()
    names = texture_names(pt)
    arr = patient_arrays(pt, names)
    P, D = p_block(arr, names), d_block(arr, names)
    assert P.shape == (6, len(MODS) * len(NAMES) * 2)
    assert D.shape == (6, len(PAIRS) * len(NAMES) * 2)
    g = pt[pt["patient_id"] == "P2"]
    expect = (g["T2__f1"] - g["FLAIR__f1"]).agg(["mean", "std"])
    assert D.loc["P2", "D_T2-FLAIR__f1__mean"] == pytest.approx(expect["mean"])
    assert D.loc["P2", "D_T2-FLAIR__f1__sd"] == pytest.approx(expect["std"])
    assert P.loc["P2", "P_FLAIR__f3__sd"] == pytest.approx(g["FLAIR__f3"].std())


def test_single_patch_sd_is_missing():
    pt = toy_patches()
    pt = pt[~((pt["patient_id"] == "P0") & (pt["patch_idx"] > 0))]
    arr = patient_arrays(pt, NAMES)
    assert np.isnan(d_block(arr, NAMES).loc["P0", "D_T1-T2__f1__sd"])


def test_pairing_null_preserves_marginals_and_P():
    pt = toy_patches()
    arr = patient_arrays(pt, NAMES)
    X = arr["P5"]
    Y = pair_permuted(X, 3, "P5")
    assert np.array_equal(X[:, 0], Y[:, 0])
    for j in range(X.shape[1]):
        assert np.allclose(np.sort(X[:, j], 0), np.sort(Y[:, j], 0))
    D0, D1 = d_block(arr, NAMES), d_block(arr, NAMES, shuffle_seed=3)
    mean_cols = [c for c in D0 if c.endswith("__mean")]
    assert np.allclose(D0[mean_cols], D1[mean_cols])          # means of differences depend on marginals only
    assert not np.allclose(D0.drop(columns=mean_cols), D1.drop(columns=mean_cols), equal_nan=True)


def test_cosine_scaling_uses_training_patients_only():
    pt = toy_patches()
    arr = patient_arrays(pt, NAMES)
    train, test = ["P0", "P1", "P2", "P3"], ["P4", "P5"]
    arr_shift = {k: (v + 100 if k in test else v) for k, v in arr.items()}
    s1, s2 = fit_sequence_scaling(arr, train), fit_sequence_scaling(arr_shift, train)
    assert np.allclose(s1[0], s2[0]) and np.allclose(s1[1], s2[1])
    c = cos_block(arr, train + test, s1)
    assert ((c.filter(like="__mean") >= -1 - 1e-9) & (c.filter(like="__mean") <= 1 + 1e-9)).all().all()


def test_builder_and_model_columns():
    pt = toy_patches()
    arr = patient_arrays(pt, NAMES)
    ids = list(arr)
    b = CMDFBuilder(toy_base(ids), arr, NAMES)
    Ftr, Fte, info = b.build(ids[:4], ids[4:])
    assert info["scale_fit_patients"] == ids[:4] and list(Fte.index) == ids[4:]
    assert len(model_columns("P", Ftr)) == 24
    assert len(model_columns("D", Ftr)) == 36 + 12
    assert len(model_columns("D_T2FL", Ftr)) == 6 + 2
    assert set(model_columns("P+D", Ftr)) == set(model_columns("P", Ftr)) | set(model_columns("D", Ftr))
    assert model_columns("M1_clin+cohort", Ftr) == ["age", "sex_M", "cohort_UCSF"]
    assert not any(c.startswith("D_T2-FLAIR__") for c in model_columns("P+D", Ftr, ("D_T2-FLAIR__",)))


def test_voxel_mismatch_on_synthetic_volume():
    z, y, x = np.mgrid[:30, :30, :30]
    mask = ((z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2) <= 10 ** 2
    rng = np.random.default_rng(1)
    base = rng.normal(size=mask.shape).astype(np.float32)
    imgs = {"T1": base, "T1CE": base * 0.5, "T2": base + 2.0, "FLAIR": base.copy()}
    imgs["FLAIR"][mask] -= 0.0
    cfg = {"mismatch_threshold": 1.0, "core_depth_norm": 0.5, "rim_depth_mm": 3.0, "min_region_voxels": 50,
           "spearman_subsample": 5000}
    f, info = voxel_features(mask, imgs, (1.0, 1.0, 1.0), cfg, "toy")
    assert set(f) == set(voxel_columns())
    assert f["MM__mismatch_frac"] == pytest.approx(1.0)           # T2 exceeds FLAIR by 2 z-units everywhere
    assert f["V_T2-FLAIR__d_mean"] == pytest.approx(2.0, abs=1e-5)
    assert f["V_T1-T2__pearson"] == pytest.approx(1.0)
    assert info["n_core"] > 0 and info["n_rim"] > 0
