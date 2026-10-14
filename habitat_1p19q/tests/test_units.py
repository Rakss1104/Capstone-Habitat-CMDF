import numpy as np
import pandas as pd
import pytest

from habitat1p19q import stats as st
from habitat1p19q.config import build_config, config_fingerprint
from habitat1p19q.imaging.extraction import axis_starts
from habitat1p19q.modeling import model_columns
from habitat1p19q.spatial import composition_cols, grid_pairs, spatial_cols, toy_grid_self_test


def test_toy_grid_spatial_metrics_separate_composition_equal_arrangements():
    table, ok = toy_grid_self_test(stride=16, min_patches=4)
    assert ok, table


def test_grid_pairs_counts_face_neighbours():
    starts = np.array([[0, 0, 0], [16, 0, 0], [0, 16, 0], [32, 32, 32]])
    i, j = grid_pairs(starts, 16)
    assert sorted(zip(i, j)) == [(0, 1), (0, 2)]


def test_wilson_ci_brackets_proportion():
    lo, hi = st.wilson_ci(9, 93)
    assert 0 < lo < 9 / 93 < hi < 1


def test_bh_fdr_is_monotone_and_bounded():
    q = st.bh_fdr([0.01, 0.04, 0.03, 0.5])
    assert np.all(q <= 1) and np.all(q >= np.array([0.01, 0.04, 0.03, 0.5]))


def test_nadeau_bengio_wider_than_naive():
    d = np.random.default_rng(0).normal(0.02, 0.05, 50)
    nb = st.nadeau_bengio(d, n_train=112, n_test=28)
    naive_half = 2.01 * d.std(ddof=1) / np.sqrt(len(d))
    assert (nb["hi"] - nb["lo"]) / 2 > naive_half


def test_axis_starts_cover_extent_and_stay_in_volume():
    s = axis_starts(10, 90, 32, 100, 16)
    assert s[0] == 10 and s[-1] + 32 >= 90 and max(s) + 32 <= 100


def test_model_columns_ladder_nests():
    K = 4
    F = pd.DataFrame(columns=["age", "sex_M", "log_volume_ml", "cohort_UCSF", "wt_a"] + composition_cols(K) + spatial_cols(K))
    m3, m4, m5 = (set(model_columns(m, F, K)) for m in ["M3", "M4", "M5"])
    assert m3 < m4 < m5
    assert model_columns("M3", F, None) is None
    assert model_columns("M4+cohort", F, K)[-1] == "cohort_UCSF"


def test_config_overrides_and_fingerprint():
    a = build_config("d", "o", "real")
    b = build_config("d", "o", "real", {"patch_size": 16})
    assert config_fingerprint(a) != config_fingerprint(b)
    with pytest.raises(KeyError):
        build_config("d", "o", "real", {"no_such_key": 1})
