"""Patient-level feature blocks from the cached patch radiomics (PROTOCOL.md section 3).

P   single-sequence patch radiomics: per sequence x feature, mean and SD over the patient's patches.
D   patch CMDF: per sequence pair x feature, Δ = A − B on the same patch; mean and SD of Δ over patches.
cos per pair, cosine similarity of the A and B feature profiles of each patch after per-sequence standardisation with
    training-fold statistics; mean and SD over patches. This is the only block fitted on data, so it is rebuilt per fold.

The pairing null permutes the patch order of every sequence but the first, independently per patient. Each sequence's
own distribution (hence P) is unchanged; only the co-location of features across sequences is lost.
"""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd

from .config import MODS, PAIRS

STATS = ("mean", "sd")


def stable_int(text):
    return int(hashlib.sha1(str(text).encode()).hexdigest()[:8], 16)


def texture_names(patches):
    per = {m: [c[len(m) + 2:] for c in patches.columns if c.startswith(f"{m}__")] for m in MODS}
    names = per[MODS[0]]
    assert names and all(per[m] == names for m in MODS), "sequences do not share the same feature list"
    return names


def patient_arrays(patches, names, ids=None):
    """{patient_id: array (n_patches, n_sequences, n_features)} in patch order."""
    cols = [f"{m}__{n}" for m in MODS for n in names]
    sub = patches if ids is None else patches[patches["patient_id"].isin(ids)]
    out = {}
    for pid, g in sub.groupby("patient_id", sort=False):
        g = g.sort_values("patch_idx")
        out[pid] = g[cols].to_numpy(float).reshape(len(g), len(MODS), len(names))
    return out


def mean_sd(X):
    """Mean and SD over axis 0; the SD of a single patch is undefined (NaN, imputed later inside the fold)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(X, 0)
        sd = np.nanstd(X, 0, ddof=1) if len(X) > 1 else np.full(X.shape[1:], np.nan)
    return mean, sd


def pair_permuted(X, seed, pid):
    rng = np.random.default_rng([int(seed), stable_int(pid)])
    Y = X.copy()
    for j in range(1, X.shape[1]):
        Y[:, j, :] = X[rng.permutation(len(X)), j, :]
    return Y


def p_columns(names):
    return [f"P_{m}__{n}__{s}" for m in MODS for s in STATS for n in names]


def d_columns(names):
    return [f"D_{a}-{b}__{n}__{s}" for a, b in PAIRS for s in STATS for n in names]


def cos_columns():
    return [f"D_{a}-{b}__cos__{s}" for a, b in PAIRS for s in STATS]


def p_block(arrays, names):
    rows = {}
    for pid, X in arrays.items():
        mean, sd = mean_sd(X)                                       # (n_seq, n_feat) each
        rows[pid] = np.concatenate([np.concatenate([mean[i], sd[i]]) for i in range(len(MODS))])
    return pd.DataFrame.from_dict(rows, orient="index", columns=p_columns(names))


def d_block(arrays, names, shuffle_seed=None):
    idx = {m: i for i, m in enumerate(MODS)}
    rows = {}
    for pid, X in arrays.items():
        if shuffle_seed is not None:
            X = pair_permuted(X, shuffle_seed, pid)
        parts = []
        for a, b in PAIRS:
            mean, sd = mean_sd(X[:, idx[a], :] - X[:, idx[b], :])
            parts += [mean, sd]
        rows[pid] = np.concatenate(parts)
    return pd.DataFrame.from_dict(rows, orient="index", columns=d_columns(names))


def fit_sequence_scaling(arrays, train_ids):
    """Per sequence x feature mean and SD over every training patch (label-free)."""
    allp = np.concatenate([arrays[p] for p in train_ids if p in arrays], 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, sd = np.nanmean(allp, 0), np.nanstd(allp, 0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, np.nan)
    return mu, sd


def cos_block(arrays, ids, scaling, shuffle_seed=None):
    mu, sd = scaling
    idx = {m: i for i, m in enumerate(MODS)}
    rows = {}
    for pid in ids:
        X = arrays[pid]
        if shuffle_seed is not None:
            X = pair_permuted(X, shuffle_seed, pid)
        Z = np.nan_to_num((X - mu) / sd)                           # features with no training variation drop out
        parts = []
        for a, b in PAIRS:
            za, zb = Z[:, idx[a], :], Z[:, idx[b], :]
            den = np.linalg.norm(za, axis=1) * np.linalg.norm(zb, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.where(den > 0, (za * zb).sum(1) / den, np.nan)
            m, s = mean_sd(c[:, None])
            parts += [m, s]
        rows[pid] = np.concatenate(parts)
    return pd.DataFrame.from_dict(rows, orient="index", columns=cos_columns())


# ----------------------------------------------------------------------------------------------- model columns
GROUPS = {
    "P": lambda cols: [c for c in cols if c.startswith("P_")],
    "D": lambda cols: [c for c in cols if c.startswith("D_")],
    "D_T2FL": lambda cols: [c for c in cols if c.startswith("D_T2-FLAIR__")],
    "V": lambda cols: [c for c in cols if c.startswith(("V_", "MM__"))],
    "MM": lambda cols: [c for c in cols if c.startswith("MM__")],
    "M2": lambda cols: [c for c in cols if c.startswith("wt_")],
    "clin": lambda cols: [c for c in ["age", "sex_M"] if c in cols],
    "vol": lambda cols: [c for c in ["log_volume_ml"] if c in cols],
    "cohort": lambda cols: [c for c in ["cohort_UCSF"] if c in cols],
}
ALIASES = {"M0": "cohort", "M1_clin": "clin", "M1": "vol+clin"}


def expand(model):
    toks = []
    for t in model.split("+"):
        t = ALIASES.get(t, t)
        toks += expand(t) if "+" in t else [t]
    return toks


def model_columns(model, F, drop_prefixes=()):
    cols, out = list(F.columns), []
    for t in expand(model):
        out += GROUPS[t](cols)
    out = [c for c in dict.fromkeys(out) if not c.startswith(tuple(drop_prefixes))] if drop_prefixes else list(dict.fromkeys(out))
    return out or None


class CMDFBuilder:
    """Builds patient-level features for one train/test split; only the cosine scaling is fitted (training patients)."""

    def __init__(self, base, arrays, names, voxel=None):
        self.base, self.arrays, self.names = base, arrays, names
        self.P = p_block(arrays, names)
        self.V = voxel if voxel is not None else pd.DataFrame(index=self.P.index)
        self._D = {}

    def D(self, shuffle_seed=None):
        if shuffle_seed not in self._D:
            self._D[shuffle_seed] = d_block(self.arrays, self.names, shuffle_seed)
        return self._D[shuffle_seed]

    def build(self, train_ids, test_ids, shuffle_seed=None):
        train_ids, test_ids = list(train_ids), list(test_ids)
        ids = train_ids + test_ids
        missing = [p for p in ids if p not in self.arrays]
        assert not missing, f"patients without patch features: {missing[:5]}"
        scaling = fit_sequence_scaling(self.arrays, train_ids)
        F = (self.base.loc[ids].join(self.P).join(self.D(shuffle_seed)).join(cos_block(self.arrays, ids, scaling, shuffle_seed))
             .join(self.V))
        return F.loc[train_ids], F.loc[test_ids], {"scale_fit_patients": train_ids}

    def full_table(self, ids):
        """All blocks with the cosine scaling fitted on `ids` themselves (used only by the naive-protocol arm)."""
        F, _, _ = self.build(ids, [])
        return F
