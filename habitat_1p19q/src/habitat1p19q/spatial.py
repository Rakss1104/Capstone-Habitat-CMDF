"""Patient-level habitat features (notebook section 10).

Composition (M3): `comp_h{k}`, `entropy_norm`, `n_habitats_present` — functions of habitat counts only.
Spatial arrangement (added in M4), each constructed so its expectation under random arrangement does not depend on
composition: `mixing_excess`, `radial_eta2_excess` (observed minus within-patient permutation-null mean),
`depth_shift_h{k}` and the `spatial_defined` indicator. The z-scores are descriptive only: they grow with patch count.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np
import pandas as pd


def grid_pairs(starts, stride):
    lookup = {tuple(s): i for i, s in enumerate(starts)}
    ii, jj = [], []
    for i, s in enumerate(starts):
        for ax in range(3):
            nb = list(s)
            nb[ax] += stride
            j = lookup.get(tuple(nb))
            if j is not None:
                ii.append(i)
                jj.append(j)
    return np.array(ii, int), np.array(jj, int)


def eta2_rows(labels_2d, depth, K):
    """labels_2d: (R, n) habitat labels per row; returns eta^2 of depth explained by habitat, per row."""
    R, n = labels_2d.shape
    tot = ((depth - depth.mean()) ** 2).sum()
    if tot <= 0:
        return np.full(R, np.nan)
    offs = labels_2d + K * np.arange(R)[:, None]
    sums = np.bincount(offs.ravel(), weights=np.tile(depth, R), minlength=R * K).reshape(R, K)
    cnts = np.bincount(offs.ravel(), minlength=R * K).reshape(R, K)
    means = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)
    between = (cnts * (means - depth.mean()) ** 2).sum(1)
    return between / tot


def patient_habitat_features(hab, starts, depth_norm, K, rng, n_null, stride, min_patches, shuffle=False):
    n = len(hab)
    if shuffle:
        hab = rng.permutation(hab)          # spatial-arrangement null: composition preserved, positions scrambled
    comp = np.bincount(hab, minlength=K) / n
    nz = comp[comp > 0]
    f = {f"comp_h{k}": comp[k] for k in range(K)}
    f["entropy_norm"] = float(-(nz * np.log(nz)).sum() / math.log(K)) if K > 1 else 0.0
    f["n_habitats_present"] = int((comp > 0).sum())
    for k in range(K):
        sel = hab == k
        f[f"depth_shift_h{k}"] = float(depth_norm[sel].mean() - depth_norm.mean()) if sel.any() else 0.0
    i, j = grid_pairs(starts, stride)
    defined = n >= min_patches and len(i) > 0
    f["spatial_defined"] = int(defined)
    f["mixing_obs"], f["mixing_excess"], f["mixing_z"] = np.nan, 0.0, 0.0
    f["radial_eta2_excess"], f["radial_eta2_z"] = 0.0, 0.0
    if defined:
        obs = float((hab[i] != hab[j]).mean())
        perms = rng.permuted(np.tile(hab, (n_null, 1)), axis=1)
        null = (perms[:, i] != perms[:, j]).mean(1)
        f["mixing_obs"] = obs
        f["mixing_excess"] = obs - float(null.mean())
        f["mixing_z"] = float((obs - null.mean()) / null.std()) if null.std() > 0 else 0.0
        e_obs = eta2_rows(hab[None, :], depth_norm, K)[0]
        e_null = eta2_rows(perms, depth_norm, K)
        if np.isfinite(e_obs) and np.isfinite(e_null).any():
            f["radial_eta2_excess"] = float(e_obs - np.nanmean(e_null))
            if np.nanstd(e_null) > 0:
                f["radial_eta2_z"] = float((e_obs - np.nanmean(e_null)) / np.nanstd(e_null))
    else:
        for k in range(K):
            f[f"depth_shift_h{k}"] = 0.0
    return f


def build_patient_features(patch_df, K, seed, cfg, shuffle_spatial=False):
    """patch_df must carry 'habitat'. Deterministic per patient given seed."""
    out = {}
    for pid, g in patch_df.groupby("patient_id", sort=True):
        h = int(hashlib.md5(f"{seed}|{pid}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        out[pid] = patient_habitat_features(g["habitat"].to_numpy(int), g[["z0", "y0", "x0"]].to_numpy(int),
                                            g["depth_norm"].to_numpy(float), K, rng, cfg["spatial"]["n_inner_null"],
                                            cfg["patch_stride"], cfg["spatial"]["min_patches"], shuffle=shuffle_spatial)
    return pd.DataFrame.from_dict(out, orient="index")


def composition_cols(K):
    return [f"comp_h{k}" for k in range(K)] + ["entropy_norm", "n_habitats_present"]


def spatial_cols(K):
    # Excess-over-null metrics, not z-scores: a z-score grows with the number of patches for the same effect.
    return ["mixing_excess", "radial_eta2_excess", "spatial_defined"] + [f"depth_shift_h{k}" for k in range(K)]


# ---------------------------------------------------------------- 10.1 unit test on toy patch grids
def toy_grid(kind, stride, n=6, seed=0):
    g = np.array(list(itertools.product(range(n), repeat=3)))
    starts = g * stride
    centre = (n - 1) / 2
    dist = np.abs(g - centre).max(1)                      # 0 at centre, 2.5 at the outer shell
    depth = 1 - dist / dist.max()                         # 1 = deepest, 0 = margin
    if kind == "segregated":
        hab = (g[:, 0] >= n // 2).astype(int)
    elif kind == "checkerboard":
        hab = (g.sum(1) % 2).astype(int)
    elif kind == "random":
        hab = np.random.default_rng(seed).permutation((np.arange(len(g)) % 2).astype(int))
    elif kind == "radial":
        hab = (depth <= np.median(depth)).astype(int)     # habitat 1 at the margin, 0 in the core
    else:
        raise ValueError(kind)
    return hab, starts, depth


def toy_grid_self_test(stride=16, min_patches=4):
    """Return (table, passed). Arrangements with identical composition must be separated by the spatial metrics."""
    toy = {}
    for kind in ["segregated", "checkerboard", "random", "radial"]:
        hab, starts, depth = toy_grid(kind, stride)
        toy[kind] = patient_habitat_features(hab, starts, depth, 3, np.random.default_rng(1), 500, stride, min_patches)
    table = pd.DataFrame(toy).T[["comp_h0", "comp_h1", "comp_h2", "entropy_norm", "mixing_obs", "mixing_excess", "mixing_z",
                                 "radial_eta2_excess", "radial_eta2_z", "depth_shift_h0", "depth_shift_h1", "depth_shift_h2"]]
    ok = (abs(toy["segregated"]["comp_h0"] - toy["checkerboard"]["comp_h0"]) < 1e-9
          and toy["segregated"]["mixing_z"] < -3 and toy["checkerboard"]["mixing_z"] > 3
          and abs(toy["random"]["mixing_z"]) < 3
          and toy["segregated"]["mixing_excess"] < 0 < toy["checkerboard"]["mixing_excess"]
          and toy["radial"]["radial_eta2_z"] > 3 and toy["radial"]["radial_eta2_excess"] > 0
          and toy["radial"]["depth_shift_h0"] > 0 > toy["radial"]["depth_shift_h1"]
          and toy["radial"]["depth_shift_h2"] == 0 and toy["radial"]["comp_h2"] == 0)
    return table, bool(ok)
