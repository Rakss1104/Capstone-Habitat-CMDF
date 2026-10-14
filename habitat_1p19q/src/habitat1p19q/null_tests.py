"""Null tests (notebook sections 13.4–13.5).

Label permutation: the whole learning procedure is rerun with labels shuffled within cohort, on one dedicated 5-fold
split stratified by cohort only (so the folds cannot depend on the labels), shared by observed and permuted runs.

Spatial-arrangement null: habitat labels are shuffled across each patient's patch positions (composition kept) and M4
is refitted; tests whether arrangement carries label information beyond composition.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .validation import run_cv


def permute_within_cohort(base, ids, rng):
    y = base.loc[ids, "label"].to_numpy(int).copy()
    coh = base.loc[ids, "cohort"].to_numpy()
    for c in np.unique(coh):
        sel = np.flatnonzero(coh == c)
        y[sel] = rng.permutation(y[sel])
    return y


def run_chunks(fn, args_list, n_jobs):
    """Spread independent replicates over processes; results do not depend on n_jobs (per-replicate seeds)."""
    if n_jobs > 1 and len(args_list) > 1:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs)(delayed(fn)(*a) for a in args_list)
    else:
        parts = [fn(*a) for a in args_list]
    return [x for part in parts for x in part]


def _perm_chunk(ids, models, builder, perm_seed, seeds):
    out = []
    for sd in seeds:
        yp = permute_within_cohort(builder.base, ids, np.random.default_rng(int(sd)))
        r = run_cv(ids, models, builder, n_repeats=1, y_override=yp, thresholds=False, verbose=False,
                   strata_mode="cohort", seed=perm_seed)
        out.append({"y": yp, "auc": {m: roc_auc_score(yp, r.oof[m][0]) for m in models}, "folds": r.fold_test_idx})
    return out


def label_permutation_test(set_name, ids, models, builder, n_perm, nulls_out):
    cfg = builder.cfg
    perm_seed = cfg["seed"] + 77      # fixes the cohort-stratified folds shared by the observed and permuted runs
    # Observed statistic on the SAME label-independent folds that the permuted runs use
    ref = run_cv(ids, models, builder, n_repeats=1, thresholds=False, verbose=False, strata_mode="cohort", seed=perm_seed)
    obs = {m: roc_auc_score(ref.y, ref.oof[m][0]) for m in models if m in ref.oof}
    seeds = cfg["seed"] + 1000 + np.arange(n_perm)
    chunks = [c for c in np.array_split(seeds, max(1, min(cfg["n_jobs"], n_perm))) if len(c)]
    reps = run_chunks(_perm_chunk, [(ids, list(obs), builder, perm_seed, c) for c in chunks], cfg["n_jobs"])
    for rep in reps:
        same = all(np.array_equal(a, b) for a, b in zip(rep["folds"], ref.fold_test_idx))
        assert same, "permuted runs used different folds from the observed run"
    out = []
    for m in obs:
        nd = np.array([rep["auc"][m] for rep in reps])
        out.append({"set": set_name, "model": m, "observed AUROC (cohort-stratified folds)": obs[m], "null mean": nd.mean(),
                    "null 95th pct": np.percentile(nd, 95), "p (one-sided)": (1 + (nd >= obs[m]).sum()) / (1 + len(nd)),
                    "n_perm": len(nd)})
        nulls_out[f"labelperm|{set_name}|{m}"] = nd
    return out


def run_label_permutations(plan, analysis_sets, n_perm, nulls_out, rep):
    """plan: list of (set_name, models). Returns the permutation table."""
    rows = []
    t0 = time.time()
    for set_name, models in plan:
        ids, _, builder = analysis_sets[set_name]
        rep.say(f"  label permutation: {set_name} {models} ({n_perm} permutations)")
        rows += label_permutation_test(set_name, ids, models, builder, n_perm, nulls_out)
    df = pd.DataFrame(rows)
    rep.banner()
    rep.show(df, "Label-permutation test:", round_to=4)
    rep.save_table(df, "13_label_permutation")
    rep.record_check("Permuted and observed runs share identical label-independent folds", True, "asserted")
    rep.say(f"({time.time() - t0:.0f} s) Note: permuting within cohort keeps each cohort's prevalence, so M0 (cohort only) "
            "is expected to score about the same under permutation; a large p for M0 is the intended behaviour.")
    return df


def _spatial_chunk(ids, builder, y_main, auc_m3, seeds):
    out = []
    for i in seeds:
        r = run_cv(ids, ["M4"], builder, n_repeats=1, thresholds=False, shuffle_seed=int(i), verbose=False)
        out.append(roc_auc_score(y_main, r.oof["M4"][0]) - auc_m3)
    return out


def spatial_arrangement_null(ids, builder, main, n_null, rep):
    rep.banner()
    cfg = builder.cfg
    auc_m3 = roc_auc_score(main.y, main.oof["M3"][0])
    auc_m4 = roc_auc_score(main.y, main.oof["M4"][0])
    d_obs = auc_m4 - auc_m3
    t0 = time.time()
    seeds = np.arange(n_null)
    chunks = [c for c in np.array_split(seeds, max(1, min(cfg["n_jobs"], len(seeds)))) if len(c)]
    nd = np.array(run_chunks(_spatial_chunk, [(ids, builder, main.y, auc_m3, c) for c in chunks], cfg["n_jobs"]))
    p_sp = (1 + (nd >= d_obs).sum()) / (1 + len(nd))
    rep.say(f"Spatial-arrangement null ({len(nd)} replicates, {time.time() - t0:.0f} s): observed ΔAUROC (repeat 1) = "
            f"{d_obs:.3f}; null mean {nd.mean():.3f}, 95th percentile {np.percentile(nd, 95):.3f}; p = {p_sp:.3f}")
    return {"delta_obs_rep1": d_obs, "null_mean": nd.mean(), "null_95": np.percentile(nd, 95), "p": p_sp, "n": len(nd)}, nd
