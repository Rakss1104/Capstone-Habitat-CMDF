"""Leakage-free repeated CV, null tests and leave-one-cohort-out for the CMDF models.

This mirrors habitat1p19q.validation.run_cv (same folds, estimators, inner tuning and training-only thresholds) but takes
the CMDF column specification and runs folds in parallel. Results use habitat1p19q's CVResult, so its summaries and
paired comparisons apply unchanged.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from habitat1p19q.modeling import make_estimator, train_only_threshold
from habitat1p19q.null_tests import permute_within_cohort
from habitat1p19q.stats import auc_with_ci, auprc_with_ci, fmt_ci
from habitat1p19q.validation import CVResult

from .features import model_columns


def _fold(f, rep, tr, te, ids, y, models, builder, estimator, seed, cfg, thresholds, shuffle_seed, drop):
    tr_ids, te_ids = ids[tr], ids[te]
    assert not set(tr_ids) & set(te_ids), "train/test overlap"
    Ftr, Fte, info = builder.build(tr_ids, te_ids, shuffle_seed=shuffle_seed)
    assert set(info["scale_fit_patients"]) <= set(tr_ids), "cosine scaling saw a test patient"
    recs, preds = [], {}
    for m in models:
        cols = model_columns(m, Ftr, drop.get(m, ()))
        if cols is None:
            continue
        Xtr, Xte = Ftr[cols].to_numpy(float), Fte[cols].to_numpy(float)
        est = make_estimator(estimator, seed + f, cfg).fit(Xtr, y[tr])
        p = est.predict_proba(Xte)[:, 1]
        preds[m] = p
        rec = {"fold": f, "repeat": rep, "model": m, "n_train": len(tr), "n_test": len(te), "n_test_pos": int(y[te].sum()),
               "auc": roc_auc_score(y[te], p) if len(np.unique(y[te])) == 2 else np.nan, "n_features": len(cols)}
        if estimator == "logreg":
            rec["C"] = est.best_params_["clf__C"]
        if thresholds:
            t = train_only_threshold(estimator, Xtr, y[tr], seed + f, cfg)
            tn, fp, fn, tp = confusion_matrix(y[te], (p >= t).astype(int), labels=[0, 1]).ravel()
            rec.update(threshold=t, tp=tp, fp=fp, tn=tn, fn=fn)
        recs.append(rec)
    return f, rep, te, recs, preds


def run_cv(ids, models, builder, cfg, estimator="logreg", n_repeats=None, seed=None, y_override=None, thresholds=True,
           shuffle_seed=None, strata_mode="cohort_label", n_jobs=1, drop=None, log=None):
    base = builder.base
    n_repeats = n_repeats or cfg["cv"]["n_repeats"]
    n_splits = cfg["cv"]["n_splits"]
    seed = cfg["seed"] if seed is None else seed
    df = base.loc[list(ids)]
    ids = df.index.to_numpy()
    y = df["label"].to_numpy(int) if y_override is None else np.asarray(y_override, int)
    strata = (df["cohort"] + "_" + df["label"].astype(str)).to_numpy() if strata_mode == "cohort_label" else df["cohort"].to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(rskf.split(np.zeros(len(ids)), strata))
    args = [(f, f // n_splits, tr, te, ids, y, models, builder, estimator, seed, cfg, thresholds, shuffle_seed, drop or {})
            for f, (tr, te) in enumerate(splits)]
    t0 = time.time()
    if n_jobs > 1:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs)(delayed(_fold)(*a) for a in args)
    else:
        outs = [_fold(*a) for a in args]
    res = CVResult(ids, y, n_repeats, n_splits)
    for f, rep, te, recs, preds in sorted(outs, key=lambda o: o[0]):
        res.fold_test_idx.append(te)
        res.fold_info.append({"fold": f, "repeat": rep})
        for m, p in preds.items():
            res.oof.setdefault(m, np.full((n_repeats, len(ids)), np.nan))[rep, te] = p
        res.records += recs
    res.records = pd.DataFrame(res.records)
    if log:
        log(f"  {len(splits)} folds x {len(models)} models in {time.time() - t0:.0f} s")
    return res


def rep_auc(res, model, r=0):
    return roc_auc_score(res.y, res.oof[model][r])


# ------------------------------------------------------------------------------------------------- null tests
def _pairing_chunk(ids, model, ref_auc, builder, cfg, seeds):
    out = []
    for s in seeds:
        r = run_cv(ids, [model], builder, cfg, n_repeats=1, thresholds=False, shuffle_seed=int(s))
        out.append(rep_auc(r, model) - ref_auc)
    return out


def pairing_null(ids, main, test, ref, builder, cfg, n_null, n_jobs, rep):
    """ΔAUROC(test − ref) on the first-repeat folds with co-location shuffled in D (ref keeps no D features)."""
    from joblib import Parallel, delayed
    ref_auc, d_obs = rep_auc(main, ref), rep_auc(main, test) - rep_auc(main, ref)
    t0 = time.time()
    seeds = np.arange(n_null) + 1
    chunks = [c for c in np.array_split(seeds, max(1, min(n_jobs, n_null))) if len(c)]
    nd = np.array([x for part in Parallel(n_jobs=n_jobs)(delayed(_pairing_chunk)(ids, test, ref_auc, builder, cfg, c)
                                                         for c in chunks) for x in part])
    p = (1 + (nd >= d_obs).sum()) / (1 + len(nd))
    rep.say(f"Pairing null ({len(nd)} replicates, {time.time() - t0:.0f} s): observed Δ{test}−{ref} (repeat 1) = {d_obs:.3f}; "
            f"null mean {nd.mean():.3f}, 95th percentile {np.percentile(nd, 95):.3f}; p = {p:.3f}")
    return {"comparison": f"{test} − {ref}", "delta_obs_rep1": d_obs, "null_mean": nd.mean(),
            "null_95": np.percentile(nd, 95), "p": p, "n": len(nd)}, nd


def _perm_chunk(ids, models, builder, cfg, perm_seed, seeds):
    out = []
    for sd in seeds:
        yp = permute_within_cohort(builder.base, ids, np.random.default_rng(int(sd)))
        r = run_cv(ids, models, builder, cfg, n_repeats=1, y_override=yp, thresholds=False, strata_mode="cohort", seed=perm_seed)
        out.append({"auc": {m: roc_auc_score(yp, r.oof[m][0]) for m in models}, "folds": r.fold_test_idx})
    return out


def label_permutation(set_name, ids, models, builder, cfg, n_perm, n_jobs, nulls_out):
    from joblib import Parallel, delayed
    perm_seed = cfg["seed"] + 77
    ref = run_cv(ids, models, builder, cfg, n_repeats=1, thresholds=False, strata_mode="cohort", seed=perm_seed, n_jobs=n_jobs)
    obs = {m: roc_auc_score(ref.y, ref.oof[m][0]) for m in models}
    seeds = cfg["seed"] + 1000 + np.arange(n_perm)
    chunks = [c for c in np.array_split(seeds, max(1, min(n_jobs, n_perm))) if len(c)]
    reps = [x for part in Parallel(n_jobs=n_jobs)(delayed(_perm_chunk)(ids, models, builder, cfg, perm_seed, c) for c in chunks)
            for x in part]
    assert all(all(np.array_equal(a, b) for a, b in zip(r["folds"], ref.fold_test_idx)) for r in reps), \
        "permuted runs used different folds from the observed run"
    rows = []
    for m in models:
        nd = np.array([r["auc"][m] for r in reps])
        nulls_out[f"labelperm|{set_name}|{m}"] = nd
        rows.append({"set": set_name, "model": m, "observed AUROC (cohort-stratified folds)": obs[m], "null mean": nd.mean(),
                     "null 95th pct": np.percentile(nd, 95), "p (one-sided)": (1 + (nd >= obs[m]).sum()) / (1 + len(nd)),
                     "n_perm": len(nd)})
    return rows


# --------------------------------------------------------------------------------------- cross-cohort transport
def run_loco(train_c, test_c, pool_ids, models, builder, cfg, min_events):
    base = builder.base
    tr = [p for p in pool_ids if base.loc[p, "cohort"] == train_c]
    te = [p for p in pool_ids if base.loc[p, "cohort"] == test_c]
    ytr, yte = base.loc[tr, "label"].to_numpy(int), base.loc[te, "label"].to_numpy(int)
    if ytr.sum() < min_events:
        return [{"train": train_c, "test": test_c, "model": "all",
                 "status": f"NOT RUN: {int(ytr.sum())} codeleted training patients < {min_events}"}]
    Ftr, Fte, info = builder.build(tr, te)
    assert set(info["scale_fit_patients"]) <= set(tr)
    out = []
    for m in models:
        cols = model_columns(m, Ftr)
        Xtr, Xte = Ftr[cols].to_numpy(float), Fte[cols].to_numpy(float)
        est = make_estimator("logreg", cfg["seed"], cfg).fit(Xtr, ytr)
        p = est.predict_proba(Xte)[:, 1]
        t = train_only_threshold("logreg", Xtr, ytr, cfg["seed"], cfg)
        tn, fp, fn, tp = confusion_matrix(yte, (p >= t).astype(int), labels=[0, 1]).ravel()
        a, pr = auc_with_ci(yte, p, seed=41), auprc_with_ci(yte, p, seed=42)
        out.append({"train": f"{train_c} (n={len(tr)}, {int(ytr.sum())} codel.)", "test": f"{test_c} (n={len(te)}, {int(yte.sum())} codel.)",
                    "model": m, "status": "run", "AUROC (95% CI)": fmt_ci(a["auc"], a["lo"], a["hi"]),
                    "AUPRC (95% CI)": fmt_ci(pr["auprc"], pr["lo"], pr["hi"]), "test prevalence": round(pr["prevalence"], 3),
                    "sensitivity": tp / (tp + fn) if tp + fn else np.nan, "specificity": tn / (tn + fp) if tn + fp else np.nan,
                    "TP/FP/TN/FN": f"{tp}/{fp}/{tn}/{fn}", "Brier": brier_score_loss(yte, p), "_auc": a["auc"]})
    return out
