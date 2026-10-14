"""Leakage-free repeated cross-validation, summaries and paired model comparisons (notebook sections 12–13).

Outer loop: repeated stratified K-fold on patients (stratified by cohort x label). Inside every training fold: fit the
habitat model, assign patches, compute patient features, tune C by inner CV, and choose the threshold from inner
out-of-fold predictions. The test fold is used only to score.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .modeling import make_estimator, model_columns, train_only_threshold
from .stats import auc_with_ci, auprc_with_ci, delta_auc_bootstrap, fmt_ci, nadeau_bengio


class CVResult:
    def __init__(self, ids, y, n_repeats, n_splits):
        self.ids, self.y = np.asarray(ids), np.asarray(y).astype(int)
        self.n_repeats, self.n_splits = n_repeats, n_splits
        self.oof, self.records, self.fold_test_idx, self.fold_info = {}, [], [], []
        self.models_rep0 = {}


def run_cv(analysis_ids, models, builder, estimator="logreg", n_repeats=None, seed=None, y_override=None,
           thresholds=True, shuffle_seed=None, keep_models_rep0=False, verbose=True, strata_mode="cohort_label", log=print):
    cfg, base = builder.cfg, builder.base
    n_repeats = n_repeats or cfg["cv"]["n_repeats"]
    n_splits = cfg["cv"]["n_splits"]
    seed = cfg["seed"] if seed is None else seed
    df = base.loc[analysis_ids]
    ids = df.index.to_numpy()
    y = df["label"].to_numpy(int) if y_override is None else np.asarray(y_override, int)
    # Folds are stratified by cohort x label (main analyses) or by cohort only (label-permutation test, so that the
    # folds cannot depend on the labels being permuted).
    if strata_mode == "cohort_label":
        strata = (df["cohort"] + "_" + df["label"].astype(str)).to_numpy()
    else:
        strata = df["cohort"].to_numpy()
    res = CVResult(ids, y, n_repeats, n_splits)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    t0 = time.time()
    for f, (tr, te) in enumerate(rskf.split(np.zeros(len(ids)), strata)):
        rep = f // n_splits
        tr_ids, te_ids = ids[tr], ids[te]
        assert not set(tr_ids) & set(te_ids), "train/test overlap"
        Ftr, Fte, info = builder.build(tr_ids, te_ids, shuffle_seed=shuffle_seed)
        assert set(info["habitat_fit_patients"]) <= set(tr_ids), "habitat model saw a test patient"
        res.fold_test_idx.append(te)
        res.fold_info.append({"fold": f, "repeat": rep, **{k: v for k, v in info.items() if k != "habitat_fit_patients"}})
        for m in models:
            cols = model_columns(m, Ftr, info["K"])
            if cols is None:
                continue
            res.oof.setdefault(m, np.full((n_repeats, len(ids)), np.nan))
            Xtr, Xte = Ftr[cols].to_numpy(float), Fte[cols].to_numpy(float)
            est = make_estimator(estimator, seed + f, cfg)
            est.fit(Xtr, y[tr])
            p = est.predict_proba(Xte)[:, 1]
            res.oof[m][rep, te] = p
            rec = {"fold": f, "repeat": rep, "model": m, "n_train": len(tr), "n_test": len(te), "n_test_pos": int(y[te].sum()),
                   "auc": roc_auc_score(y[te], p) if len(np.unique(y[te])) == 2 else np.nan, "K": info["K"],
                   "n_features": len(cols)}
            if estimator == "logreg":
                rec["C"] = est.best_params_["clf__C"]
            if thresholds:
                t = train_only_threshold(estimator, Xtr, y[tr], seed + f, cfg)
                pred = (p >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
                rec.update(threshold=t, tp=tp, fp=fp, tn=tn, fn=fn)
            res.records.append(rec)
            if keep_models_rep0 and rep == 0:
                res.models_rep0[(f, m)] = (est, cols, Fte[cols].copy(), y[te])
        if verbose and (f + 1) % n_splits == 0:
            log(f"  repeat {rep + 1}/{n_repeats} done ({time.time() - t0:.0f} s)")
    res.records = pd.DataFrame(res.records)
    return res


def summarise_cv(res, label=""):
    rows = []
    for m, P in res.oof.items():
        rep_auc = [roc_auc_score(res.y, P[r]) for r in range(res.n_repeats) if not np.isnan(P[r]).any()]
        avg = np.nanmean(P, 0)
        a = auc_with_ci(res.y, avg, seed=11)
        pr = auprc_with_ci(res.y, avg, seed=12)
        row = {"set": label, "model": m, "n": len(res.y), "n_codeleted": int(res.y.sum()),
               "AUROC (repeat-avg OOF, 95% CI)": fmt_ci(a["auc"], a["lo"], a["hi"]),
               "AUROC per repeat: mean [min–max]": f"{np.mean(rep_auc):.3f} [{np.min(rep_auc):.3f}–{np.max(rep_auc):.3f}]",
               "AUPRC (95% CI)": fmt_ci(pr["auprc"], pr["lo"], pr["hi"]), "prevalence": round(pr["prevalence"], 3),
               "Brier": brier_score_loss(res.y, avg), "_auc": a["auc"], "_lo": a["lo"], "_hi": a["hi"]}
        r = res.records[res.records["model"] == m]
        if "tp" in r:
            per_rep = r.groupby("repeat")[["tp", "fp", "tn", "fn"]].sum()
            sens = per_rep["tp"] / (per_rep["tp"] + per_rep["fn"])
            spec = per_rep["tn"] / (per_rep["tn"] + per_rep["fp"])
            ppv = per_rep["tp"] / (per_rep["tp"] + per_rep["fp"]).replace(0, np.nan)
            f1 = 2 * per_rep["tp"] / (2 * per_rep["tp"] + per_rep["fp"] + per_rep["fn"])
            row.update({"sensitivity": sens.mean(), "specificity": spec.mean(), "precision": ppv.mean(), "F1": f1.mean(),
                        "balanced acc.": ((sens + spec) / 2).mean(),
                        "confusion (rep 1) TP/FP/TN/FN": "/".join(str(int(per_rep.iloc[0][k])) for k in ["tp", "fp", "tn", "fn"])})
        rows.append(row)
    return pd.DataFrame(rows)


def within_cohort_discrimination(cv_results, base):
    """Per-cohort discrimination inside the pooled analyses (class counts alongside every number)."""
    rows = []
    for name, res in cv_results.items():
        coh = base.loc[res.ids, "cohort"].to_numpy()
        for m, P in res.oof.items():
            if m == "M0":
                continue   # cohort indicator is constant within a cohort: any within-cohort AUROC is CV noise
            avg = np.nanmean(P, 0)
            for c in np.unique(coh):
                sel = coh == c
                if len(np.unique(res.y[sel])) == 2 and len(np.unique(coh)) > 1:
                    a = auc_with_ci(res.y[sel], avg[sel], seed=31)
                    rows.append({"set": name, "model": m, "cohort": c, "n": int(sel.sum()), "n_codeleted": int(res.y[sel].sum()),
                                 "AUROC within cohort (95% CI)": fmt_ci(a["auc"], a["lo"], a["hi"])})
    return pd.DataFrame(rows)


def paired_comparison(res, test, ref, label=""):
    """Fold-level ΔAUROC with the Nadeau–Bengio corrected t-test, plus a bootstrap check on repeat-averaged OOF."""
    if test not in res.oof or ref not in res.oof:
        return None
    r = res.records.pivot_table(index="fold", columns="model", values="auc")
    d = (r[test] - r[ref]).to_numpy()
    n_test = res.records["n_test"].mean()
    n_train = res.records["n_train"].mean()
    nb = nadeau_bengio(d, n_train, n_test)
    avg_t, avg_r = np.nanmean(res.oof[test], 0), np.nanmean(res.oof[ref], 0)
    bs = delta_auc_bootstrap(res.y, avg_t, avg_r, seed=21)
    return {"set": label, "comparison": f"{test} − {ref}", "ΔAUROC (fold mean)": nb["mean"],
            "NB 95% CI": f"{nb['lo']:.3f} to {nb['hi']:.3f}", "NB p": nb["p"], "_nb_lo": nb["lo"], "_nb_hi": nb["hi"],
            "ΔAUROC (repeat-avg OOF)": bs["delta"], "bootstrap 95% CI": f"{bs['lo']:.3f} to {bs['hi']:.3f}",
            "folds with Δ>0": f"{int((d > 0).sum())}/{int(np.isfinite(d).sum())}"}
