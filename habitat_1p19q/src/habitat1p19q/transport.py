"""Cross-cohort generalisation: leave-one-cohort-out (notebook section 14).

The habitat model, tuning and threshold are fitted on one cohort and the other is scored once. A direction runs only if
its training cohort has at least `min_train_events_loco` codeleted patients.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, confusion_matrix

from .modeling import make_estimator, model_columns, train_only_threshold
from .stats import auc_with_ci, auprc_with_ci, fmt_ci


def run_loco(train_c, test_c, pool_ids, models, builder, estimator="logreg"):
    cfg, base = builder.cfg, builder.base
    tr = [p for p in pool_ids if base.loc[p, "cohort"] == train_c]
    te = [p for p in pool_ids if base.loc[p, "cohort"] == test_c]
    ytr, yte = base.loc[tr, "label"].to_numpy(int), base.loc[te, "label"].to_numpy(int)
    if ytr.sum() < cfg["min_train_events_loco"]:
        return [{"train": train_c, "test": test_c, "model": "all",
                 "status": f"NOT RUN: {int(ytr.sum())} codeleted training patients < {cfg['min_train_events_loco']}"}]
    Ftr, Fte, info = builder.build(np.array(tr), np.array(te))
    assert set(info["habitat_fit_patients"]) <= set(tr)
    out = []
    for m in models:
        cols = model_columns(m, Ftr, info["K"])
        if cols is None:
            continue
        Xtr, Xte = Ftr[cols].to_numpy(float), Fte[cols].to_numpy(float)
        est = make_estimator(estimator, cfg["seed"], cfg).fit(Xtr, ytr)
        p = est.predict_proba(Xte)[:, 1]
        t = train_only_threshold(estimator, Xtr, ytr, cfg["seed"], cfg)
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        a = auc_with_ci(yte, p, seed=41)
        pr = auprc_with_ci(yte, p, seed=42)
        out.append({"train": f"{train_c} (n={len(tr)}, {int(ytr.sum())} codel.)", "test": f"{test_c} (n={len(te)}, {int(yte.sum())} codel.)",
                    "model": m, "status": "run", "K": info["K"], "AUROC (95% CI)": fmt_ci(a["auc"], a["lo"], a["hi"]),
                    "AUPRC (95% CI)": fmt_ci(pr["auprc"], pr["lo"], pr["hi"]), "test prevalence": round(pr["prevalence"], 3),
                    "sensitivity": tp / (tp + fn) if tp + fn else np.nan, "specificity": tn / (tn + fp) if tn + fp else np.nan,
                    "TP/FP/TN/FN": f"{tp}/{fp}/{tn}/{fn}", "Brier": brier_score_loss(yte, p),
                    "transductive harmonisation": ", ".join(info.get("transductive_cohorts", [])) or "none"})
    return out
