"""Calibration, grouped permutation importance and robustness of the primary Δ (notebook section 15)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .modeling import FoldFeatureBuilder
from .plotting import plt
from .spatial import composition_cols, spatial_cols
from .stats import calibration_stats, wilson_ci
from .validation import paired_comparison, run_cv


def calibration(res, set_name, pal):
    """Reliability curve (quintile bins, Wilson intervals), slope and calibration-in-the-large."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", color="#8a8984", lw=1, label="ideal")
    rows = []
    for m, Pm in res.oof.items():
        avg = np.nanmean(Pm, 0)
        cs = calibration_stats(res.y, avg)
        rows.append({"set": set_name, "model": m, **cs})
        q = pd.qcut(avg, 5, labels=False, duplicates="drop")
        xs, ys, lo, hi = [], [], [], []
        for qb in np.unique(q):
            sel = q == qb
            k, n = int(res.y[sel].sum()), int(sel.sum())
            xs.append(avg[sel].mean())
            ys.append(k / n)
            w = wilson_ci(k, n)
            lo.append(max(0.0, k / n - w[0]))
            hi.append(max(0.0, w[1] - k / n))   # guard against -1e-17 rounding
        st = pal.model_style(m)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o" + st["ls"], ms=6, capsize=3, color=st["color"],
                    label=f"{m} (slope {cs['slope']:.2f})")
    ax.set_xlabel("Mean predicted probability (quintile)")
    ax.set_ylabel("Observed codeletion fraction (95% Wilson CI)")
    ax.set_title(f"Calibration, {set_name}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return pd.DataFrame(rows), fig


def grouped_importance_m4(res, seed):
    """AUROC lost when the composition or spatial group is shuffled across first-repeat test patients."""
    rng = np.random.default_rng(seed + 9)
    rows = []
    for (f, m), (est, cols, Xte, yte) in res.models_rep0.items():
        if m != "M4":
            continue
        K = int(sum(c.startswith("comp_h") for c in cols))
        groups = {"composition": composition_cols(K), "spatial": spatial_cols(K)}
        base_auc = roc_auc_score(yte, est.predict_proba(Xte.to_numpy(float))[:, 1])
        for gname, gcols in groups.items():
            drops = []
            for _ in range(50):
                Xp = Xte.copy()
                perm = rng.permutation(len(Xp))
                Xp[gcols] = Xp[gcols].to_numpy()[perm]
                drops.append(base_auc - roc_auc_score(yte, est.predict_proba(Xp.to_numpy(float))[:, 1]))
            rows.append({"fold": f, "group": gname, "AUROC drop": np.mean(drops)})
    return pd.DataFrame(rows).groupby("group")["AUROC drop"].agg(["mean", "std", "count"])


def _delta_on_rep1(ids, builder, label, rows):
    base, cfg = builder.base, builder.cfg
    if len(ids):
        strata = (base.loc[ids, "cohort"] + "_" + base.loc[ids, "label"].astype(str)).value_counts()
    if len(ids) == 0 or strata.min() < cfg["cv"]["n_splits"]:
        rows.append({"variant": label, "n": len(ids), "n_codeleted": int(base.loc[ids, "label"].sum()) if len(ids) else 0,
                     "ΔAUROC (fold mean)": np.nan, "NB 95% CI": "NOT RUN: a cohort × label stratum has fewer patients than folds"})
        return
    r = run_cv(ids, ["M3", "M4"], builder, n_repeats=1, thresholds=False, verbose=False)
    if "M4" not in r.oof:
        return
    d = paired_comparison(r, "M4", "M3", label)
    rows.append({"variant": label, "n": len(ids), "n_codeleted": int(base.loc[ids, "label"].sum()),
                 "ΔAUROC (fold mean)": d["ΔAUROC (fold mean)"], "NB 95% CI": d["NB 95% CI"],
                 "K per fold": ",".join(str(int(x)) for x in pd.DataFrame(r.fold_info)["K"])})


def robustness_primary_delta(primary_set, analysis_sets, builder_img, patches, label_sets, evaluable_ids, cfg, rep):
    """Primary Δ (first-repeat folds) under fixed K, label-definition variants and per-cohort harmonisation."""
    rows = []
    base = builder_img.base
    p_ids = analysis_sets[primary_set][0]
    for k in cfg["k_sensitivity"]:
        rep.say(f"  robustness: fixed K = {k}")
        _delta_on_rep1(p_ids, FoldFeatureBuilder(base, cfg, patches, fixed_k=k), f"{primary_set}: fixed K = {k}", rows)
    for name in ["exclude_histology_discordant", "exclude_grade4"]:
        keep = set(label_sets[name])
        _delta_on_rep1([p for p in p_ids if p in keep], builder_img, f"{primary_set}: {name}", rows)
    if "pooled" in analysis_sets:
        rep.say("  robustness: pooled per-cohort harmonisation")
        _delta_on_rep1(analysis_sets["pooled"][0], FoldFeatureBuilder(base, cfg, patches, harmonisation="per_cohort_zscore"),
                       "pooled: per-cohort z-score harmonisation (training-only)", rows)
        ev = set(evaluable_ids)
        extra = [p for p in base.index[base["cohort"].isin(["UTSW", "UCSF"])] if p in ev]
        _delta_on_rep1(extra, builder_img, "pooled: UCSF relative/unknown resolved by WHO 2021 diagnosis", rows)
    return pd.DataFrame(rows)
