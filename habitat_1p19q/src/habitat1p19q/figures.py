"""Results figures (notebook section 16). Every figure is drawn from result objects; nothing is typed in."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .plotting import plt


def roc_figure(cv_results, pal, smoke):
    sets = list(cv_results)
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 4.4), squeeze=False)
    for ax, name in zip(axes[0], sets):
        res = cv_results[name]
        ax.plot([0, 1], [0, 1], ls="--", color="#8a8984", lw=1)
        for m, Pm in res.oof.items():
            avg = np.nanmean(Pm, 0)
            fpr, tpr, _ = roc_curve(res.y, avg)
            ax.plot(fpr, tpr, **pal.model_style(m), label=f"{m}: {roc_auc_score(res.y, avg):.3f}")
        ax.set_title(f"{name} (n={len(res.y)}, {int(res.y.sum())} codel.)", fontsize=10)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle(("SYNTHETIC SMOKE TEST — " if smoke else "") + "ROC, repeat-averaged out-of-fold predictions", fontsize=11)
    fig.tight_layout()
    return fig


def forest_figure(summary, site_assoc, show_cohort_line, pal, smoke):
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(summary) + 1.2))
    yl = []
    for i, (_, r) in enumerate(summary.iloc[::-1].iterrows()):
        ax.errorbar(r["_auc"], i, xerr=[[r["_auc"] - r["_lo"]], [r["_hi"] - r["_auc"]]], fmt="o", color="#2b2b2a", ms=5, capsize=3)
        yl.append(f"{r['set']} · {r['model']} (n={r['n']}, {r['n_codeleted']} codel.)")
    ax.axvline(0.5, color="#8a8984", ls="--", lw=1)
    if show_cohort_line:
        a0 = site_assoc["cohort_only_auc"]["auc"]
        ax.axvline(a0, color=pal.cohort["UCSF"], ls=":", lw=1.5)
        ax.annotate(f"cohort only, pooled\n(analytic) {a0:.3f}", xy=(a0, len(yl) - 0.55), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color="#52514e", va="top")
    ax.set_yticks(range(len(yl)))
    ax.set_yticklabels(yl, fontsize=8)
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.set_ylim(-0.6, len(yl) - 0.4)
    ax.set_title(("SYNTHETIC SMOKE TEST — " if smoke else "") + "Discrimination by analysis set and model")
    fig.tight_layout()
    return fig


def null_figure(nulls, label_perm, spatial_null, pal):
    keys = list(nulls)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.2), squeeze=False)
    for ax, k in zip(axes[0], keys):
        ax.hist(nulls[k], bins=20, color="#9ec5f4", edgecolor="#2a78d6")
        if k == "spatial":
            obs = spatial_null["delta_obs_rep1"]
            ax.set_xlabel("ΔAUROC M4 − M3 under null")
        else:
            _, sname, m = k.split("|")
            row = label_perm[(label_perm["set"] == sname) & (label_perm["model"] == m)]
            obs = float(row["observed AUROC (cohort-stratified folds)"].iloc[0])
            ax.set_xlabel("AUROC under label permutation")
        ax.axvline(obs, color=pal.codeleted, lw=2, label="observed")
        ax.set_title(k.replace("labelperm|", "perm: ").replace("|", " · "), fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
