"""Results figures. Every value is drawn from result objects; nothing is typed in.

Colours: the habitat study's validated categorical slots, assigned to models in a fixed order (colour follows the
model, never its rank); classes use the habitat class colours; null distributions are neutral grey with the observed
value in the accent colour. Each ROC panel shows at most five models, so the adjacent-pair palette check applies.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from habitat1p19q.plotting import apply_style, plt

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOR = {"P": SLOTS[0], "P+D": SLOTS[1], "D": SLOTS[2], "V": SLOTS[3], "M2": SLOTS[4], "D_T2FL": SLOTS[5],
               "MM": SLOTS[6], "M2+V": SLOTS[7]}
CLASS_COLOR = {0: "#2a78d6", 1: "#eb6834"}
INK, MUTED, GRID = "#2b2b2a", "#8a8984", "#d9d8d4"
ROC_MODELS = ["P", "P+D", "D", "V", "M2"]


def _color(m):
    return MODEL_COLOR.get(m.replace("+cohort", ""), MUTED)


def roc_figure(cv, sets=("UTSW", "pooled")):
    apply_style()
    sets = [s for s in sets if s in cv]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.8 * len(sets), 4.6), squeeze=False)
    for ax, name in zip(axes[0], sets):
        res = cv[name]
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
        for m in ROC_MODELS:
            mm = m if m in res.oof else f"{m}+cohort"
            if mm not in res.oof:
                continue
            avg = np.nanmean(res.oof[mm], 0)
            fpr, tpr, _ = roc_curve(res.y, avg)
            ax.plot(fpr, tpr, color=_color(m), ls="--" if mm.endswith("+cohort") else "-",
                    label=f"{mm}  {roc_auc_score(res.y, avg):.3f}")
        ax.set_title(f"{name}: n = {len(res.y)}, {int(res.y.sum())} codeleted", fontsize=10, color=INK)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="lower right", title="model  AUROC", title_fontsize=8)
    fig.suptitle("ROC curves, repeat-averaged out-of-fold predictions", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def forest_figure(summary, cohort_only_auc=None):
    apply_style()
    s = summary.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(s) + 1.3))
    y = np.arange(len(s))[::-1]
    for yi, (_, r) in zip(y, s.iterrows()):
        c = _color(r["model"]) if r["model"].replace("+cohort", "") in MODEL_COLOR else MUTED
        ax.plot([r["_lo"], r["_hi"]], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(r["_auc"], yi, "o", ms=7, color=c, mec="white", mew=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['set']} · {r['model']}" for _, r in s.iterrows()], fontsize=8)
    ax.axvline(0.5, color=MUTED, ls="--", lw=1)
    if cohort_only_auc is not None:
        ax.axvline(cohort_only_auc, color=SLOTS[6], ls=":", lw=1.2)
        ax.text(cohort_only_auc, y.max() + 0.7, " cohort only", color=INK, fontsize=7.5, va="bottom")
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.set_xlim(0.25, 1.0)
    ax.grid(axis="y", visible=False)
    ax.set_title("Discrimination by analysis set and model", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def null_figure(panels):
    """panels: list of (title, null_values, observed, observed_label)."""
    apply_style()
    n = len(panels)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, (title, nd, obs, lab) in zip(axes.flat, panels):
        ax.hist(nd, bins=20, color="#b9b8b3", edgecolor="white", linewidth=1)
        ax.axvline(obs, color=SLOTS[1], lw=2)
        p = (1 + (np.asarray(nd) >= obs).sum()) / (1 + len(nd))
        ax.set_title(title, fontsize=9, color=INK)
        ax.text(0.98, 0.95, f"{lab} = {obs:.3f}\np = {p:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
        ax.set_ylabel("replicates")
    fig.tight_layout()
    return fig


def mismatch_figure(base, V, ids_by_set, feature="MM__mismatch_frac", label="T2–FLAIR mismatch fraction"):
    apply_style()
    fig, axes = plt.subplots(1, len(ids_by_set), figsize=(3.4 * len(ids_by_set), 3.8), squeeze=False, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (name, ids) in zip(axes[0], ids_by_set.items()):
        d = base.loc[ids, ["label"]].join(V[[feature]]).dropna()
        groups = [d.loc[d["label"] == k, feature].to_numpy() for k in (0, 1)]
        bp = ax.boxplot(groups, positions=[0, 1], widths=0.5, showfliers=False, patch_artist=True,
                        medianprops={"color": INK, "lw": 1.5}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        for patch, k in zip(bp["boxes"], (0, 1)):
            patch.set(facecolor="white", edgecolor=CLASS_COLOR[k], lw=1.5)
        for k, g in enumerate(groups):
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(g)), g, s=14, color=CLASS_COLOR[k], alpha=0.8, edgecolor="white", lw=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"non-codeleted\n(n={len(groups[0])})", f"codeleted\n(n={len(groups[1])})"], fontsize=8)
        ax.set_title(name, fontsize=10, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel(label)
    fig.tight_layout()
    return fig


def ablation_figure(abl):
    apply_style()
    a = abl.sort_values("ΔAUROC vs full P+D")
    fig, ax = plt.subplots(figsize=(6, 0.42 * len(a) + 1.2))
    y = np.arange(len(a))
    ax.barh(y, a["ΔAUROC vs full P+D"], color=SLOTS[0], height=0.6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(a["variant"], fontsize=8)
    for yi, v in zip(y, a["ΔAUROC vs full P+D"]):
        ax.text(v, yi, f" {v:+.3f}" if v >= 0 else f"{v:+.3f} ", va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK)
    ax.set_xlabel("ΔAUROC when the pair is removed from D (repeat-averaged OOF)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Leave-one-pair-out ablation (UTSW)", fontsize=11, color=INK)
    fig.tight_layout()
    return fig
