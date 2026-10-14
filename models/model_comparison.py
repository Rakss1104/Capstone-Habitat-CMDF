"""Habitat vs CMDF: nine classifiers on each approach's feature sets, under one leakage-free protocol.

Feature sets (all built inside each training fold):
  habitat  M3  habitat composition                      (habitat1p19q; habitat model refitted per training fold)
           M4  composition + composition-adjusted spatial metrics
  cmdf     V   voxel-level CMDF incl. quantitative T2-FLAIR mismatch (fixed per-patient transform)
           P+D single-sequence + patch-level CMDF (cosine block scaled on training patients)
Pooled analyses add the cohort indicator to every feature set.

Validation: repeated stratified K-fold on patients (strata cohort x label; the habitat study's seed and patient order,
so the first repeats use the same folds as both studies). Inside each training fold: features, inner-CV hyper-parameter
grid, and the decision threshold (maximising balanced accuracy on inner out-of-fold predictions). Test folds only score.

Outputs: models/habitat/, models/cmdf/ (per-approach tables, fold records, figures) and models/ (comparison).
Usage (WSL):  ~/venvs/habitat2/bin/python models/model_comparison.py [--repeats 5] [--n-jobs 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, matthews_corrcoef,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CMDF"))

from habitat1p19q import stats as st                                      # noqa: E402
from habitat1p19q.modeling import FoldFeatureBuilder                      # noqa: E402
from habitat1p19q.modeling import model_columns as habitat_columns       # noqa: E402
from habitat1p19q.plotting import apply_style, plt                        # noqa: E402
from habitat1p19q.reporting import Reporter, environment, jsonable        # noqa: E402
from habitat1p19q.stats import nadeau_bengio                              # noqa: E402

from cmdf1p19q.config import build_cmdf_config                            # noqa: E402
from cmdf1p19q.features import model_columns as cmdf_columns              # noqa: E402
from cmdf1p19q.study import CMDFStudy                                     # noqa: E402

FEATURE_SETS = {"habitat": ["M3", "M4"], "cmdf": ["V", "P+D"]}
LABEL = {"M3": "Habitat composition (M3)", "M4": "Habitat composition + spatial (M4)",
         "V": "Voxel CMDF + T2-FLAIR mismatch (V)", "P+D": "Patch radiomics + patch CMDF (P+D)"}
PRIMARY_PAIRS = [("V", "M4"), ("V", "M3"), ("P+D", "M4")]       # CMDF set − habitat set, per classifier
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SET_COLOR = {"M3": SLOTS[0], "M4": SLOTS[6], "V": SLOTS[1], "P+D": SLOTS[3]}
INK, MUTED = "#2b2b2a", "#8a8984"


def _pre(scale=True):
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    return steps + ([("scale", StandardScaler())] if scale else [])


def classifiers(seed):
    """name -> (pipeline, inner-CV grid). Grids are small: n_train is ~59 (UTSW) or ~120 (pooled)."""
    return {
        "LogReg-L2": (Pipeline(_pre() + [("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]),
                      {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}),
        "LogReg-ElasticNet": (Pipeline(_pre() + [("clf", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                                                            class_weight="balanced", max_iter=4000, tol=1e-3))]),
                              {"clf__C": [0.1, 1.0], "clf__l1_ratio": [0.2, 0.8]}),
        "SVM-RBF": (Pipeline(_pre() + [("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed))]),
                    {"clf__C": [0.3, 1.0, 3.0]}),
        "RandomForest": (Pipeline(_pre(False) + [("clf", RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                                                                class_weight="balanced_subsample", random_state=seed, n_jobs=1))]), None),
        "ExtraTrees": (Pipeline(_pre(False) + [("clf", ExtraTreesClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                                                            class_weight="balanced_subsample", random_state=seed, n_jobs=1))]), None),
        "GradientBoosting": (Pipeline(_pre(False) + [("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                                                            min_samples_leaf=5, l2_regularization=1.0, max_features=0.3,
                                                                                            class_weight="balanced", random_state=seed))]), None),
        "KNN": (Pipeline(_pre() + [("clf", KNeighborsClassifier())]), {"clf__n_neighbors": [5, 9, 15]}),
        "GaussianNB": (Pipeline(_pre() + [("clf", GaussianNB())]), None),
        "MLP": (Pipeline(_pre() + [("clf", MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=seed))]),
                {"clf__alpha": [0.01, 1.0, 10.0]}),
    }


def fit_tuned(est, grid, X, y, seed):
    if grid is None:
        return clone(est).fit(X, y), {}
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    gs = GridSearchCV(clone(est), grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=True).fit(X, y)
    return gs.best_estimator_, gs.best_params_


def train_threshold(est, X, y, seed):
    """Cut-off maximising balanced accuracy on inner out-of-fold predictions (training fold only, tuned params fixed)."""
    inner = StratifiedKFold(5, shuffle=True, random_state=seed + 101)
    oof = cross_val_predict(clone(est), X, y, cv=inner, method="predict_proba")[:, 1]
    best_t, best = 0.5, -np.inf
    for t in np.unique(oof):
        pred = oof >= t
        ba = (pred[y == 1].mean() + (~pred[y == 0]).mean()) / 2
        if ba > best:
            best, best_t = ba, t
    return float(best_t)


class JointBuilder:
    """Habitat and CMDF features for one split; each part fitted on training patients only."""

    def __init__(self, base, hcfg, patches, cmdf_builder):
        self.hab = FoldFeatureBuilder(base, hcfg, patches)
        self.cmdf = cmdf_builder

    def build(self, tr, te):
        Htr, Hte, info = self.hab.build(tr, te)
        assert set(info["habitat_fit_patients"]) <= set(tr), "habitat model saw a test patient"
        Ctr, Cte, cinfo = self.cmdf.build(tr, te)
        assert set(cinfo["scale_fit_patients"]) <= set(tr), "CMDF scaling saw a test patient"
        extra = [c for c in Ctr.columns if c not in Htr.columns]
        return Htr.join(Ctr[extra]), Hte.join(Cte[extra]), info["K"]


def columns(fs, F, K, add_cohort):
    cols = habitat_columns(fs, F, K) if fs in FEATURE_SETS["habitat"] else cmdf_columns(fs, F)
    return cols + (["cohort_UCSF"] if add_cohort else [])


def _fold(f, rep, tr, te, ids, y, builder, seed, add_cohort):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    tr_ids, te_ids = ids[tr], ids[te]
    assert not set(tr_ids) & set(te_ids)
    Ftr, Fte, K = builder.build(tr_ids, te_ids)
    recs, preds = [], {}
    for approach, sets in FEATURE_SETS.items():
        for fs in sets:
            cols = columns(fs, Ftr, K, add_cohort)
            Xtr, Xte = Ftr[cols].to_numpy(float), Fte[cols].to_numpy(float)
            for name, (est, grid) in classifiers(seed + f).items():
                t0 = time.time()
                model, params = fit_tuned(est, grid, Xtr, y[tr], seed + f)
                p = model.predict_proba(Xte)[:, 1]
                thr = train_threshold(model, Xtr, y[tr], seed + f)
                tn, fp, fn, tp = confusion_matrix(y[te], (p >= thr).astype(int), labels=[0, 1]).ravel()
                preds[(fs, name)] = p
                recs.append({"fold": f, "repeat": rep, "approach": approach, "feature_set": fs, "classifier": name,
                             "n_features": len(cols), "K": K, "auc": roc_auc_score(y[te], p), "threshold": thr,
                             "tp": tp, "fp": fp, "tn": tn, "fn": fn, "params": json.dumps(params), "seconds": round(time.time() - t0, 2),
                             "n_train": len(tr), "n_test": len(te)})
    return f, rep, te, recs, preds


def run_cv(ids, base, builder, seed, n_repeats, n_jobs, add_cohort, log):
    from joblib import Parallel, delayed
    df = base.loc[ids]
    ids = df.index.to_numpy()
    y = df["label"].to_numpy(int)
    strata = (df["cohort"] + "_" + df["label"].astype(str)).to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed).split(np.zeros(len(ids)), strata))
    t0 = time.time()
    outs = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_fold)(f, f // 5, tr, te, ids, y, builder, seed, add_cohort)
                                              for f, (tr, te) in enumerate(splits))
    oof, recs = {}, []
    for f, rep, te, r, preds in sorted(outs, key=lambda o: o[0]):
        recs += r
        for k, p in preds.items():
            oof.setdefault(k, np.full((n_repeats, len(ids)), np.nan))[rep, te] = p
    log(f"  {len(splits)} folds in {time.time() - t0:.0f} s")
    return y, oof, pd.DataFrame(recs)


def summarise(y, oof, recs, set_name):
    rows = []
    for (fs, name), P in oof.items():
        avg = np.nanmean(P, 0)
        a, pr = st.auc_with_ci(y, avg, seed=11), st.auprc_with_ci(y, avg, seed=12)
        r = recs[(recs["feature_set"] == fs) & (recs["classifier"] == name)]
        per = r.groupby("repeat")[["tp", "fp", "tn", "fn"]].sum()
        tp, fp, tn, fn = (per[k].astype(float) for k in ["tp", "fp", "tn", "fn"])
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        ppv, npv = tp / (tp + fp).replace(0, np.nan), tn / (tn + fn).replace(0, np.nan)
        mcc = [matthews_corrcoef(np.r_[np.ones(int(a1 + b1)), np.zeros(int(c1 + d1))],
                                 np.r_[np.ones(int(a1)), np.zeros(int(b1)), np.zeros(int(c1)), np.ones(int(d1))])
               for a1, b1, c1, d1 in zip(tp, fn, tn, fp)]
        rows.append({"set": set_name, "approach": r["approach"].iloc[0], "feature_set": fs, "classifier": name,
                     "n": len(y), "n_codeleted": int(y.sum()), "n_features": int(r["n_features"].median()),
                     "AUROC": a["auc"], "AUROC 95% CI": f"{a['lo']:.3f}–{a['hi']:.3f}", "AUROC fold mean": r["auc"].mean(),
                     "AUPRC": pr["auprc"], "Brier": brier_score_loss(y, avg),
                     "precision (PPV)": ppv.mean(), "recall (sensitivity)": sens.mean(), "specificity": spec.mean(),
                     "NPV": npv.mean(), "F1": (2 * tp / (2 * tp + fp + fn)).mean(), "balanced accuracy": ((sens + spec) / 2).mean(),
                     "accuracy": ((tp + tn) / (tp + tn + fp + fn)).mean(), "MCC": float(np.mean(mcc)),
                     "precision SD over repeats": ppv.std(ddof=1), "recall SD over repeats": sens.std(ddof=1),
                     "_lo": a["lo"], "_hi": a["hi"]})
    return pd.DataFrame(rows)


def paired(recs, set_name):
    rows = []
    piv = recs.pivot_table(index=["fold", "classifier"], columns="feature_set", values="auc")
    n_tr, n_te = recs["n_train"].mean(), recs["n_test"].mean()
    for name in recs["classifier"].unique():
        sub = piv.xs(name, level="classifier")
        for t, r in PRIMARY_PAIRS:
            nb = nadeau_bengio((sub[t] - sub[r]).to_numpy(), n_tr, n_te)
            rows.append({"set": set_name, "classifier": name, "comparison": f"{t} − {r}", "ΔAUROC (fold mean)": nb["mean"],
                         "NB 95% CI": f"{nb['lo']:.3f} to {nb['hi']:.3f}", "NB p": nb["p"],
                         "folds CMDF better": f"{int(((sub[t] - sub[r]) > 0).sum())}/{len(sub)}"})
    out = pd.DataFrame(rows)
    out["NB p (BH, within set)"] = st.bh_fdr(out["NB p"].fillna(1).to_numpy())
    return out


# ------------------------------------------------------------------------------------------------- figures
def heatmap(summary, metric, title):
    apply_style()
    sets = list(dict.fromkeys(summary["set"]))
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.4), squeeze=False)
    present = set(summary["feature_set"])
    order = [fs for v in FEATURE_SETS.values() for fs in v if fs in present]
    for ax, s in zip(axes[0], sets):
        m = summary[summary["set"] == s].pivot(index="classifier", columns="feature_set", values=metric)[order]
        m = m.loc[m.mean(1).sort_values(ascending=False).index]
        im = ax.imshow(m.to_numpy(), cmap="Blues", vmin=0.4 if metric != "MCC" else -0.1, vmax=0.9 if metric != "MCC" else 0.6, aspect="auto")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{'Habitat' if fs in FEATURE_SETS['habitat'] else 'CMDF'}\n{fs}" for fs in order], fontsize=8)
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(m.index, fontsize=8)
        ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > (0.72 if metric != "MCC" else 0.35) else INK)
        if len(order) == 4:
            ax.axvline(1.5, color="white", lw=3)        # habitat | CMDF
        ax.set_title(f"{s}", fontsize=10, color=INK)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def best_roc(y, oof, summary, set_name):
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 4.8))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    s = summary[summary["set"] == set_name]
    for fs in [x for v in FEATURE_SETS.values() for x in v]:
        r = s[s["feature_set"] == fs].sort_values("AUROC", ascending=False).iloc[0]
        avg = np.nanmean(oof[(fs, r["classifier"])], 0)
        fpr, tpr, _ = roc_curve(y, avg)
        ax.plot(fpr, tpr, color=SET_COLOR[fs], label=f"{fs} · {r['classifier']}  {r['AUROC']:.3f}")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, loc="lower right", title="feature set · best classifier  AUROC", title_fontsize=7.5)
    ax.set_title(f"{set_name}: best classifier per feature set (descriptive)", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


def pr_scatter(summary, set_name):
    apply_style()
    s = summary[summary["set"] == set_name]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    for fs, g in s.groupby("feature_set"):
        ax.scatter(g["recall (sensitivity)"], g["precision (PPV)"], s=46, color=SET_COLOR[fs], edgecolor="white", lw=1.2,
                   label=LABEL[fs])
    prev = s["n_codeleted"].iloc[0] / s["n"].iloc[0]
    ax.axhline(prev, color=MUTED, ls="--", lw=1)
    ax.text(0.02, prev, f" prevalence {prev:.2f}", va="bottom", fontsize=7.5, color=INK, transform=ax.get_yaxis_transform())
    ax.set_xlabel("Recall (sensitivity), training-fold threshold")
    ax.set_ylabel("Precision (PPV)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title(f"{set_name}: precision vs recall, one point per classifier", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--sets", default="UTSW,pooled")
    ap.add_argument("--out", default=str(ROOT / "models"), help="output folder (a scratch folder for smoke tests)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    for sub in ["habitat", "cmdf"]:
        for d in ["tables", "figures"]:
            (out / sub / d).mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    rep = Reporter(out)
    hcfg, ccfg = build_cmdf_config()
    st.set_default_n_boot(hcfg["n_bootstrap"])
    study = CMDFStudy(hcfg, ccfg)
    study.protocol_sha = "models"
    study.rep = rep
    study.stage_data()
    study.stage_voxel()
    study.stage_blocks()
    builder = JointBuilder(study.base, hcfg, study.patches[study.patches["patient_id"].isin(study.ids["pooled"])], study.builder)
    t0 = time.time()
    summaries, pairs, all_recs, oofs = [], [], [], {}
    for set_name, add_cohort in [(s, s == "pooled") for s in a.sets.split(",")]:
        ids = study.ids[set_name]
        rep.say(f"\n=== {set_name}: {len(ids)} patients | {sum(len(v) for v in FEATURE_SETS.values())} feature sets x "
                f"{len(classifiers(0))} classifiers | {a.repeats} x 5-fold")
        y, oof, recs = run_cv(ids, study.base, builder, hcfg["seed"], a.repeats, a.n_jobs, add_cohort, rep.say)
        recs["set"] = set_name
        summ = summarise(y, oof, recs, set_name)
        summaries.append(summ)
        pairs.append(paired(recs, set_name))
        all_recs.append(recs)
        oofs[set_name] = (y, oof)
    summary = pd.concat(summaries, ignore_index=True)
    pair_df = pd.concat(pairs, ignore_index=True)
    recs = pd.concat(all_recs, ignore_index=True)
    show = [c for c in summary if not c.startswith("_")]

    for approach in FEATURE_SETS:
        d = out / approach
        sa = summary[summary["approach"] == approach].sort_values(["set", "AUROC"], ascending=[True, False])
        sa[show].to_csv(d / "tables" / "metrics_by_classifier.csv", index=False)
        recs[recs["approach"] == approach].to_csv(d / "tables" / "fold_records.csv", index=False)
        for metric, fname in [("AUROC", "auroc_heatmap"), ("precision (PPV)", "precision_heatmap"), ("F1", "f1_heatmap")]:
            fig = heatmap(summary[summary["approach"] == approach], metric, f"{approach}: {metric} by classifier and feature set")
            fig.savefig(d / "figures" / f"{fname}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        (d / "README.md").write_text(approach_readme(approach, sa), encoding="utf-8")
    summary[show].to_csv(out / "tables" / "comparison_all_metrics.csv", index=False)
    pair_df.to_csv(out / "tables" / "comparison_paired_tests.csv", index=False)
    for metric, fname, title in [("AUROC", "comparison_auroc_heatmap", "AUROC (repeat-averaged OOF)"),
                                 ("precision (PPV)", "comparison_precision_heatmap", "Precision (PPV) at the training-fold threshold"),
                                 ("recall (sensitivity)", "comparison_recall_heatmap", "Recall (sensitivity) at the training-fold threshold"),
                                 ("F1", "comparison_f1_heatmap", "F1 at the training-fold threshold"),
                                 ("MCC", "comparison_mcc_heatmap", "Matthews correlation at the training-fold threshold")]:
        fig = heatmap(summary, metric, title)
        fig.savefig(out / "figures" / f"{fname}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    for s in oofs:
        y, oof = oofs[s]
        for fig, nm in [(best_roc(y, oof, summary, s), f"best_roc_{s}"), (pr_scatter(summary, s), f"precision_recall_{s}")]:
            fig.savefig(out / "figures" / f"{nm}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
    (out / "comparison_report.md").write_text(report(summary, pair_df, a, time.time() - t0), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"environment": environment(), "repeats": a.repeats, "seed": hcfg["seed"],
                                                   "feature_sets": FEATURE_SETS, "classifiers": list(classifiers(0)),
                                                   "seconds": round(time.time() - t0)}, indent=1, default=jsonable), encoding="utf-8")
    rep.say((out / "comparison_report.md").read_text(encoding="utf-8"))


def _fmt_table(df, cols, digits=3):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        body += "| " + " | ".join(f"{r[c]:.{digits}f}" if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |\n"
    return head + body


METRIC_COLS = ["classifier", "feature_set", "n_features", "AUROC", "AUROC 95% CI", "AUPRC", "precision (PPV)",
               "recall (sensitivity)", "specificity", "NPV", "F1", "MCC", "balanced accuracy", "Brier"]


def approach_readme(approach, sa):
    t = f"# {approach.upper()} feature sets — nine classifiers\n\n"
    t += "Feature sets: " + "; ".join(f"**{fs}** = {LABEL[fs]}" for fs in FEATURE_SETS[approach]) + ".\n\n"
    t += "Threshold-dependent metrics (precision … balanced accuracy) use a cut-off chosen on the training fold only, "
    t += "averaged over repeats. AUROC/AUPRC/Brier use repeat-averaged out-of-fold predictions.\n\n"
    for s in dict.fromkeys(sa["set"]):
        t += f"## {s}\n\n" + _fmt_table(sa[sa["set"] == s], METRIC_COLS) + "\n"
    return t


def report(summary, pair_df, a, secs):
    t = "# Habitat vs CMDF — classifier comparison\n\n"
    t += (f"Nine classifiers × four feature sets (habitat M3, M4; CMDF V, P+D), {a.repeats} × repeated stratified 5-fold CV "
          "on patients, every learned step (habitat model, CMDF scaling, imputation, scaling, hyper-parameters, threshold) "
          f"fitted on the training fold only. Pooled analyses include the cohort indicator. Runtime {secs / 60:.0f} min.\n\n")
    t += ("**Reading the numbers.** Precision (PPV) depends on prevalence: 0.49 codeleted in UTSW, 0.29 pooled. "
          "Threshold-dependent metrics come from a training-fold cut-off maximising balanced accuracy, so they are honest "
          "out-of-sample values, not the best achievable cut-off. Choosing the best of 9 classifiers after the fact is optimistic; "
          "the pre-specified estimator in both studies is L2 logistic regression.\n\n")
    for s in dict.fromkeys(summary["set"]):
        ss = summary[summary["set"] == s]
        t += f"## {s} (n = {ss['n'].iloc[0]}, {ss['n_codeleted'].iloc[0]} codeleted)\n\n"
        t += "### Best classifier per feature set (by AUROC)\n\n"
        best = ss.sort_values("AUROC", ascending=False).groupby("feature_set", sort=False).head(1)
        t += _fmt_table(best, ["feature_set", "classifier"] + METRIC_COLS[2:]) + "\n"
        t += "### Mean over the nine classifiers\n\n"
        mean = ss.groupby(["approach", "feature_set"])[["AUROC", "AUPRC", "precision (PPV)", "recall (sensitivity)", "specificity",
                                                        "F1", "MCC", "balanced accuracy"]].mean().reset_index()
        t += _fmt_table(mean, list(mean.columns)) + "\n"
        t += "### Pre-specified estimator (L2 logistic regression)\n\n"
        t += _fmt_table(ss[ss["classifier"] == "LogReg-L2"], METRIC_COLS[1:]) + "\n"
        t += "### Paired CMDF − habitat ΔAUROC per classifier (Nadeau–Bengio, BH within set)\n\n"
        pp = pair_df[pair_df["set"] == s]
        t += _fmt_table(pp, ["classifier", "comparison", "ΔAUROC (fold mean)", "NB 95% CI", "NB p", "NB p (BH, within set)",
                             "folds CMDF better"]) + "\n"
        wins = pp.groupby("comparison").apply(lambda g: f"{int((g['ΔAUROC (fold mean)'] > 0).sum())}/{len(g)} classifiers favour CMDF; "
                                                        f"{int((g['NB p (BH, within set)'] < 0.05).sum())} significant after BH",
                                              include_groups=False)
        t += "".join(f"- {k}: {v}\n" for k, v in wins.items()) + "\n"
    t += "## Full per-classifier tables\n\n- `habitat/README.md`, `cmdf/README.md`, `tables/comparison_all_metrics.csv`\n"
    return t


if __name__ == "__main__":
    main()
