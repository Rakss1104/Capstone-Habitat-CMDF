"""Exploratory analysis and cohort audit (notebook section 6). Runs on metadata only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from .cohort import CLASS_NAME
from .plotting import plt
from .stats import auc_with_ci, fmt_ci, rank_biserial, rank_biserial_ci, wilson_ci


def describe_cohort(df):
    rows = {}
    for (coh, lab), g in df.groupby(["cohort", "label"]):
        age = g["age"]
        rows[f"{coh} | {CLASS_NAME[lab]}"] = {
            "n": len(g),
            "age, mean (SD)": f"{age.mean():.1f} ({age.std():.1f})",
            "age, median [IQR]": f"{age.median():.0f} [{age.quantile(.25):.0f}–{age.quantile(.75):.0f}]",
            "male, n (%)": f"{(g['sex'] == 'M').sum()} ({100 * (g['sex'] == 'M').mean():.0f}%)",
            "grade 2 / 3 / 4 / NA": " / ".join(str(int((g['grade'] == k).sum())) for k in [2, 3, 4]) + f" / {int(g['grade'].isna().sum())}",
            "field strength 1.5 T / 3 T / other / NA": (
                f"{int((g['field_strength_T'] == 1.5).sum())} / {int((g['field_strength_T'] == 3).sum())} / "
                f"{int((g['field_strength_T'].notna() & ~g['field_strength_T'].isin([1.5, 3])).sum())} / "
                f"{int(g['field_strength_T'].isna().sum())}"),
            "segmentation provenance": "; ".join(f"{k}: {v}" for k, v in g["seg_provenance"].value_counts().items()),
        }
    return pd.DataFrame(rows)


def site_association(eligible, rep):
    """Site–label association and the analytic cohort-only baseline."""
    prev_rows = []
    for coh, g in eligible.groupby("cohort"):
        k, n = int(g["label"].sum()), len(g)
        lo, hi = wilson_ci(k, n)
        prev_rows.append({"cohort": coh, "n": n, "codeleted": k, "prevalence": k / n, "95% CI": f"{lo:.2f}–{hi:.2f}"})
    prevalence = pd.DataFrame(prev_rows).set_index("cohort")
    rep.show(prevalence, "Codeletion prevalence by cohort:")

    ct = pd.crosstab(eligible["cohort"], eligible["label"]).loc[["UTSW", "UCSF"], [1, 0]]
    or_res = stats.contingency.odds_ratio(ct.values)
    or_ci = or_res.confidence_interval()
    fisher_p = stats.fisher_exact(ct.values)[1]
    site_auc = auc_with_ci(eligible["label"], eligible["cohort"].eq("UTSW").astype(float), seed=1)
    rep.say(f"Odds of codeletion, UTSW vs UCSF: OR {fmt_ci(or_res.statistic, or_ci.low, or_ci.high, 2)}, Fisher p = {fisher_p:.2g}")
    rep.say(f"AUROC of cohort membership alone (pooled eligible cohort, n = {site_auc['n']}, {site_auc['n_pos']} codeleted): "
            f"{fmt_ci(site_auc['auc'], site_auc['lo'], site_auc['hi'])}")
    rep.record_check("Cohort–label association quantified", True, f"cohort-only AUROC {site_auc['auc']:.3f}")
    rep.save_table(prevalence, "06_prevalence_by_cohort")
    return prevalence, {"odds_ratio_UTSW_vs_UCSF": or_res.statistic, "or_lo": or_ci.low, "or_hi": or_ci.high,
                        "fisher_p": fisher_p, "cohort_only_auc": site_auc}


def clinical_associations(eligible):
    """Pre-operative clinical variables by label, within each cohort."""
    rows = []
    for coh, g in eligible.groupby("cohort"):
        pos, neg = g.loc[g["label"] == 1, "age"], g.loc[g["label"] == 0, "age"]
        r, p = rank_biserial(pos, neg)
        lo, hi = rank_biserial_ci(pos, neg, seed=2)
        rows.append({"cohort": coh, "variable": "age (years)", "codeleted median": pos.median(), "non-codeleted median": neg.median(),
                     "effect": "rank-biserial r (>0: older if codeleted)", "estimate (95% CI)": fmt_ci(r, lo, hi, 2), "p": p})
        ct_sex = pd.crosstab(g["sex"], g["label"]).reindex(index=["M", "F"], columns=[1, 0], fill_value=0)
        o = stats.contingency.odds_ratio(ct_sex.values)
        oci = o.confidence_interval()
        rows.append({"cohort": coh, "variable": "male sex", "codeleted median": np.nan, "non-codeleted median": np.nan,
                     "effect": "odds ratio (male vs female)", "estimate (95% CI)": fmt_ci(o.statistic, oci.low, oci.high, 2),
                     "p": stats.fisher_exact(ct_sex.values)[1]})
    g = eligible[eligible["cohort"].eq("UTSW") & eligible["field_strength_T"].isin([1.5, 3.0])]
    ct_fs = pd.crosstab(g["field_strength_T"], g["label"]).reindex(index=[3.0, 1.5], columns=[1, 0], fill_value=0)
    o = stats.contingency.odds_ratio(ct_fs.values)
    oci = o.confidence_interval()
    rows.append({"cohort": "UTSW", "variable": "3 T (vs 1.5 T) scanner", "codeleted median": np.nan, "non-codeleted median": np.nan,
                 "effect": "odds ratio", "estimate (95% CI)": fmt_ci(o.statistic, oci.low, oci.high, 2),
                 "p": stats.fisher_exact(ct_fs.values)[1]})
    return pd.DataFrame(rows)


def audit_figure(eligible, prevalence, pal):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1.4, 1.2]})
    ax = axes[0]
    for i, (coh, row) in enumerate(prevalence.iterrows()):
        lo, hi = wilson_ci(row["codeleted"], row["n"])
        ax.errorbar(i, row["prevalence"], yerr=[[row["prevalence"] - lo], [hi - row["prevalence"]]], fmt="o", ms=8,
                    color=pal.cohort[coh], capsize=4)
        ax.annotate(f"{int(row['codeleted'])}/{int(row['n'])}", (i, hi), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=9)
    ax.set_xticks(range(len(prevalence)))
    ax.set_xticklabels(prevalence.index)
    ax.set_xlim(-0.6, len(prevalence) - 0.4)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Codeletion prevalence (95% Wilson CI)")
    ax.set_title("Outcome prevalence differs by cohort")

    ax = axes[1]
    pos, ticks, labels = 0, [], []
    for coh in ["UTSW", "UCSF"]:
        for lab in [0, 1]:
            vals = eligible.loc[eligible["cohort"].eq(coh) & eligible["label"].eq(lab), "age"].values
            col = pal.cls[lab]
            ax.boxplot(vals, positions=[pos], widths=0.55, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=col, alpha=0.35, edgecolor=col), medianprops=dict(color=col, linewidth=2),
                       whiskerprops=dict(color=col), capprops=dict(color=col))
            jitter = np.random.default_rng(3).uniform(-0.15, 0.15, len(vals))
            ax.scatter(pos + jitter, vals, s=10, color=col, alpha=0.7, linewidths=0)
            ticks.append(pos)
            labels.append(f"{coh}\n{'codel.' if lab else 'non-codel.'}\n(n={len(vals)})")
            pos += 1
        pos += 0.6
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Age at imaging (years)")
    ax.set_title("Age by 1p/19q status within cohort")
    ax.legend(handles=[Patch(facecolor=pal.cls[k], alpha=0.6, label=CLASS_NAME[k]) for k in [0, 1]], loc="upper right", fontsize=8)

    ax = axes[2]
    g = eligible[eligible["cohort"].eq("UTSW")].copy()
    g["fs"] = g["field_strength_T"].map(lambda v: "NA" if pd.isna(v) else ("1.5 T" if v == 1.5 else ("3 T" if v == 3 else "<1.5 T / other")))
    order = ["<1.5 T / other", "1.5 T", "3 T", "NA"]
    cts = pd.crosstab(g["fs"], g["label"]).reindex(order, fill_value=0)
    x = np.arange(len(order))
    w = 0.38
    for j, lab in enumerate([0, 1]):
        ax.bar(x + (j - 0.5) * w, cts[lab].values, width=w - 0.04, color=pal.cls[lab], label=CLASS_NAME[lab])
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("Patients")
    ax.set_title("UTSW field strength by label")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def run_audit(cohort, rep, pal):
    eligible = cohort.eligible
    table1 = describe_cohort(eligible)
    rep.show(table1, "Table 1 (by cohort and label):")
    rep.save_table(table1, "06_table1_by_cohort_and_label")
    prevalence, site_assoc = site_association(eligible, rep)
    clinical = clinical_associations(eligible)
    rep.show(clinical, "Clinical and acquisition variables by label:")
    rep.save_table(clinical, "06_clinical_and_acquisition_by_label")
    rep.say("Grade is post-operative pathology: it is described above but never used as a predictor.")
    rep.banner()
    rep.save_fig(audit_figure(eligible, prevalence, pal), "06_cohort_audit")
    return {"table1": table1, "prevalence": prevalence, "site_assoc": site_assoc, "clinical": clinical}
