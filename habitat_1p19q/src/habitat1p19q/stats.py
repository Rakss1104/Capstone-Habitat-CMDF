"""Statistical helpers used throughout, so that intervals and effect sizes are computed one way everywhere."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEFAULT_N_BOOT = 2000


def set_default_n_boot(n):
    """Bootstrap resamples used when a caller does not pass n_boot (CONFIG["n_bootstrap"] in the notebook)."""
    global DEFAULT_N_BOOT
    DEFAULT_N_BOOT = int(n)


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (centre - half, centre + half)


def stratified_bootstrap_indices(y, n_boot, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])


def auc_with_ci(y, score, n_boot=None, seed=0):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    n_boot = n_boot or DEFAULT_N_BOOT
    if len(np.unique(y)) < 2:
        return {"auc": np.nan, "lo": np.nan, "hi": np.nan, "n": len(y), "n_pos": int(y.sum())}
    auc = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    boots = [roc_auc_score(y[i], score[i]) for i in stratified_bootstrap_indices(y, n_boot, rng)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": auc, "lo": lo, "hi": hi, "n": len(y), "n_pos": int(y.sum())}


def auprc_with_ci(y, score, n_boot=None, seed=0):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    n_boot = n_boot or DEFAULT_N_BOOT
    ap = average_precision_score(y, score)
    rng = np.random.default_rng(seed)
    boots = [average_precision_score(y[i], score[i]) for i in stratified_bootstrap_indices(y, n_boot, rng)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auprc": ap, "lo": lo, "hi": hi, "prevalence": y.mean()}


def delta_auc_bootstrap(y, s_test, s_ref, n_boot=None, seed=0):
    y = np.asarray(y).astype(int)
    n_boot = n_boot or DEFAULT_N_BOOT
    rng = np.random.default_rng(seed)
    d = roc_auc_score(y, s_test) - roc_auc_score(y, s_ref)
    boots = [roc_auc_score(y[i], s_test[i]) - roc_auc_score(y[i], s_ref[i])
             for i in stratified_bootstrap_indices(y, n_boot, rng)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": d, "lo": lo, "hi": hi}


def rank_biserial(x_pos, x_neg):
    """Cliff's delta. > 0 means values tend to be higher in the codeleted group."""
    x_pos, x_neg = np.asarray(x_pos, float), np.asarray(x_neg, float)
    x_pos, x_neg = x_pos[~np.isnan(x_pos)], x_neg[~np.isnan(x_neg)]
    u = stats.mannwhitneyu(x_pos, x_neg, alternative="two-sided")
    r = 2 * u.statistic / (len(x_pos) * len(x_neg)) - 1
    return r, u.pvalue


def rank_biserial_ci(x_pos, x_neg, n_boot=None, seed=0):
    rng = np.random.default_rng(seed)
    x_pos, x_neg = np.asarray(x_pos, float), np.asarray(x_neg, float)
    n_boot = n_boot or DEFAULT_N_BOOT
    b = []
    for _ in range(n_boot):
        bp, bn = rng.choice(x_pos, len(x_pos)), rng.choice(x_neg, len(x_neg))
        b.append(2 * stats.mannwhitneyu(bp, bn).statistic / (len(bp) * len(bn)) - 1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def cramers_v(table):
    table = np.asarray(table, float)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    return math.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else np.nan


def nadeau_bengio(diffs, n_train, n_test, alpha=0.05):
    """Corrected resampled t-test for repeated k-fold CV (Nadeau & Bengio, 2003)."""
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    J = len(d)
    mean, var = d.mean(), d.var(ddof=1)
    se = math.sqrt((1 / J + n_test / n_train) * var) if var > 0 else 0.0
    if se == 0:
        return {"mean": mean, "lo": mean, "hi": mean, "t": np.nan, "p": np.nan, "J": J}
    t = mean / se
    tcrit = stats.t.ppf(1 - alpha / 2, J - 1)
    return {"mean": mean, "lo": mean - tcrit * se, "hi": mean + tcrit * se, "t": t,
            "p": 2 * stats.t.sf(abs(t), J - 1), "J": J}


def calibration_stats(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    slope_m = LogisticRegression(C=1e6, max_iter=5000).fit(lp[:, None], y)
    # calibration-in-the-large: intercept with the logit as a fixed offset (Newton iterations)
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp)))
        g = (y - mu).sum()
        h = (mu * (1 - mu)).sum()
        a += g / h
    return {"slope": float(slope_m.coef_[0, 0]), "intercept_citl": a, "brier": brier_score_loss(y, p)}


def fmt_ci(est, lo, hi, d=3):
    return f"{est:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"
