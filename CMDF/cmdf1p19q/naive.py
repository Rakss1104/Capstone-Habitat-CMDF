"""Methodological arm (PROTOCOL.md section 7): the original CMDF notebooks' evaluation protocol on the same features.

Variant "notebook" reproduces `train_test_evaluate_cmdf_final.ipynb` (pipeline_cmdf.py): one StratifiedKFold(5,
shuffle, seed 42); inside each fold StandardScaler and top-30 mutual-information selection fitted on the training part;
KNN(5), decision tree, Gaussian NB and random forest (100 trees), plus logistic regression; the classifier with the best
mean fold AUROC is reported. Variant "select_all" fits the scaler and the selection once on all patients before
splitting, as the notebooks' final model-export cell does and as many published radiomics pipelines do.

Each variant is re-run on permuted labels to show the AUROC it reports when there is no signal. Not replicated: the
VAE oversampling and the 1D-CNN / attention-MIL models (TensorFlow is not part of this environment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

VARIANTS = ["notebook", "select_all"]


def classifiers(seed):
    return {"KNN": KNeighborsClassifier(n_neighbors=5), "DT": DecisionTreeClassifier(random_state=seed),
            "GNB": GaussianNB(), "RF": RandomForestClassifier(n_estimators=100, random_state=seed),
            "LR": LogisticRegression(max_iter=5000)}


def _prep(Xfit, yfit, Xapply, top_k, seed):
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xfit)
    sc = StandardScaler().fit(imp.transform(Xfit))
    Zfit = np.nan_to_num(sc.transform(imp.transform(Xfit)))
    sel = np.argsort(-mutual_info_classif(Zfit, yfit, random_state=seed))[:top_k]
    return [np.nan_to_num(sc.transform(imp.transform(A)))[:, sel] for A in Xapply]


def naive_protocol(X, y, top_k, seed, variant):
    """Per-classifier fold-mean AUROC and SD under the given variant."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    if variant == "select_all":
        (Xall,) = _prep(X, y, [X], top_k, seed)
    out = {n: [] for n in classifiers(seed)}
    for tr, te in skf.split(X, y):
        if variant == "notebook":
            Xtr, Xte = _prep(X[tr], y[tr], [X[tr], X[te]], top_k, seed)
        else:
            Xtr, Xte = Xall[tr], Xall[te]
        for name, clf in classifiers(seed).items():
            clf.fit(Xtr, y[tr])
            out[name].append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return {n: (float(np.mean(a)), float(np.std(a))) for n, a in out.items()}


def _null_chunk(X, y, top_k, seed, variant, perm_seeds):
    res = []
    for s in perm_seeds:
        yp = np.random.default_rng(int(s)).permutation(y)
        res.append(max(v[0] for v in naive_protocol(X, yp, top_k, seed, variant).values()))
    return res


def run_naive_arm(F, cols, y, ccfg, n_jobs, rep):
    from joblib import Parallel, delayed
    X = F[cols].to_numpy(float)
    k, seed = ccfg["naive_top_k"], ccfg["naive_seed"]
    rows, summaries, nulls = [], [], {}
    for variant in VARIANTS:
        obs = naive_protocol(X, y, k, seed, variant)
        rows += [{"variant": variant, "classifier": n, "fold-mean AUROC": m, "fold SD": s} for n, (m, s) in obs.items()]
        best_name = max(obs, key=lambda n: obs[n][0])
        seeds = 900_000 + np.arange(ccfg["naive_permutations"])
        chunks = [c for c in np.array_split(seeds, max(1, min(n_jobs, len(seeds)))) if len(c)]
        nd = np.array([x for part in Parallel(n_jobs=n_jobs)(delayed(_null_chunk)(X, y, k, seed, variant, c) for c in chunks)
                       for x in part])
        best = obs[best_name][0]
        p = (1 + (nd >= best).sum()) / (1 + len(nd))
        rep.say(f"[{variant}] best classifier {best_name}: AUROC {best:.3f} ± {obs[best_name][1]:.3f}. On permuted labels "
                f"({len(nd)}): best-classifier AUROC mean {nd.mean():.3f}, 95th pct {np.percentile(nd, 95):.3f}, "
                f"max {nd.max():.3f}; permutation p = {p:.3f}")
        summaries.append({"variant": variant, "best_classifier": best_name, "best_auc": best, "best_sd": obs[best_name][1],
                          "null_mean": float(nd.mean()), "null_95": float(np.percentile(nd, 95)), "null_max": float(nd.max()),
                          "p": float(p), "n_perm": int(len(nd)), "n_features_in": len(cols)})
        nulls[variant] = nd
    return pd.DataFrame(rows), pd.DataFrame(summaries), nulls
