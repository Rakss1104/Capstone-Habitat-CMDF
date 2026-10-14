"""Baseline model ladder, estimators and fold-wise feature construction (notebook section 11).

| Model   | Features                                                        |
| M0      | cohort indicator (pooled analyses only)                         |
| M1_clin | age, sex                                                        |
| M1      | log tumour volume, age, sex                                     |
| M2      | whole-tumour radiomics (wt_*)                                   |
| M3      | habitat composition                                             |
| M4      | M3 + composition-adjusted spatial metrics  (primary: M4 vs M3)  |
| M5      | M4 + log volume, age, sex                                       |
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .habitat import HabitatModel
from .spatial import build_patient_features, composition_cols, spatial_cols
from .stats import cramers_v

BASE_MODELS = ["M0", "M1_clin", "M1", "M2", "M3", "M4", "M5"]


def make_logreg(seed, cfg):
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))])
    inner = StratifiedKFold(cfg["cv"]["inner_splits"], shuffle=True, random_state=seed)
    return GridSearchCV(pipe, {"clf__C": cfg["logreg_C_grid"]}, scoring="roc_auc", cv=inner, n_jobs=1, refit=True)


def make_rf(seed, cfg):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(**cfg["rf"], random_state=seed, n_jobs=1))])


ESTIMATORS = {"logreg": make_logreg, "rf": make_rf}


def make_estimator(name, seed, cfg):
    return ESTIMATORS[name](seed, cfg)


def train_only_threshold(est_name, X, y, seed, cfg):
    """Cut-off maximising balanced accuracy on inner out-of-fold predictions of the training fold only."""
    inner = StratifiedKFold(cfg["cv"]["inner_splits"], shuffle=True, random_state=seed + 101)
    oof = cross_val_predict(make_estimator(est_name, seed, cfg), X, y, cv=inner, method="predict_proba")[:, 1]
    best_t, best = 0.5, -np.inf
    for t in np.unique(oof):
        pred = oof >= t
        ba = (pred[y == 1].mean() + (~pred[y == 0]).mean()) / 2
        if ba > best:
            best, best_t = ba, t
    return float(best_t)


def model_columns(model, F, K):
    base = model.replace("+cohort", "")
    extra = ["cohort_UCSF"] if model.endswith("+cohort") else []
    comp = composition_cols(K) if K else None
    spat = spatial_cols(K) if K else None
    spec = {"M0": ["cohort_UCSF"], "M1_clin": ["age", "sex_M"], "M1": ["log_volume_ml", "age", "sex_M"],
            "M2": [c for c in F.columns if c.startswith("wt_")] or None,
            "M3": comp, "M4": (comp + spat) if comp else None,
            "M5": (comp + spat + ["log_volume_ml", "age", "sex_M"]) if comp else None}[base]
    if spec is None or not all(c in F.columns for c in spec):
        return None
    return spec + extra


def build_base_table(study, patient_img, imaging_available):
    """Patient-level base table (clinical + imaging summaries), indexed by patient_id."""
    base = study.set_index("patient_id")[["cohort", "label", "age", "sex", "in_primary"]].copy()
    base["label"] = base["label"].astype(int)
    base["in_primary"] = base["in_primary"].astype(bool)
    base["sex_M"] = (base["sex"] == "M").astype(float)
    base["cohort_UCSF"] = (base["cohort"] == "UCSF").astype(float)
    if imaging_available and len(patient_img):
        vol = patient_img["tumour_volume_ml"]
        base = base.join(np.log(vol.where(vol > 0)).rename("log_volume_ml"))
        base = base.join(patient_img[[c for c in patient_img.columns if c.startswith("wt_")]])
    return base


class FoldFeatureBuilder:
    """Builds patient-level features for one train/test split.

    All learned steps (the habitat model) are fitted on training patients only; held-out patches are assigned with
    the training model. Habitat models are label-free, so they are cached per training set and reused by the
    permutation tests.
    """

    def __init__(self, base, cfg, patches=None, harmonisation=None, fixed_k=None):
        self.base, self.cfg = base, cfg
        self.patches = patches
        self.harmonisation, self.fixed_k = harmonisation, fixed_k
        self.cache, self.feat_cache = {}, {}
        self.has_imaging = patches is not None and len(patches) > 0

    def key(self, train_ids):
        return hashlib.sha1(("|".join(sorted(train_ids)) + f"|{self.harmonisation}|{self.fixed_k}").encode()).hexdigest()[:16]

    def build(self, train_ids, test_ids, shuffle_seed=None):
        Ftr, Fte = self.base.loc[train_ids].copy(), self.base.loc[test_ids].copy()
        info = {"K": None, "habitat_fit_patients": []}
        if not self.has_imaging:
            return Ftr, Fte, info
        k = self.key(train_ids)
        if k not in self.cache:
            ptr = self.patches[self.patches["patient_id"].isin(train_ids)]
            hm = HabitatModel(self.cfg, k=self.fixed_k, seed=self.cfg["seed"], harmonisation=self.harmonisation).fit(ptr)
            self.cache[k] = {"model": hm, "assigned": {}}
        entry = self.cache[k]
        hm = entry["model"]
        ids = list(train_ids) + list(test_ids)
        need = [p for p in ids if p not in entry["assigned"]]
        if need:
            sub = self.patches[self.patches["patient_id"].isin(need)]
            if len(sub):
                lab = hm.predict(sub)
                slim = sub[["patient_id", "cohort", "z0", "y0", "x0", "depth_norm"]].assign(habitat=lab)
                for pid, g in slim.groupby("patient_id"):
                    entry["assigned"][pid] = g
        fkey = (k, shuffle_seed, hashlib.sha1("|".join(sorted(test_ids)).encode()).hexdigest()[:12])
        if fkey not in self.feat_cache:
            frames = [entry["assigned"][p] for p in ids if p in entry["assigned"]]
            seed = self.cfg["seed"] if shuffle_seed is None else self.cfg["seed"] + 7919 * (shuffle_seed + 1)
            self.feat_cache[fkey] = build_patient_features(pd.concat(frames), hm.k_, seed, self.cfg,
                                                           shuffle_spatial=shuffle_seed is not None)
        pf = self.feat_cache[fkey]
        Ftr, Fte = Ftr.join(pf), Fte.join(pf)
        info.update(K=hm.k_, habitat_fit_patients=hm.train_patients_,
                    transductive_cohorts=list(getattr(hm, "transductive_cohorts_", [])))
        if self.harmonisation is None and "cohort" in self.patches:
            tr_p = entry["assigned"]
            hab = pd.concat([tr_p[p] for p in train_ids if p in tr_p])
            ct = pd.crosstab(hab["habitat"], hab["cohort"])
            info["cramers_v_train"] = cramers_v(ct.values) if ct.shape[1] > 1 else np.nan
        return Ftr, Fte, info
