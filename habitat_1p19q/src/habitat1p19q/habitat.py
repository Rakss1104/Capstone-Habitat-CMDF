"""Habitat model (notebook section 9): feature filter -> standardise -> PCA (95% variance) -> k-means.

The whole model, including the choice of K, has fit/predict so it can be refitted inside every training fold.
K is the smallest value whose bootstrap-over-patients ARI is within `stability_tolerance` of the best, among solutions
where every habitat holds at least `min_habitat_share` of patches.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .plotting import plt
from .spatial import build_patient_features, composition_cols
from .stats import bh_fdr, cramers_v, rank_biserial


def habitat_columns(df, modalities):
    cols = [c for c in df.columns if any(c.startswith(f"{m}__") for m in modalities)]
    assert not any(c.startswith("shape__") for c in cols)
    return cols


class HabitatModel:
    """Unsupervised: never sees labels. fit() on training-patient patches only; predict() for any patches."""

    def __init__(self, cfg, k=None, seed=0, harmonisation=None):
        self.k_fixed, self.seed, self.harmonisation = k, seed, harmonisation
        self.hc = cfg["habitat"]
        self.modalities = cfg["modalities"]
        self.transductive_cohorts_ = []

    def _harmonise(self, X, cohorts, fit):
        # Optional per-cohort z-scoring (sensitivity analysis). Parameters come from training patches; a cohort never
        # seen in training (leave-one-cohort-out) gets label-free parameters from its own patches and is flagged.
        if self.harmonisation != "per_cohort_zscore":
            return X
        X = X.copy()
        if fit:
            self.coh_stats_ = {}
        for c in np.unique(cohorts):
            sel = cohorts == c
            if c not in self.coh_stats_:
                self.coh_stats_[c] = (X[sel].mean(0), X[sel].std(0) + 1e-8)
                if not fit:
                    self.transductive_cohorts_.append(c)
            mu, sd = self.coh_stats_[c]
            X[sel] = (X[sel] - mu) / sd
        return X

    def fit(self, df):
        cols = habitat_columns(df, self.modalities)
        X = df[cols].to_numpy(float)
        keep = np.isfinite(X).all(0) & (np.nanstd(X, 0) > 1e-8)
        self.cols_ = [c for c, k in zip(cols, keep) if k]
        self.dropped_ = [c for c, k in zip(cols, keep) if not k]
        X = self._harmonise(df[self.cols_].to_numpy(float), df["cohort"].to_numpy(), fit=True)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.pca_ = PCA(n_components=self.hc["pca_variance"] if min(Z.shape) > 2 else 1, random_state=self.seed).fit(Z)
        Zp = self.pca_.transform(Z)
        pids = df["patient_id"].to_numpy()
        if self.k_fixed is None and self.hc["fixed_k"] is None:
            self.k_, self.selection_ = self._select_k(Zp, pids)
        else:
            self.k_ = int(self.k_fixed or self.hc["fixed_k"])
            self.selection_ = None
        self.km_ = KMeans(n_clusters=self.k_, n_init=self.hc["kmeans_n_init"], random_state=self.seed).fit(Zp)
        self.train_patients_ = sorted(set(pids))
        return self

    def _select_k(self, Zp, pids):
        rng = np.random.default_rng(self.seed)
        uniq = np.unique(pids)
        by_pid = {p: np.flatnonzero(pids == p) for p in uniq}
        rows = []
        for k in self.hc["k_grid"]:
            if k >= len(Zp):
                continue
            ref = KMeans(n_clusters=k, n_init=self.hc["selection_n_init"], random_state=self.seed).fit(Zp)
            ref_lab = ref.labels_
            share = np.bincount(ref_lab, minlength=k) / len(ref_lab)
            aris = []
            for b in range(self.hc["stability_bootstraps"]):
                boot = rng.choice(uniq, len(uniq), replace=True)
                idx = np.concatenate([by_pid[p] for p in boot])
                km = KMeans(n_clusters=k, n_init=self.hc["selection_n_init"], random_state=self.seed + b + 1).fit(Zp[idx])
                aris.append(adjusted_rand_score(ref_lab, km.predict(Zp)))
            ss = min(self.hc["silhouette_sample"], len(Zp))
            sil = silhouette_score(Zp, ref_lab, sample_size=ss, random_state=self.seed) if k > 1 else np.nan
            rows.append({"K": k, "ARI_mean": np.mean(aris), "ARI_sd": np.std(aris), "silhouette": sil,
                         "inertia": ref.inertia_, "min_share": share.min()})
        sel = pd.DataFrame(rows)
        ok = sel[sel["min_share"] >= self.hc["min_habitat_share"]]
        if ok.empty:
            ok = sel
        best = ok["ARI_mean"].max()
        k = int(ok.loc[ok["ARI_mean"] >= best - self.hc["stability_tolerance"], "K"].min())
        return k, sel

    def predict(self, df):
        X = self._harmonise(df[self.cols_].to_numpy(float), df["cohort"].to_numpy(), fit=False)
        X = np.where(np.isfinite(X), X, 0.0)
        return self.km_.predict(self.pca_.transform(self.scaler_.transform(X)))


# ---------------------------------------------------------------- descriptive model (description only)
def descriptive_model(patches, evaluable_ids, patient_img, cfg, rep, pal):
    """Fit one habitat model on all habitat-evaluable patients for description only (never used to score anyone)."""
    rep.banner()
    pe = patches[patches["patient_id"].isin(evaluable_ids)].reset_index(drop=True).copy()
    t0 = time.time()
    desc = HabitatModel(cfg, seed=cfg["seed"]).fit(pe)
    pe = pe.assign(habitat=desc.predict(pe))
    rep.say(f"Descriptive model: {len(pe)} patches, {len(desc.cols_)} habitat features ({len(desc.dropped_)} dropped as "
            f"constant/non-finite), {desc.pca_.n_components_} PCs, K = {desc.k_} ({time.time() - t0:.0f} s)")
    if desc.selection_ is not None:
        rep.show(desc.selection_, "K selection (descriptive model):", round_to=3)
        rep.save_table(desc.selection_, "09_k_selection_descriptive")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        s = desc.selection_
        ax[0].errorbar(s["K"], s["ARI_mean"], yerr=s["ARI_sd"], fmt="o-", color=pal.series[0], capsize=3)
        ax[0].axvline(desc.k_, color="#8a8984", ls="--", lw=1)
        ax[0].set_xlabel("K")
        ax[0].set_ylabel("Bootstrap ARI (mean ± SD)")
        ax[0].set_title(f"Stability; selected K = {desc.k_}")
        ax[1].plot(s["K"], s["silhouette"], "o-", color=pal.series[2])
        ax[1].set_xlabel("K")
        ax[1].set_ylabel("Silhouette")
        ax[1].set_title("Silhouette (reported, not used to select)")
        fig.tight_layout()
        rep.save_fig(fig, "09_k_selection_descriptive")

    # Phenotypes from the EXACT mean-intensity columns (iteration 2 also matched MeanAbsoluteDeviation)
    mods = cfg["modalities"]
    mean_cols = [f"{m}__original_firstorder_Mean" for m in mods]
    assert all(c in pe.columns for c in mean_cols) and len(set(mean_cols)) == len(mods)
    rep.record_check("Phenotype columns are exact Mean features (no MeanAbsoluteDeviation)",
                     not any("MeanAbsoluteDeviation" in c for c in mean_cols), ", ".join(mean_cols))
    pheno = pe.groupby("habitat")[mean_cols].median()
    pheno.columns = mods
    pheno.insert(0, "patches", pe.groupby("habitat").size())
    pheno["median depth_norm (0 = margin)"] = pe.groupby("habitat")["depth_norm"].median()
    rep.show(pheno, "Habitat phenotypes (median patch mean intensity, brain z-score):", round_to=2)
    rep.save_table(pheno, "09_habitat_phenotypes_descriptive")
    fig, ax = plt.subplots(figsize=(5.5, 0.5 * desc.k_ + 1.5))
    vals = pheno[mods].to_numpy()
    lim = np.nanmax(np.abs(vals))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels(mods)
    ax.set_yticks(range(desc.k_))
    ax.set_yticklabels([f"H{k} (n={n})" for k, n in pheno["patches"].items()])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(vals[i, j]) > 0.6 * lim else "#0b0b0b")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="median patch mean intensity (brain z-score)")
    ax.set_title("Habitat phenotypes (descriptive model)")
    fig.tight_layout()
    rep.save_fig(fig, "09_habitat_phenotypes")

    # Do habitats separate cohorts? (a habitat that is one cohort's scanner signature cannot be biology)
    hx = pd.crosstab(pe["habitat"], pe["cohort"])
    v = cramers_v(hx.values) if hx.shape[1] > 1 else np.nan
    share = (hx["UCSF"] / hx.sum(axis=1)).round(2) if "UCSF" in hx else 0.0
    rep.show(hx.assign(**{"UCSF share": share}), f"Habitat x cohort (patches); Cramér's V = {v:.2f}")
    dom = pe.groupby("patient_id")["habitat"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    rep.say(f"Per-patient share of the dominant habitat: median {dom.median():.2f}; "
            f"{(dom >= 0.9).mean():.0%} of patients have >=90% of patches in one habitat.")
    summary = {"K": desc.k_, "n_patches": len(pe), "cramers_v_habitat_cohort": v, "dominant_share_median": dom.median()}

    # Section 10 descriptive check: are the spatial metrics confounded with size, patch count or composition?
    feats = build_patient_features(pe, desc.k_, cfg["seed"], cfg)
    feats = feats.join(patient_img[["tumour_volume_ml", "n_patches"]])
    feats["log_volume"] = np.log(feats["tumour_volume_ml"])
    rows = []
    for m in ["mixing_obs", "mixing_excess", "mixing_z", "radial_eta2_excess", "radial_eta2_z", "entropy_norm"]:
        for cov in ["log_volume", "n_patches", "n_habitats_present"]:
            d = feats[[m, cov]].dropna()
            if len(d) > 5 and d[m].std() > 0 and d[cov].std() > 0:
                rho, p = stats.spearmanr(d[m], d[cov])
                rows.append({"metric": m, "covariate": cov, "spearman_rho": rho, "p": p, "n": len(d)})
    conf = pd.DataFrame(rows)
    if len(conf):
        rep.show(conf.pivot(index="metric", columns="covariate", values="spearman_rho"),
                 "Spearman rho of spatial metrics with size / patch count / composition:", round_to=2)
        rep.save_table(conf, "10_spatial_metric_confounding")
    rep.say(f"Patients with spatial metrics defined: {int(feats['spatial_defined'].sum())}/{len(feats)}")
    return desc, pe, feats, summary


def exploratory_univariate(feats_desc, base, K, rep):
    """Section 13.6: exploratory univariate associations on the descriptive model (description, not evaluation)."""
    rep.banner()
    fd = feats_desc.join(base[["label", "cohort", "in_primary"]])
    fd = fd[fd["in_primary"].astype(bool)]
    rows = []
    for set_name, g in [("UTSW", fd[fd["cohort"].eq("UTSW")]), ("pooled", fd)]:
        for c in composition_cols(K) + ["mixing_obs", "mixing_excess", "radial_eta2_excess"] + [f"depth_shift_h{k}" for k in range(K)]:
            xp, xn = g.loc[g["label"] == 1, c].dropna(), g.loc[g["label"] == 0, c].dropna()
            if len(xp) > 2 and len(xn) > 2 and pd.concat([xp, xn]).nunique() > 1:
                r, p = rank_biserial(xp, xn)
                rows.append({"set": set_name, "feature": c, "n_codel": len(xp), "n_non": len(xn),
                             "Cliff's delta (>0: higher if codeleted)": r, "p": p,
                             "higher in": "codeleted" if r > 0 else ("non-codeleted" if r < 0 else "neither")})
    uni = pd.DataFrame(rows)
    if len(uni):
        uni["q (BH within set)"] = uni.groupby("set")["p"].transform(lambda s: bh_fdr(s.to_numpy()))
        rep.show(uni.sort_values(["set", "p"]), "Exploratory univariate associations (descriptive habitats):", round_to=4)
        rep.save_table(uni, "13_exploratory_univariate")
    return uni
