"""End-to-end CMDF study (PROTOCOL.md). Each stage is a method; `run()` executes them in order."""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from habitat1p19q import stats as st
from habitat1p19q.cohort import load_cohort
from habitat1p19q.config import config_fingerprint
from habitat1p19q.imaging.extraction import exclusion_accounting, extract_all
from habitat1p19q.imaging.qc import run_qc, select_study_patients
from habitat1p19q.modeling import build_base_table
from habitat1p19q.reporting import Reporter, environment, jsonable, sha256
from habitat1p19q.validation import paired_comparison, summarise_cv, within_cohort_discrimination

from . import figures as fg
from .config import CMDF_ROOT, PAIRS
from .cv import label_permutation, pairing_null, run_cv, run_loco
from .features import CMDFBuilder, model_columns, patient_arrays, texture_names
from .naive import run_naive_arm
from .voxel import extract_voxel_all

UTSW_MODELS = ["M1_clin", "M1", "M2", "P", "D", "P+D", "D_T2FL", "V", "MM", "M2+V"]
POOLED_MODELS = ["M0", "M1_clin", "M2", "P", "D", "P+D", "V", "P+cohort", "P+D+cohort", "V+cohort", "M2+cohort"]
PRIMARY = ("UTSW", "P+D", "P")
SECONDARY = [("S1", "UTSW", "D", "P"), ("S2", "UTSW", "P+D", "M2"), ("S3", "UTSW", "D_T2FL", "M1_clin"),
             ("S4", "UTSW", "V", "M1_clin"), ("S5", "UTSW", "M2+V", "M2"), ("S6", "pooled", "P+D+cohort", "P+cohort"),
             ("S7", "pooled", "V+cohort", "M0")]
LOCO_MODELS = ["M1_clin", "M2", "P", "D", "P+D", "V"]
EXPECTED_BLOCKS = {"P": 744, "D": 1128, "D_T2FL": 188, "V": 45, "MM": 3}
EXPECTED_SETS = {"UTSW": (74, 36), "UCSF": (76, 8)}


class CMDFStudy:
    def __init__(self, hcfg, ccfg):
        self.hcfg, self.ccfg = hcfg, ccfg
        self.out = ccfg["output_dir"]
        for sub in ["tables", "figures", "features", "cache"]:
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.rep = Reporter(self.out)
        st.set_default_n_boot(hcfg["n_bootstrap"])
        np.random.seed(hcfg["seed"])
        self.R = {"cv": {}, "nulls": {}, "notes": [], "deviations": [
            "After PROTOCOL.md was fixed and before the full run, a one-repeat timing probe on UTSW printed fold-mean AUROCs "
            "for P (0.657) and P+D (0.601). Nothing in the protocol, features or models was changed afterwards."]}
        self.timings = {}
        self.nj = ccfg["cv_n_jobs"]

    def _stage(self, name, fn):
        self.rep.section(name)
        t0 = time.time()
        fn()
        self.timings[name] = round(time.time() - t0, 1)
        self.rep.say(f"[{name}: {self.timings[name]:.0f} s]")
        self.done = name.split()[0]
        self.save_state()

    # Checkpoint after every stage so an interrupted run can resume (--resume) instead of repeating finished stages.
    def _state_path(self):
        return self.out / "cache" / "state.pkl"

    def save_state(self):
        skip = {"hcfg", "ccfg", "rep", "out"}
        state = {k: v for k, v in self.__dict__.items() if k not in skip}
        state["checks"] = dict(self.rep.checks)
        tmp = self._state_path().with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(state))
        tmp.replace(self._state_path())

    def load_state(self):
        state = pickle.loads(self._state_path().read_bytes())
        if state.get("protocol_sha") != self.protocol_sha:
            raise RuntimeError("checkpoint was written under a different PROTOCOL.md; rerun without --resume")
        for name, c in state.pop("checks").items():
            self.rep.checks[name] = c
        self.__dict__.update(state)
        self.rep.say(f"Resumed from checkpoint after stage {self.done} ({self._state_path()})")

    def run(self, stages=None, resume=False):
        plan = [("1 Cohort, imaging QC and cached patch features", self.stage_data),
                ("2 Voxel-level CMDF and T2-FLAIR mismatch", self.stage_voxel),
                ("3 Feature blocks", self.stage_blocks),
                ("4 Leakage-free cross-validation", self.stage_cv),
                ("5 Primary and secondary analyses", self.stage_analyses),
                ("6 Null tests", self.stage_nulls),
                ("7 Cross-cohort transport", self.stage_loco),
                ("8 Robustness and calibration", self.stage_robustness),
                ("9 Methodological arm: original notebook protocol", self.stage_naive),
                ("10 Figures, conclusion, manifest", self.stage_report)]
        self.rep.say(json.dumps(environment(), indent=1))
        self.rep.say(f"data root: {self.hcfg['data_root']} | habitat cache: {self.hcfg['output_dir']} | outputs: {self.out}")
        self.protocol_sha = sha256(CMDF_ROOT / "PROTOCOL.md")
        self.rep.say(f"PROTOCOL.md SHA-256: {self.protocol_sha}")
        done = 0
        if resume and self._state_path().is_file():
            self.load_state()
            done = int(self.done)
        for name, fn in plan:
            n = name.split()[0]
            if int(n) > done and (stages is None or n in stages):
                self._stage(name, fn)
        return self

    # ------------------------------------------------------------------------------------------------ data
    def stage_data(self):
        cache = self.out / "cache" / "data.pkl"
        key = {"fingerprint": config_fingerprint(self.hcfg),
               "meta": [sha256(self.hcfg["data_root"] / self.hcfg[k]) for k in ["utsw_metadata", "ucsf_metadata"]]}
        if cache.is_file():
            d = pickle.loads(cache.read_bytes())
            if d["key"] == key:
                self.rep.say(f"Reusing cohort/QC/feature tables from {cache} (same metadata hashes and feature fingerprint)")
                self.__dict__.update(d["data"])
                for name, c in d.get("checks", {}).items():
                    self.rep.record_check(name, c["status"] == "PASS", c["detail"], status=c["status"])
                return
        rep, h = self.rep, self.hcfg
        cohort = load_cohort(h, rep)
        study, root = select_study_patients(h, cohort, rep)
        qc, avail = run_qc(study, h, root, rep)
        assert avail, "no readable imaging"
        cdir = h["output_dir"] / "features" / "per_patient"
        ok = qc.loc[qc["qc_status"].eq("ok"), "patient_id"]
        fps = [json.loads((cdir / f"{p}.json").read_text()).get("fingerprint") if (cdir / f"{p}.json").is_file() else None for p in ok]
        n_hit = sum(f == key["fingerprint"] for f in fps)
        rep.record_check("Patch features reused from the habitat study cache", n_hit == len(fps),
                         f"{n_hit}/{len(fps)} QC-passing patients cached with fingerprint {key['fingerprint']}")
        patient_img, patches = extract_all(study, qc, h, rep)
        evaluable = exclusion_accounting(study, qc, patient_img, patches, h, rep)
        data = {"study": study, "qc": qc, "patient_img": patient_img, "patches": patches, "evaluable_ids": evaluable,
                "label_sets": cohort.label_sets, "input_hashes": cohort.input_hashes}
        cache.write_bytes(pickle.dumps({"key": key, "data": data, "checks": dict(rep.checks)}))
        self.__dict__.update(data)

    def _ids(self, cohorts):
        b, ev = self.base, set(self.evaluable_ids)
        pool = b.index[b["cohort"].isin(cohorts) & b["in_primary"]]
        return [p for p in pool if p in ev]

    def stage_voxel(self):
        self.base = build_base_table(self.study, self.patient_img, True)
        self.ids = {"UTSW": self._ids(["UTSW"]), "UCSF": self._ids(["UCSF"]), "pooled": self._ids(["UTSW", "UCSF"])}
        for c, (n, k) in EXPECTED_SETS.items():
            y = self.base.loc[self.ids[c], "label"]
            self.rep.record_check(f"{c} analysis set matches the protocol ({n} patients, {k} codeleted)",
                                  len(y) == n and int(y.sum()) == k, f"{len(y)} patients, {int(y.sum())} codeleted")
        self.V, self.voxel_status = extract_voxel_all(self.qc, self.ids["pooled"], self.hcfg, self.ccfg, self.rep)
        self.rep.save_table(self.voxel_status, "02_voxel_status")
        self.rep.show(self.base.loc[self.ids["pooled"], ["cohort", "label"]].join(self.V[["MM__mismatch_frac"]])
                      .groupby(["cohort", "label"])["MM__mismatch_frac"].describe()[["count", "25%", "50%", "75%"]],
                      "T2-FLAIR mismatch fraction by cohort and label:", round_to=3)

    def stage_blocks(self):
        rep = self.rep
        self.names = texture_names(self.patches)
        self.arrays = patient_arrays(self.patches, self.names, self.ids["pooled"])
        self.builder = CMDFBuilder(self.base, self.arrays, self.names, self.V)
        Ftr, _, _ = self.builder.build(self.ids["UTSW"][:10], self.ids["UTSW"][10:12])
        sizes = {g: len(model_columns(g, Ftr)) for g in EXPECTED_BLOCKS}
        rep.record_check("Feature block sizes match the protocol", sizes == EXPECTED_BLOCKS, json.dumps(sizes))
        n_p = self.patches[self.patches["patient_id"].isin(self.ids["pooled"])].groupby("patient_id").size()
        rep.say(f"Patches per patient: median {n_p.median():.0f} (IQR {n_p.quantile(.25):.0f}–{n_p.quantile(.75):.0f}), "
                f"min {n_p.min()}, {int((n_p < 2).sum())} patient(s) with a single patch (SD features imputed)")
        # pairing-null sanity: shuffling co-location leaves every sequence's own distribution unchanged
        pid = n_p.idxmax()
        from .features import pair_permuted
        X, Y = self.arrays[pid], pair_permuted(self.arrays[pid], 1, pid)
        same = all(np.allclose(np.sort(X[:, j], 0), np.sort(Y[:, j], 0), equal_nan=True) for j in range(X.shape[1]))
        moved = not np.allclose(X[:, 1:], Y[:, 1:], equal_nan=True)
        rep.record_check("Pairing null keeps each sequence's patch distribution and breaks co-location", same and moved,
                         f"checked on {pid} ({len(X)} patches)")

    # -------------------------------------------------------------------------------------------- analyses
    def stage_cv(self):
        for name, models in [("UTSW", UTSW_MODELS), ("pooled", POOLED_MODELS)]:
            ids = self.ids[name]
            self.rep.say(f"\n=== {name}: {len(ids)} patients ({int(self.base.loc[ids, 'label'].sum())} codeleted) | models {models}")
            res = run_cv(ids, models, self.builder, self.hcfg, n_jobs=self.nj, log=self.rep.say)
            self.R["cv"][name] = res
            self.rep.record_check(f"[{name}] train/test disjoint and cosine scaling fitted on training patients only (all folds)",
                                  True, f"{res.n_repeats * res.n_splits} folds asserted")
            self.rep.save_table(res.records, f"04_fold_records_{name}")

    def stage_analyses(self):
        R, rep = self.R, self.rep
        summary = pd.concat([summarise_cv(res, n) for n, res in R["cv"].items()], ignore_index=True)
        R["summary"] = summary
        rep.show(summary[[c for c in summary if not c.startswith("_")]], "Discrimination summary:", round_to=3)
        rep.save_table(summary[[c for c in summary if not c.startswith("_")]], "05_discrimination_summary")
        wc = within_cohort_discrimination({"pooled": R["cv"]["pooled"]}, self.base)
        rep.show(wc, "Discrimination within each cohort, from the pooled cross-validation:")
        rep.save_table(wc, "05_within_cohort_discrimination")

        s, t, r = PRIMARY
        prim = paired_comparison(R["cv"][s], t, r, s)
        R["primary"] = prim
        rep.show(pd.DataFrame([prim]).drop(columns=[c for c in prim if c.startswith("_")]), "PRE-SPECIFIED PRIMARY COMPARISON:", round_to=4)
        rep.save_table(pd.DataFrame([prim]), "05_primary_comparison")

        sec = []
        for sid, s_, t_, r_ in SECONDARY:
            c = paired_comparison(R["cv"][s_], t_, r_, s_)
            sec.append({"id": sid, **c})
        sec = pd.DataFrame(sec)
        sec["NB p (BH-adjusted)"] = st.bh_fdr(sec["NB p"].fillna(1).to_numpy())
        R["secondary"] = sec
        rep.show(sec.drop(columns=[c for c in sec if c.startswith("_")]), "Secondary comparisons (Benjamini–Hochberg, S1–S7):", round_to=4)
        rep.save_table(sec, "05_secondary_comparisons")

        rows = []
        for name in ["UTSW", "pooled"]:
            ids = self.ids[name]
            d = self.base.loc[ids, ["label"]].join(self.V)
            for f in self.V.columns:
                pos, neg = d.loc[d["label"] == 1, f].dropna(), d.loc[d["label"] == 0, f].dropna()
                delta, p = st.rank_biserial(pos, neg)
                lo, hi = st.rank_biserial_ci(pos, neg, seed=5) if f == "MM__mismatch_frac" else (np.nan, np.nan)
                rows.append({"set": name, "feature": f, "n_codel": len(pos), "n_non": len(neg),
                             "median codeleted": pos.median(), "median non-codeleted": neg.median(),
                             "Cliff's delta (>0: higher if codeleted)": delta, "95% CI": f"{lo:.3f} to {hi:.3f}" if np.isfinite(lo) else "",
                             "p": p})
        uni = pd.DataFrame(rows)
        uni["q (BH within set)"] = uni.groupby("set")["p"].transform(lambda p: st.bh_fdr(p.to_numpy()))
        R["voxel_univariate"] = uni
        h2 = uni[uni["feature"].eq("MM__mismatch_frac")]
        R["H2"] = h2
        rep.show(h2, "H2 (pre-specified): T2–FLAIR mismatch fraction, codeleted vs non-codeleted:", round_to=4)
        rep.show(uni.sort_values(["set", "p"]).groupby("set").head(8), "Voxel CMDF features, exploratory (top 8 per set by p):", round_to=4)
        rep.save_table(uni, "05_voxel_univariate")

    def stage_nulls(self):
        R, rep = self.R, self.rep
        s, t, r = PRIMARY
        R["pairing_null"], nd = pairing_null(self.ids[s], R["cv"][s], t, r, self.builder, self.hcfg,
                                             self.ccfg["pairing_null_reps"], self.ccfg["null_n_jobs"], rep)
        R["nulls"]["pairing"] = nd
        rep.save_table(pd.DataFrame([R["pairing_null"]]), "06_pairing_null")
        rows = label_permutation("UTSW", self.ids["UTSW"], ["P+D", "V"], self.builder, self.hcfg,
                                 self.ccfg["label_permutations"], self.ccfg["null_n_jobs"], R["nulls"])
        R["label_permutation"] = pd.DataFrame(rows)
        rep.show(R["label_permutation"], "Label-permutation test (within cohort, label-independent folds):", round_to=4)
        rep.save_table(R["label_permutation"], "06_label_permutation")
        rep.record_check("Permuted and observed runs share identical label-independent folds", True, "asserted")

    def stage_loco(self):
        rows = run_loco("UTSW", "UCSF", self.ids["pooled"], LOCO_MODELS, self.builder, self.hcfg, self.hcfg["min_train_events_loco"])
        rows += run_loco("UCSF", "UTSW", self.ids["pooled"], LOCO_MODELS, self.builder, self.hcfg, self.hcfg["min_train_events_loco"])
        loco = pd.DataFrame(rows)
        self.R["loco"] = loco
        self.rep.show(loco.drop(columns=[c for c in loco if c.startswith("_")]), "Leave-one-cohort-out:")
        self.rep.save_table(loco, "07_leave_one_cohort_out")

    def stage_robustness(self):
        R, rep, ids = self.R, self.rep, self.ids["UTSW"]
        rob = []
        rf = run_cv(ids, ["P", "P+D"], self.builder, self.hcfg, estimator="rf", n_jobs=self.nj, log=rep.say)
        rob.append({"variant": "random forest", **paired_comparison(rf, "P+D", "P", "UTSW")})
        n_p = self.patches.groupby("patient_id").size()
        ids4 = [p for p in ids if n_p.get(p, 0) >= self.ccfg["min_patches_robustness"]]
        r4 = run_cv(ids4, ["P", "P+D"], self.builder, self.hcfg, n_jobs=self.nj, log=rep.say)
        rob.append({"variant": f">= {self.ccfg['min_patches_robustness']} patches (n={len(ids4)}, "
                               f"{int(self.base.loc[ids4, 'label'].sum())} codel.)", **paired_comparison(r4, "P+D", "P", "UTSW")})
        rob = pd.DataFrame(rob)
        R["robustness"] = rob
        rep.show(rob.drop(columns=[c for c in rob if c.startswith("_")]), "Robustness of the primary Δ (P+D − P):", round_to=4)
        rep.save_table(rob, "08_robustness_primary")

        full = st.auc_with_ci(R["cv"]["UTSW"].y, np.nanmean(R["cv"]["UTSW"].oof["P+D"], 0), n_boot=200)["auc"]
        abl = []
        for a, b in PAIRS:
            res = run_cv(ids, ["P+D"], self.builder, self.hcfg, thresholds=False, n_jobs=self.nj, drop={"P+D": (f"D_{a}-{b}__",)})
            auc = st.auc_with_ci(res.y, np.nanmean(res.oof["P+D"], 0), n_boot=200)["auc"]
            abl.append({"variant": f"without {a}–{b}", "AUROC (repeat-avg OOF)": auc, "ΔAUROC vs full P+D": auc - full})
            rep.say(f"  ablation without {a}–{b}: AUROC {auc:.3f} (Δ {auc - full:+.3f})")
        R["ablation"] = pd.DataFrame(abl)
        rep.save_table(R["ablation"], "08_pair_ablation")

        cal = []
        res = R["cv"]["UTSW"]
        for m in res.oof:
            c = st.calibration_stats(res.y, np.nanmean(res.oof[m], 0))
            cal.append({"set": "UTSW", "model": m, **c})
        R["calibration"] = pd.DataFrame(cal)
        rep.show(R["calibration"], "Calibration (UTSW, repeat-averaged OOF):", round_to=3)
        rep.save_table(R["calibration"], "08_calibration")

    def stage_naive(self):
        ids = self.ids["UTSW"]
        F = self.builder.full_table(ids)
        cols = model_columns("P+D", F)
        y = self.base.loc[ids, "label"].to_numpy(int)
        table, summ, nulls = run_naive_arm(F, cols, y, self.ccfg, self.ccfg["null_n_jobs"], self.rep)
        rig = self.R["summary"].query("set == 'UTSW' and model == 'P+D'").iloc[0]
        summ["rigorous P+D AUROC (repeat-avg OOF)"] = rig["_auc"]
        summ["optimism (best − rigorous)"] = summ["best_auc"] - rig["_auc"]
        self.R["naive"], self.R["naive_table"] = summ, table
        self.R["nulls"].update({f"naive|{k}": v for k, v in nulls.items()})
        self.rep.show(table, "Original protocol, every classifier (UTSW, P+D features):", round_to=3)
        self.rep.show(summ, "Original protocol vs the leakage-free estimate:", round_to=3)
        self.rep.save_table(table, "09_original_protocol_classifiers")
        self.rep.save_table(summ, "09_original_protocol_summary")
        self.R["deviations"].append("Methodological arm: the protocol described one variant (selection on all patients); a second, "
                                    "'notebook' variant (selection inside each fold, exactly as pipeline_cmdf.py) was added so the "
                                    "original notebooks are reproduced faithfully. Their VAE oversampling, 1D-CNN and attention-MIL "
                                    "models were not replicated (TensorFlow is not installed).")

    # -------------------------------------------------------------------------------------------- reporting
    def stage_report(self):
        R, rep = self.R, self.rep
        rep.save_fig(fg.roc_figure(R["cv"]), "10_roc_curves")
        m0 = R["summary"].query("set == 'pooled' and model == 'M0'")["_auc"]
        rep.save_fig(fg.forest_figure(R["summary"], float(m0.iloc[0]) if len(m0) else None), "10_auroc_forest")
        panels = [("Pairing null: Δ(P+D − P), UTSW", R["nulls"]["pairing"], R["pairing_null"]["delta_obs_rep1"], "observed Δ")]
        for _, r in R["label_permutation"].iterrows():
            panels.append((f"Label permutation: {r['model']}, UTSW", R["nulls"][f"labelperm|UTSW|{r['model']}"],
                           r["observed AUROC (cohort-stratified folds)"], "observed AUROC"))
        for _, r in R["naive"].iterrows():
            panels.append((f"Original protocol ({r['variant']}) on permuted labels", R["nulls"][f"naive|{r['variant']}"],
                           r["best_auc"], "reported AUROC"))
        rep.save_fig(fg.null_figure(panels), "10_null_distributions")
        rep.save_fig(fg.mismatch_figure(self.base, self.V, {"UTSW": self.ids["UTSW"], "UCSF": self.ids["UCSF"]}), "10_t2flair_mismatch")
        rep.save_fig(fg.ablation_figure(R["ablation"]), "10_pair_ablation")

        from .conclusion import conclusion_text
        text = conclusion_text(self)
        (self.out / "conclusion.md").write_text(text, encoding="utf-8")
        rep.say("CONCLUSION\n" + text)
        manifest = {"protocol_sha256": self.protocol_sha, "input_hashes": self.input_hashes, "environment": environment(),
                    "habitat_feature_fingerprint": config_fingerprint(self.hcfg), "settings": {k: str(v) for k, v in self.ccfg.items()},
                    "cv": self.hcfg["cv"], "seed": self.hcfg["seed"], "checks": rep.checks, "timings": self.timings,
                    "tables": sorted(p.name for p in (self.out / "tables").glob("*.csv")),
                    "figures": sorted(p.name for p in (self.out / "figures").glob("*.png"))}
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=jsonable), encoding="utf-8")
        checks = pd.DataFrame(rep.checks).T
        rep.show(checks, "Study checklist:")
        counts = checks["status"].value_counts().to_dict()
        rep.say(f"Manifest written to {self.out / 'manifest.json'} | automated checks: {counts}")
