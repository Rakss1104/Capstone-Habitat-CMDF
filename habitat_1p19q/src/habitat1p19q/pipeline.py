"""End-to-end study pipeline. Runs the notebook's sections in order; each stage is a method on `Study`."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import stats as st
from .audit import run_audit
from .cohort import load_cohort
from .conclusion import conclusion_text, write_manifest
from .config import is_smoke, prepare_output_dirs
from .figures import forest_figure, null_figure, roc_figure
from .habitat import descriptive_model, exploratory_univariate
from .imaging.extraction import exclusion_accounting, extract_all
from .imaging.qc import run_qc, select_study_patients
from .modeling import FoldFeatureBuilder, build_base_table
from .null_tests import run_label_permutations, spatial_arrangement_null
from .plotting import Palette, apply_style
from .reporting import Reporter, environment
from .robustness import calibration, grouped_importance_m4, robustness_primary_delta
from .spatial import toy_grid_self_test
from .transport import run_loco
from .validation import paired_comparison, run_cv, summarise_cv, within_cohort_discrimination

SECONDARY = [("UTSW", "M3", "M1"), ("UTSW", "M3", "M2"), ("UTSW", "M5", "M1"), ("UTSW", "M4", "M2"),
             ("pooled", "M4", "M3"), ("pooled", "M3", "M0"), ("pooled", "M4+cohort", "M3+cohort"),
             ("pooled", "M5+cohort", "M1"), ("pooled_meta", "M1_clin", "M0")]


class Study:
    def __init__(self, cfg):
        self.cfg = cfg
        self.smoke = is_smoke(cfg)
        self.out = prepare_output_dirs(cfg)
        self.rep = Reporter(self.out, smoke=self.smoke)
        self.pal = Palette(cfg)
        apply_style()
        st.set_default_n_boot(cfg["n_bootstrap"])
        np.random.seed(cfg["seed"])
        self.results = {"cv": {}, "comparisons": [], "loco": [], "nulls": {}, "notes": []}
        self.patient_img, self.patches = pd.DataFrame(), pd.DataFrame()
        self.evaluable_ids, self.desc = [], None
        self.timings = {}

    # ------------------------------------------------------------------ stages
    def _stage(self, name, fn):
        self.rep.section(name)
        t0 = time.time()
        out = fn()
        self.timings[name] = round(time.time() - t0, 1)
        self.rep.say(f"[{name}: {self.timings[name]:.0f} s]")
        return out

    def run(self):
        r = self.rep
        r.say(json.dumps(environment(), indent=1))
        r.say(f"Imaging mode: {self.cfg['imaging_mode']} | data root: {self.cfg['data_root'].resolve()} | outputs: {self.out.resolve()}")
        r.banner()
        self._stage("4-5 Cohort ingestion, integrity checks, eligibility", self.stage_cohort)
        self._stage("6 Cohort audit", self.stage_audit)
        self._stage("7 Imaging quality control", self.stage_qc)
        self._stage("8 Patch construction and feature extraction", self.stage_extraction)
        self._stage("9-10 Descriptive habitat model and spatial-metric checks", self.stage_descriptive)
        self._stage("12 Leakage-free cross-validation", self.stage_cv)
        self._stage("13 Primary and secondary analyses", self.stage_analyses)
        self._stage("13.4-13.5 Null tests", self.stage_nulls)
        self._stage("14 Cross-cohort generalisation", self.stage_loco)
        self._stage("15 Calibration, importance, robustness", self.stage_robustness)
        self._stage("16-18 Figures, conclusion, manifest", self.stage_report)
        return self

    def stage_cohort(self):
        self.cohort = load_cohort(self.cfg, self.rep)
        table, ok = toy_grid_self_test(self.cfg["patch_stride"], self.cfg["spatial"]["min_patches"])
        self.rep.show(table, "Spatial-metric unit test on toy patch grids:", round_to=3)
        self.rep.record_check("Spatial metrics behave as expected on toy grids (composition-equal arrangements separated)", ok,
                              "segregated mixing_z < -3, checkerboard > +3, random within ±3, radial eta2_z > +3, absent habitat shift = 0")
        assert ok, "spatial metric unit test failed"

    def stage_audit(self):
        self.audit = run_audit(self.cohort, self.rep, self.pal)

    def stage_qc(self):
        self.study, self.imaging_root = select_study_patients(self.cfg, self.cohort, self.rep)
        self.qc, self.imaging_available = run_qc(self.study, self.cfg, self.imaging_root, self.rep)

    def stage_extraction(self):
        if not self.imaging_available:
            self.rep.say("NOT EVALUATED: no readable imaging. Patch extraction skipped.")
            return
        self.patient_img, self.patches = extract_all(self.study, self.qc, self.cfg, self.rep)
        if len(self.patches):
            self.evaluable_ids = exclusion_accounting(self.study, self.qc, self.patient_img, self.patches, self.cfg, self.rep)
        else:
            self.rep.say("NOT EVALUATED: no patch features.")

    def stage_descriptive(self):
        if len(self.evaluable_ids) < 10:
            self.rep.say("NOT EVALUATED: fewer than 10 habitat-evaluable patients.")
            self.desc_summary = None
            return
        self.desc, self.desc_patches, self.feats_desc, self.desc_summary = descriptive_model(
            self.patches, self.evaluable_ids, self.patient_img, self.cfg, self.rep, self.pal)

    def stage_cv(self):
        cfg, R = self.cfg, self.results
        self.base = build_base_table(self.study, self.patient_img, self.imaging_available)
        self.rep.say(f"Base table: {self.base.shape}")
        self.builder_none = FoldFeatureBuilder(self.base, cfg)
        self.builder_img = FoldFeatureBuilder(self.base, cfg, self.patches if self.imaging_available else None,
                                              harmonisation=cfg["harmonisation"])
        ev = set(self.evaluable_ids)

        def ids_for(cohorts, require_imaging, primary_only=True):
            b = self.base
            pool = b.index[b["cohort"].isin(cohorts) & (b["in_primary"] if primary_only else True)]
            return [p for p in pool if p in ev] if require_imaging else list(pool)

        self.analysis_sets = {"UTSW_meta": (ids_for(["UTSW"], False), ["M1_clin"], self.builder_none),
                              "pooled_meta": (ids_for(["UTSW", "UCSF"], False), ["M0", "M1_clin"], self.builder_none)}
        if self.imaging_available and len(self.evaluable_ids) >= 20:
            self.analysis_sets["UTSW"] = (ids_for(["UTSW"], True), ["M1_clin", "M1", "M2", "M3", "M4", "M5"], self.builder_img)
            self.analysis_sets["pooled"] = (ids_for(["UTSW", "UCSF"], True),
                                            ["M0", "M1_clin", "M1", "M2", "M3", "M4", "M5", "M3+cohort", "M4+cohort", "M5+cohort"],
                                            self.builder_img)
        else:
            R["notes"].append("Imaging-based models (M1 with volume, M2–M5) NOT EVALUATED: no habitat-evaluable imaging.")
            self.rep.say("NOT EVALUATED: " + R["notes"][-1])
        for name, (ids, models, b) in self.analysis_sets.items():
            yy = self.base.loc[ids, "label"]
            self.rep.say(f"\n=== {name}: {len(ids)} patients ({int(yy.sum())} codeleted) | models {models}")
            if b is self.builder_img:
                self.rep.banner()
            res = run_cv(ids, models, b, keep_models_rep0=(name == cfg["primary"]["analysis_set"]), log=self.rep.say)
            R["cv"][name] = res
            self.rep.record_check(f"[{name}] train/test disjoint and habitat fit on training patients only (all folds)", True,
                                  f"{res.n_repeats * res.n_splits} folds asserted")
            self.rep.save_table(res.records, f"16_fold_records_{name}")

    def stage_analyses(self):
        R, P, rep = self.results, self.cfg["primary"], self.rep
        rep.banner()
        summary = pd.concat([summarise_cv(res, name) for name, res in R["cv"].items()], ignore_index=True)
        R["summary_table"] = summary
        show_cols = [c for c in summary.columns if not c.startswith("_")]
        rep.show(summary[show_cols], "Discrimination summary:", round_to=3)
        rep.save_table(summary[show_cols], "13_discrimination_summary")
        per_cohort = within_cohort_discrimination(R["cv"], self.base)
        if len(per_cohort):
            rep.show(per_cohort, "Discrimination within each cohort, from the pooled cross-validation:")
            rep.save_table(per_cohort, "13_within_cohort_discrimination")

        prim = None
        if P["analysis_set"] in R["cv"]:
            prim = paired_comparison(R["cv"][P["analysis_set"]], P["test"], P["reference"], P["analysis_set"])
        if prim is None:
            msg = (f"PRIMARY COMPARISON NOT EVALUATED: {P['test']} vs {P['reference']} in the {P['analysis_set']} set needs "
                   "habitat features, which require the MRI volumes and segmentations.")
            rep.say(msg)
            R["notes"].append(msg)
        else:
            rep.banner()
            rep.show(pd.DataFrame([prim]).drop(columns=[c for c in prim if c.startswith("_")]),
                     "PRE-SPECIFIED PRIMARY COMPARISON:", round_to=4)
            rep.save_table(pd.DataFrame([prim]), "13_primary_comparison")
        R["primary"] = prim

        sec = [paired_comparison(R["cv"][s], t, r, s) for s, t, r in SECONDARY if s in R["cv"]]
        sec = [x for x in sec if x is not None]
        if sec:
            sec_df = pd.DataFrame(sec)
            sec_df["NB p (BH-adjusted)"] = st.bh_fdr(sec_df["NB p"].fillna(1).to_numpy())
            rep.show(sec_df.drop(columns=[c for c in sec_df if c.startswith("_")]),
                     "Secondary comparisons (Benjamini–Hochberg across those evaluated):", round_to=4)
            rep.save_table(sec_df, "13_secondary_comparisons")
            R["comparisons"] = sec_df
        not_eval = [f"{s}: {t} vs {r}" for s, t, r in SECONDARY
                    if s not in R["cv"] or t not in R["cv"][s].oof or r not in R["cv"][s].oof]
        if not_eval:
            rep.say("Secondary comparisons NOT EVALUATED (features unavailable): " + "; ".join(not_eval))

        if self.desc is not None:
            exploratory_univariate(self.feats_desc, self.base, self.desc.k_, rep)

    def stage_nulls(self):
        R, P = self.results, self.cfg["primary"]
        plan = []
        if P["analysis_set"] in R["cv"]:
            plan.append((P["analysis_set"], ["M3", "M4"]))
        plan += [("UTSW_meta", ["M1_clin"]), ("pooled_meta", ["M0", "M1_clin"])]
        R["label_permutation"] = run_label_permutations(plan, self.analysis_sets, self.cfg["n_label_permutations"],
                                                        R["nulls"], self.rep)
        R["spatial_null"] = None
        if P["analysis_set"] in R["cv"] and "M4" in R["cv"][P["analysis_set"]].oof:
            ids, _, b = self.analysis_sets[P["analysis_set"]]
            R["spatial_null"], nd = spatial_arrangement_null(ids, b, R["cv"][P["analysis_set"]], self.cfg["n_spatial_null"], self.rep)
            R["nulls"]["spatial"] = nd
            self.rep.save_table(pd.DataFrame([R["spatial_null"]]), "13_spatial_arrangement_null")
        else:
            self.rep.say("NOT EVALUATED: spatial-arrangement null needs habitat features.")

    def stage_loco(self):
        if "pooled" in self.analysis_sets:
            b, models, pool = self.builder_img, ["M1_clin", "M1", "M2", "M3", "M4", "M5"], self.analysis_sets["pooled"][0]
            self.rep.banner()
        else:
            b, models, pool = self.builder_none, ["M1_clin"], self.analysis_sets["pooled_meta"][0]
        rows = []
        for tr_c, te_c in [("UTSW", "UCSF"), ("UCSF", "UTSW")]:
            rows += run_loco(tr_c, te_c, pool, models, b)
        loco = pd.DataFrame(rows)
        self.rep.show(loco, "Leave-one-cohort-out:")
        self.rep.save_table(loco, "14_leave_one_cohort_out")
        self.results["loco"] = loco
        if "pooled" in self.results["cv"]:
            fi = pd.DataFrame(self.results["cv"]["pooled"].fold_info)
            if "cramers_v_train" in fi:
                self.rep.say(f"Habitat × cohort association in pooled training folds: Cramér's V median "
                             f"{fi['cramers_v_train'].median():.2f} [{fi['cramers_v_train'].min():.2f}–{fi['cramers_v_train'].max():.2f}]. "
                             "Values near 1 mean habitats are cohort signatures.")

    def stage_robustness(self):
        R, P, rep = self.results, self.cfg["primary"], self.rep
        cal_set = P["analysis_set"] if P["analysis_set"] in R["cv"] else "UTSW_meta"
        cal_df, fig = calibration(R["cv"][cal_set], cal_set, self.pal)
        rep.save_fig(fig, f"15_calibration_{cal_set}")
        rep.show(cal_df, "Calibration:", round_to=3)
        rep.save_table(cal_df, "15_calibration")
        R["calibration"] = cal_df
        if P["analysis_set"] in R["cv"] and R["cv"][P["analysis_set"]].models_rep0:
            rep.banner()
            gi = grouped_importance_m4(R["cv"][P["analysis_set"]], self.cfg["seed"])
            rep.show(gi, "Grouped permutation importance (M4, first-repeat test folds):", round_to=4)
            rep.save_table(gi, "15_grouped_permutation_importance_M4")
            rob = robustness_primary_delta(P["analysis_set"], self.analysis_sets, self.builder_img, self.patches,
                                           self.cohort.label_sets, self.evaluable_ids, self.cfg, rep)
            rep.show(rob, "Robustness of the primary Δ (M4 − M3, first-repeat folds):", round_to=4)
            rep.save_table(rob, "15_robustness_primary_delta")
            R["robustness"] = rob
        else:
            rep.say("NOT EVALUATED: grouped importance and robustness need the primary M4 model.")

    def stage_report(self):
        R, rep = self.results, self.rep
        rep.save_fig(roc_figure(R["cv"], self.pal, self.smoke), "16_roc_curves")
        rep.save_fig(forest_figure(R["summary_table"], self.audit["site_assoc"], "pooled_meta" in R["cv"], self.pal, self.smoke),
                     "16_auroc_forest")
        if R["nulls"]:
            rep.save_fig(null_figure(R["nulls"], R["label_permutation"], R.get("spatial_null"), self.pal), "16_null_distributions")
        self.conclusion = conclusion_text(self, self.smoke)
        (self.out / "conclusion.md").write_text(self.conclusion, encoding="utf-8")
        rep.say("CONCLUSION\n" + self.conclusion)
        checks_df = write_manifest(self, self.out)
        rep.show(checks_df, "Study checklist:")
        (self.out / "timings.json").write_text(json.dumps(self.timings, indent=1), encoding="utf-8")
        counts = pd.Series([v["status"] for v in rep.checks.values()]).value_counts().to_dict()
        rep.say(f"Manifest written to {self.out / 'manifest.json'} | automated checks: {counts}")
        rep.banner()


def run_study(cfg):
    return Study(cfg).run()
