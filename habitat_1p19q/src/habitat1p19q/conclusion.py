"""Conclusion text (from executed results only, using the pre-specified decision rule) and reproducibility manifest."""
from __future__ import annotations

import json
import time

import pandas as pd

from .config import config_fingerprint
from .reporting import environment, jsonable
from .stats import fmt_ci


def conclusion_text(study, smoke):
    """study: the pipeline's Study object after a run."""
    L = []
    if smoke:
        L.append("**SYNTHETIC SMOKE TEST — these lines describe fake data and verify only that the code runs.**")
    eligible, R = study.cohort.eligible, study.results
    sa = study.audit["site_assoc"]["cohort_only_auc"]
    L.append(f"- Eligible IDH-mutant patients with a resolved 1p/19q label: {len(eligible)} "
             f"(UTSW {int((eligible.cohort == 'UTSW').sum())}, UCSF {int((eligible.cohort == 'UCSF').sum())}); "
             f"codeleted {int(eligible.label.sum())}.")
    L.append(f"- Cohort membership alone gives a pooled AUROC of {fmt_ci(sa['auc'], sa['lo'], sa['hi'])} "
             f"(UTSW vs UCSF odds ratio for codeletion {study.audit['site_assoc']['odds_ratio_UTSW_vs_UCSF']:.2f}).")
    if study.imaging_available:
        L.append(f"- Habitat-evaluable patients (all sets, incl. label-sensitivity extras): {len(study.evaluable_ids)}.")
    s = R["summary_table"]
    for _, r in s.iterrows():
        L.append(f"- {r['set']} · {r['model']} (n={r['n']}, {r['n_codeleted']} codel.): AUROC {r['AUROC (repeat-avg OOF, 95% CI)']}.")
    prim = R.get("primary")
    pset = study.cfg["primary"]["analysis_set"]
    if prim is None:
        L.append("- **Primary hypothesis H1 (M4 vs M3) was NOT TESTED in this run**: habitat features require the MRI "
                 "volumes and segmentations, which were not available to this execution.")
    else:
        sn = R.get("spatial_null") or {}
        supported = prim["_nb_lo"] > 0 and sn.get("p", 1) < 0.05
        L.append(f"- Primary ΔAUROC (M4 − M3, {pset}): {prim['ΔAUROC (fold mean)']:.3f} "
                 f"(Nadeau–Bengio 95% CI {prim['NB 95% CI']}); spatial-arrangement null p = {sn.get('p', float('nan')):.3f}.")
        L.append(f"- By the pre-specified rule, H1 is **{'SUPPORTED' if supported else 'NOT SUPPORTED'}**.")
    comps = R.get("comparisons")
    if isinstance(comps, pd.DataFrame) and len(comps):
        for _, r in comps.iterrows():
            L.append(f"- Secondary {r['set']}: {r['comparison']} = {r['ΔAUROC (fold mean)']:.3f} "
                     f"(NB 95% CI {r['NB 95% CI']}; BH-adjusted p = {r['NB p (BH-adjusted)']:.3f}).")
    lo = R.get("loco")
    if lo is not None and len(lo):
        for _, r in lo.iterrows():
            if r["status"] == "run":
                L.append(f"- Leave-one-cohort-out {r['train']} → {r['test']}, {r['model']}: AUROC {r['AUROC (95% CI)']}.")
            else:
                L.append(f"- Leave-one-cohort-out {r['train']} → {r['test']}: {r['status']}.")
    for n in R["notes"]:
        L.append(f"- Note: {n}")
    return "\n".join(L)


def write_manifest(study, out):
    imaging = study.imaging_available
    manual = [
        ("Primary comparison pre-specified before imaging results", "PASS (decided from metadata only)"),
        ("Habitat model refitted inside every training fold", "PASS (asserted per fold)" if imaging else "NOT RUN (no imaging)"),
        ("Shape features excluded from habitat clustering", "PASS (habitat_columns asserts)" if imaging else "NOT RUN (no imaging)"),
        ("Spatial metrics composition-adjusted (within-patient permutation null)", "PASS (by construction)" if imaging else "NOT RUN (no imaging)"),
        ("Thresholds chosen on training folds only", "PASS (train_only_threshold)"),
        ("Label-permutation test reruns the full learning procedure on label-independent folds", "PASS"),
        ("Leave-one-cohort-out run only when training events are sufficient", "PASS"),
        ("External untouched cohort", "NOT AVAILABLE"),
    ]
    checks = study.rep.checks
    checks_df = pd.concat([pd.DataFrame(checks).T.rename(columns={"status": "result"}),
                           pd.DataFrame(manual, columns=["check", "result"]).set_index("check")])
    study.rep.save_table(checks_df, "18_checklist")
    manifest = {
        "package": "habitat1p19q", "source_notebook": "Habitat_1p19q_IDHmut_v3.ipynb",
        "run_finished": time.strftime("%Y-%m-%d %H:%M:%S"), "imaging_mode": study.cfg["imaging_mode"],
        "environment": environment(), "config": study.cfg, "input_sha256": study.cohort.input_hashes,
        "patient_flow_metadata": study.cohort.flow.to_dict(orient="records"),
        "original_cohort_hypothesis": study.cohort.original_cohort_hypothesis,
        "imaging_available": bool(imaging), "n_habitat_evaluable": len(study.evaluable_ids),
        "extraction_fingerprint": config_fingerprint(study.cfg), "checks": checks,
        "outputs": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and "per_patient" not in str(p)
                          and "smoke_test_data" not in str(p)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=jsonable), encoding="utf-8")
    return checks_df
