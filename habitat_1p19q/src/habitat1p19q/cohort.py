"""Cohort ingestion, patient-level integrity checks, eligibility and label provenance (notebook sections 4–5).

Both metadata files are read as text with no automatic missing-value conversion; every categorical field goes through
a strict map, so an unexpected value stops the run instead of silently becoming a default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .reporting import sha256

UTSW_REQUIRED = ["Subject ID", "Sex at birth", "Age at Imaging", "Tumor Type", "Tumor Grade", "IDH", "IDH_Testing Method",
                 "1p19Q CODEL", "Operation Status", "T1", "T1GD", "T2", "T2FLAIR", "Scanner Make", "Scanner Model",
                 "Scanner Strength", "Manually Refined Segmentation"]
UCSF_REQUIRED = ["ID", "Sex", "Age at MRI", "WHO CNS Grade", "Final pathologic diagnosis (WHO 2021)", "1p/19q", "IDH",
                 "Biopsy prior to imaging", "BraTS21 ID", "BraTS21 Segmentation Cohort"]

# Strict value maps. Anything not listed raises an error.
UTSW_IDH_MAP = {"mutated": "mutant", "wild type": "wildtype", "NA": "unknown"}
UTSW_CODEL_MAP = {"co-deleted": "codeleted", "non co-deleted": "non-codeleted", "NA": "unknown"}
UCSF_CODEL_MAP = {"co-deletion": "codeleted", "intact": "non-codeleted",
                  "relative co-deletion": "relative_codeletion", "unknown": "unknown"}
SEX_MAP = {"M": "M", "F": "F"}
YESNO_MAP = {"Yes": True, "No": False}
UCSF_ID_RE = re.compile(r"^UCSF-PDGM-(\d{3,4})(?:_FU(\d+)d)?$")

CLASS_NAME = {0: "1p/19q non-codeleted", 1: "1p/19q codeleted"}


@dataclass
class Cohort:
    """Everything derived from the two metadata files."""
    utsw_raw: pd.DataFrame
    ucsf_raw: pd.DataFrame
    all_rows: pd.DataFrame            # every metadata row, harmonised (incl. UCSF follow-ups)
    patients: pd.DataFrame            # one row per patient, with label and flags
    eligible: pd.DataFrame            # IDH-mutant, 1p/19q resolved (primary label)
    sens: pd.DataFrame                # IDH-mutant, with WHO-2021-resolved label for the sensitivity analysis
    flow: pd.DataFrame
    flag_summary: pd.DataFrame
    followups: pd.DataFrame
    missingness: pd.DataFrame
    label_sets: dict
    input_hashes: dict
    original_cohort_hypothesis: dict = field(default_factory=dict)


# ---------------------------------------------------------------- parsing helpers
def strict_map(series, mapping, field_name):
    values = series.astype(str).str.strip()
    unknown = sorted(set(values) - set(mapping))
    if unknown:
        raise ValueError(f"{field_name}: unmapped values {unknown}. Add them to the explicit map after checking their meaning.")
    return values.map(mapping)


def map_ucsf_idh(value):
    v = str(value).strip()
    if v.lower() == "wildtype":
        return "wildtype"
    if re.match(r"^(IDH[12] p\.\S+|mutated \(NOS\))$", v):
        return "mutant"
    if v.lower() in {"unknown", ""}:
        return "unknown"
    raise ValueError(f"UCSF IDH: unmapped value {v!r}")


def to_float(series, missing=("NA", "Not Reported", "unknown", "")):
    s = series.astype(str).str.strip()
    return pd.to_numeric(s.where(~s.isin(missing)), errors="raise")


def read_text_table(path, sep):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}. Set GLIOMA_DATA_ROOT / --data-root.")
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, na_filter=False)


def load_utsw(path):
    raw = read_text_table(path, "\t")
    missing = [c for c in UTSW_REQUIRED if c not in raw.columns]
    assert not missing, f"UTSW metadata missing columns: {missing}"
    ids = raw["Subject ID"].str.strip()
    assert ids.str.match(r"^BT\d{4}$").all(), "Unexpected UTSW Subject ID format"
    out = pd.DataFrame({
        "cohort": "UTSW",
        "source_id": ids,
        "patient_id": "UTSW-" + ids,
        "is_followup": False,
        "age": to_float(raw["Age at Imaging"]),
        "sex": strict_map(raw["Sex at birth"], SEX_MAP, "UTSW sex"),
        "grade": to_float(raw["Tumor Grade"]),
        "diagnosis": raw["Tumor Type"].str.strip(),
        "diagnosis_system": "histology (mixed eras)",
        "idh_raw": raw["IDH"].str.strip(),
        "idh_status": strict_map(raw["IDH"], UTSW_IDH_MAP, "UTSW IDH"),
        "idh_method": raw["IDH_Testing Method"].str.strip().replace({"NA": np.nan}),
        "codel_raw": raw["1p19Q CODEL"].str.strip(),
        "codel_status": strict_map(raw["1p19Q CODEL"], UTSW_CODEL_MAP, "UTSW 1p/19q"),
        "operation_status": raw["Operation Status"].str.strip(),
        "modalities_listed": (raw[["T1", "T1GD", "T2", "T2FLAIR"]] == "Available").all(axis=1),
        "scanner_make": raw["Scanner Make"].str.strip().replace({"Not Reported": np.nan}),
        "scanner_model": raw["Scanner Model"].str.strip().replace({"Not Reported": np.nan}),
        "field_strength_T": to_float(raw["Scanner Strength"]).round(2),
        "seg_provenance": np.where(strict_map(raw["Manually Refined Segmentation"], YESNO_MAP, "UTSW seg").astype(bool),
                                   "FeTS + manual refinement", "FeTS automatic"),
        "biopsy_prior_to_imaging": np.nan,
        "brats21_id": np.nan,
    })
    return raw, out


def load_ucsf(path):
    raw = read_text_table(path, ",")
    missing = [c for c in UCSF_REQUIRED if c not in raw.columns]
    assert not missing, f"UCSF metadata missing columns: {missing}"
    parsed = raw["ID"].str.strip().str.extract(UCSF_ID_RE)
    bad = raw.loc[parsed[0].isna(), "ID"].tolist()
    assert not bad, f"Unparseable UCSF IDs: {bad}"
    num = parsed[0].astype(int)
    out = pd.DataFrame({
        "cohort": "UCSF",
        "source_id": raw["ID"].str.strip(),
        "patient_id": "UCSF-PDGM-" + num.map(lambda n: f"{n:04d}"),
        "is_followup": parsed[1].notna(),
        "followup_days": pd.to_numeric(parsed[1]),
        "age": to_float(raw["Age at MRI"]),
        "sex": strict_map(raw["Sex"], SEX_MAP, "UCSF sex"),
        "grade": to_float(raw["WHO CNS Grade"]),
        "diagnosis": raw["Final pathologic diagnosis (WHO 2021)"].str.strip(),
        "diagnosis_system": "WHO 2021 integrated",
        "idh_raw": raw["IDH"].str.strip(),
        "idh_status": raw["IDH"].map(map_ucsf_idh),
        "idh_method": np.nan,
        "codel_raw": raw["1p/19q"].str.strip(),
        "codel_status": strict_map(raw["1p/19q"], UCSF_CODEL_MAP, "UCSF 1p/19q"),
        "operation_status": "PRE (dataset: preoperative)",
        "modalities_listed": True,   # UCSF-PDGM distributes all four structural contrasts; verified on disk in QC
        "scanner_make": np.nan, "scanner_model": np.nan,
        "field_strength_T": np.nan,  # not per-patient in metadata; dataset documentation: standardised 3 T protocol
        "seg_provenance": "BraTS21 ensemble + manual correction",
        "biopsy_prior_to_imaging": strict_map(raw["Biopsy prior to imaging"], YESNO_MAP, "UCSF biopsy").astype(bool),
        "brats21_id": raw["BraTS21 ID"].str.strip().replace({"": np.nan}),
    })
    return raw, out


def label_from_status(s):
    return {"codeleted": 1.0, "non-codeleted": 0.0}.get(s, np.nan)


# ---------------------------------------------------------------- integrity checks (section 4)
def _followup_table(ucsf):
    fu = ucsf[ucsf["is_followup"]]
    compare_cols = ["age", "sex", "idh_status", "codel_status", "diagnosis", "grade"]
    rows = []
    for _, r in fu.iterrows():
        base = ucsf[(ucsf["patient_id"] == r["patient_id"]) & (~ucsf["is_followup"])]
        conflicts = [] if len(base) != 1 else [c for c in compare_cols if str(base.iloc[0][c]) != str(r[c])]
        rows.append({"followup_source_id": r["source_id"], "patient_id": r["patient_id"],
                     "baseline_rows": len(base), "baseline_source_id": base["source_id"].iloc[0] if len(base) else None,
                     "idh_status": r["idh_status"], "conflicting_fields": ", ".join(conflicts) or "none"})
    return pd.DataFrame(rows, columns=["followup_source_id", "patient_id", "baseline_rows", "baseline_source_id",
                                       "idh_status", "conflicting_fields"])


def _missingness(cohort_all):
    fields = ["age", "sex", "grade", "idh_status", "idh_method", "codel_status", "scanner_make", "field_strength_T",
              "seg_provenance", "biopsy_prior_to_imaging"]
    miss = {}
    for coh, g in cohort_all.groupby("cohort"):
        miss[coh] = {f: int(g[f].isna().sum() + (g[f].astype(str) == "unknown").sum()) for f in fields}
    out = pd.DataFrame(miss)
    out.index.name = "field (missing or 'unknown')"
    return out


# ---------------------------------------------------------------- label flags and flow (section 5)
def _add_flags(cohort_all):
    dx = cohort_all["diagnosis"].str.upper()
    is_utsw, is_ucsf = cohort_all["cohort"].eq("UTSW"), cohort_all["cohort"].eq("UCSF")
    codel = cohort_all["codel_status"]
    flags = pd.DataFrame(index=cohort_all.index)
    flags["histology_oligodendroglioma_but_intact"] = is_utsw & dx.str.startswith("OLIGODENDROGLIOMA") & codel.eq("non-codeleted")
    flags["histology_astrocytoma_but_codeleted"] = is_utsw & dx.str.startswith("ASTROCYTOMA") & codel.eq("codeleted")
    flags["histology_oligoastrocytoma"] = is_utsw & dx.str.startswith("OLIGOASTROCYTOMA")
    flags["histology_glioblastoma_idh_mutant"] = is_utsw & dx.eq("GLIOBLASTOMA") & cohort_all["idh_status"].eq("mutant")
    flags["grade4_codeleted"] = cohort_all["grade"].eq(4) & codel.eq("codeleted")
    flags["who2021_oligodendroglioma_label_not_codeleted"] = is_ucsf & dx.str.startswith("OLIGODENDROGLIOMA") & ~codel.eq("codeleted")
    flags["who2021_astrocytoma_label_codeleted"] = (is_ucsf & dx.str.startswith("ASTROCYTOMA, IDH-MUTANT")
                                                    & codel.isin(["codeleted", "relative_codeletion"]))
    flags["ucsf_relative_codeletion"] = codel.eq("relative_codeletion")
    flags["ucsf_biopsy_prior_to_imaging"] = cohort_all["biopsy_prior_to_imaging"].eq(True)
    flags["idh_wildtype_codeleted"] = cohort_all["idh_status"].eq("wildtype") & codel.eq("codeleted")
    return pd.concat([cohort_all, flags.add_prefix("flag_")], axis=1)


def _flow_counts(df, step):
    rows = []
    for coh in ["UTSW", "UCSF"]:
        g = df[df["cohort"] == coh]
        rows.append({"step": step, "cohort": coh, "n": len(g),
                     "codeleted": int((g["codel_status"] == "codeleted").sum()),
                     "non-codeleted": int((g["codel_status"] == "non-codeleted").sum()),
                     "relative": int((g["codel_status"] == "relative_codeletion").sum()),
                     "unknown": int((g["codel_status"] == "unknown").sum())})
    return rows


def load_cohort(cfg, rep):
    """Run sections 4–5.1 and return a Cohort. Checks are recorded on the reporter."""
    utsw_path = cfg["data_root"] / cfg["utsw_metadata"]
    ucsf_path = cfg["data_root"] / cfg["ucsf_metadata"]
    utsw_raw, utsw = load_utsw(utsw_path)
    ucsf_raw, ucsf = load_ucsf(ucsf_path)
    hashes = {utsw_path.name: sha256(utsw_path), ucsf_path.name: sha256(ucsf_path)}
    rep.say(f"UTSW rows: {len(utsw_raw)} | UCSF rows: {len(ucsf_raw)}")
    rep.say(f"SHA-256: {hashes}")

    # Section 4: integrity
    rep.record_check("UTSW source IDs unique", utsw["source_id"].is_unique, f"{utsw['source_id'].nunique()} unique of {len(utsw)}")
    rep.record_check("UCSF source IDs unique", ucsf["source_id"].is_unique, f"{ucsf['source_id'].nunique()} unique of {len(ucsf)}")
    followups = _followup_table(ucsf)
    rep.show(followups, "UCSF follow-up scans detected:")
    rep.record_check("Every UCSF follow-up has exactly one baseline row",
                     bool((followups["baseline_rows"] == 1).all()) if len(followups) else True, f"{len(followups)} follow-up rows")
    ucsf_patients = ucsf[~ucsf["is_followup"]].copy()
    rep.record_check("UCSF canonical patient IDs unique after removing follow-ups", ucsf_patients["patient_id"].is_unique,
                     f"{ucsf_patients['patient_id'].nunique()} patients")
    cohort_all = pd.concat([utsw, ucsf_patients], ignore_index=True)
    rep.record_check("No patient ID shared across cohorts", cohort_all["patient_id"].is_unique)
    missingness = _missingness(cohort_all)
    rep.show(missingness, "Missingness of harmonised fields by cohort:")
    rep.save_table(missingness, "04_missingness_by_cohort")
    rep.save_table(followups, "04_ucsf_followup_scans")

    # Section 5: labels, flags, flow
    cohort_all["label"] = cohort_all["codel_status"].map(label_from_status)
    cohort_all = _add_flags(cohort_all)
    idh_mut = cohort_all["idh_status"].eq("mutant")
    flag_cols = [c for c in cohort_all if c.startswith("flag_")]
    flag_summary = cohort_all.loc[idh_mut, ["cohort"] + flag_cols].groupby("cohort").sum().T
    flag_summary.index = flag_summary.index.str.replace("flag_", "", regex=False)
    rep.show(flag_summary, "Flags among IDH-mutant patients (counts):")
    rep.save_table(flag_summary, "05_flags_idh_mutant")
    n_wt = int(cohort_all["flag_idh_wildtype_codeleted"].sum())
    n_g4 = int(cohort_all["flag_grade4_codeleted"].sum())
    rep.record_check("No IDH-wildtype tumour labelled codeleted", n_wt == 0, f"{n_wt} cases")
    rep.record_check("No grade-4 tumour labelled codeleted", n_g4 == 0, f"{n_g4} cases")

    all_rows = pd.concat([utsw, ucsf], ignore_index=True)
    flow = _flow_counts(all_rows, "S0 rows in metadata files")
    flow += _flow_counts(cohort_all, "S1 unique patients (UCSF follow-up scans removed)")
    s2 = cohort_all[cohort_all["idh_status"] != "unknown"]
    flow += _flow_counts(s2, "S2 IDH status known")
    s3 = s2[s2["idh_status"] == "mutant"]
    flow += _flow_counts(s3, "S3 IDH-mutant")
    s4 = s3[s3["codel_status"].isin(["codeleted", "non-codeleted"])]
    flow += _flow_counts(s4, "S4 1p/19q resolved (primary label)")
    flow_df = pd.DataFrame(flow)
    rep.show(flow_df.set_index(["step", "cohort"]), "Patient flow (metadata):")
    rep.save_table(flow_df, "05_patient_flow_metadata")

    eligible = s4.copy().reset_index(drop=True)
    eligible["label"] = eligible["label"].astype(int)
    assert eligible["patient_id"].is_unique
    assert set(eligible["label"].unique()) <= {0, 1}
    assert eligible["idh_status"].eq("mutant").all()
    rep.record_check("Eligible cohort is IDH-mutant only", True, f"{len(eligible)} patients")

    # Label-sensitivity sets (used in the robustness analyses)
    sens = s3.copy()
    who_label = np.where(sens["diagnosis"].str.startswith("Oligodendroglioma"), 1.0,
                         np.where(sens["diagnosis"].str.startswith("Astrocytoma, IDH-mutant"), 0.0, np.nan))
    sens["label_who2021_resolved"] = np.where(
        sens["cohort"].eq("UCSF") & sens["codel_status"].isin(["relative_codeletion", "unknown"]), who_label, sens["label"])
    label_sets = {
        "primary": eligible["patient_id"].tolist(),
        "ucsf_relative_resolved_by_who2021": sens.loc[sens["label_who2021_resolved"].notna(), "patient_id"].tolist(),
        "exclude_histology_discordant": eligible.loc[~(eligible["flag_histology_oligodendroglioma_but_intact"]
                                                       | eligible["flag_histology_astrocytoma_but_codeleted"]), "patient_id"].tolist(),
        "exclude_grade4": eligible.loc[~eligible["grade"].eq(4), "patient_id"].tolist(),
        "exclude_ucsf_biopsy_prior": eligible.loc[~eligible["biopsy_prior_to_imaging"].eq(True), "patient_id"].tolist(),
    }
    rep.say(f"Label sets: { {k: len(v) for k, v in label_sets.items()} }")

    cohort = Cohort(utsw_raw=utsw_raw, ucsf_raw=ucsf_raw, all_rows=all_rows, patients=cohort_all, eligible=eligible,
                    sens=sens, flow=flow_df, flag_summary=flag_summary, followups=followups, missingness=missingness,
                    label_sets=label_sets, input_hashes=hashes)
    cohort.original_cohort_hypothesis = reconstruct_iteration2_cohort(eligible, rep)
    return cohort


def reconstruct_iteration2_cohort(eligible, rep):
    """Section 5.1: test one candidate filter for the iteration-2 cohort (156 patients: 46 codeleted / 110 not)."""
    cand = eligible[(eligible["cohort"].eq("UTSW") & eligible["seg_provenance"].eq("FeTS + manual refinement"))
                    | (eligible["cohort"].eq("UCSF") & ~eligible["biopsy_prior_to_imaging"].eq(True))]
    rep.show(cand.groupby(["cohort", "label"]).size().unstack(fill_value=0), "Candidate iteration-2 cohort:")
    n_pos, n_neg = int(cand["label"].sum()), int((1 - cand["label"]).sum())
    match = (len(cand), n_pos, n_neg) == (156, 46, 110)
    rep.say(f"Candidate filter (UTSW manually refined segmentations + UCSF without prior biopsy): {len(cand)} patients, "
            f"{n_pos} codeleted / {n_neg} non-codeleted -> {'MATCHES' if match else 'does not match'} iteration 2 (156: 46/110).")
    site_auc = roc_auc_score(cand["label"], cand["cohort"].eq("UTSW").astype(float))
    rep.say(f"In this candidate cohort, cohort membership alone gives AUROC {site_auc:.3f} "
            f"(codeletion prevalence UTSW {cand.loc[cand.cohort.eq('UTSW'), 'label'].mean():.2f}, "
            f"UCSF {cand.loc[cand.cohort.eq('UCSF'), 'label'].mean():.2f}).")
    return {"filter": "UTSW manual refinement == Yes; UCSF biopsy prior == No; IDH-mutant; label resolved",
            "n": len(cand), "codeleted": n_pos, "non_codeleted": n_neg, "matches_iteration2_counts": bool(match),
            "cohort_only_auroc": site_auc}
