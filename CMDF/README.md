# CMDF — cross-modal differential features for 1p/19q codeletion

Do *co-located* disagreements between MRI sequences inside the tumour distinguish 1p/19q-codeleted from
non-codeleted IDH-mutant gliomas better than the same radiomics read one sequence at a time? This folder holds a
pre-registered, leakage-free rebuild of the earlier CMDF notebooks, adapted to the current dataset
(`../1P19Q -Dataset`: UTSW + UCSF-PDGM) and built on the habitat study's cohort, QC, patch features and folds.

## Layout

```
PROTOCOL.md                          analysis plan, fixed before any CMDF result (its SHA-256 is in the manifest)
cmdf1p19q/
  config.py       settings; inherits cohort/imaging/model settings from habitat1p19q
  features.py     blocks P (single sequence), D (patch CMDF), fold-fitted cosine concordance; pairing null
  voxel.py        voxel CMDF (V) and the quantitative T2–FLAIR mismatch block (MM), cached per patient
  cv.py           repeated CV (parallel folds), pairing null, label permutation, leave-one-cohort-out
  naive.py        methodological arm: the original notebooks' protocol, observed and on permuted labels
  figures.py      result figures      conclusion.py   generated conclusion      study.py   the pipeline
notebooks/
  01_cmdf_feature_extraction.ipynb   replaces feature_extraction_cmdf (1).ipynb
  02_cmdf_train_evaluate.ipynb       replaces train_test_evaluate_cmdf_final.ipynb
scripts/run_cmdf_wsl.sh             one-command run
tests/                               unit tests of the feature definitions and the pairing null
outputs/                             tables/, figures/, features/voxel/, conclusion.md, manifest.json, run.log
feature_extraction_cmdf (1).ipynb, train_test_evaluate_cmdf_final.ipynb   the original notebooks, unchanged
```

## Run

Requires the habitat study to have been run once (its patch-feature cache under `../habitat_1p19q/outputs/features`
is reused; nothing is re-extracted). From WSL:

```bash
bash scripts/run_cmdf_wsl.sh                 # full study, ~1.5–2 h on 8 cores
~/venvs/habitat2/bin/python -m pytest tests  # unit tests
```

Paths can be overridden with `GLIOMA_DATA_ROOT`, `HABITAT_OUTPUT_DIR`, `CMDF_OUTPUT_DIR`. To use the notebooks, add
`ipykernel` to `~/venvs/habitat2` and open them from `notebooks/`.

## What changed from the original notebooks

| Issue in the original | Fix |
|---|---|
| Cohort included IDH-wildtype tumours (never codeleted), so the task was largely IDH status | IDH-mutant only, label resolved, imaging QC passed |
| 62 patients with 10 in the minority class, one 5-fold split (AUROC SD 0.12–0.29) | 150 patients, 10 × repeated 5-fold, bootstrap CIs, corrected paired tests |
| Best of 6 models chosen on the test folds and reported | one pre-specified model family and a pre-specified primary comparison |
| `outlier_deviation` used σ over all patients (test leakage) | the only fitted block (cosine concordance) is fitted on training patients per fold |
| `variance` = (a − b)²/4 duplicates L2 | removed |
| Cosine on raw features, dominated by Energy-scale features | cosine after per-sequence standardisation |
| T1-Gd vs T2 only | all six sequence pairs, plus voxel-level CMDF and a quantitative T2–FLAIR mismatch |
| No test of whether co-location matters | pairing null: same per-sequence distributions, co-location destroyed |
| Accuracy/F1 with a majority positive class | AUROC, AUPRC, calibration, threshold chosen on training folds only |

Results are in `outputs/conclusion.md`.
