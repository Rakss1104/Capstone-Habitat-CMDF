# habitat1p19q

Habitat imaging for 1p/19q codeletion in IDH-mutant gliomas. This package is a modular rebuild of
`Habitat_1p19q_IDHmut_v3.ipynb`. It keeps the notebook's methods, settings and pre-specified decision rule unchanged,
and runs end to end from the command line.

**Question.** Among IDH-mutant gliomas, does the *spatial arrangement* of MRI habitats tell codeleted tumours from
non-codeleted ones better than habitat composition alone? The test is AUROC(M4) > AUROC(M3), within UTSW.
H1 counts as supported only if the lower bound of the Nadeau–Bengio 95% CI for ΔAUROC is above 0 **and** the
spatial-arrangement null gives p < 0.05.

## Layout

```
src/habitat1p19q/
  config.py          all settings (one dict), env-var/CLI overrides, extraction fingerprint
  cohort.py          §4–5   metadata ingestion, strict value maps, integrity checks, label flags, patient flow
  audit.py           §6     Table 1, site–label association, cohort-only baseline, clinical associations
  imaging/qc.py      §7     study table, file location, geometry/segmentation QC
  imaging/extraction.py §8  N4 + brain z-score, patch grid, depth map, PyRadiomics (patch + whole tumour), cache
  imaging/synthetic.py      synthetic smoke-test volumes
  habitat.py         §9     HabitatModel (filter → scale → PCA → k-means, stability-selected K), descriptive model
  spatial.py         §10    composition + composition-adjusted spatial metrics, toy-grid self-test
  modeling.py        §11    model ladder M0–M5, estimators, training-only thresholds, FoldFeatureBuilder
  validation.py      §12–13 repeated CV with habitats refit per fold, summaries, Nadeau–Bengio paired comparisons
  null_tests.py      §13.4–13.5 label permutation (within cohort) and spatial-arrangement null
  transport.py       §14    leave-one-cohort-out
  robustness.py      §15    calibration, grouped permutation importance, robustness of the primary Δ
  figures.py         §16    ROC, AUROC forest, null distributions
  conclusion.py      §17–18 conclusion text from the executed results, checklist, manifest
  pipeline.py        Study class that runs the stages in order
  cli.py             command-line entry point
tests/               unit tests, cohort tests on the real metadata, synthetic end-to-end smoke test
scripts/             WSL environment setup and run helpers
```

## Environment

PyRadiomics has no Windows wheel and needs a C toolchain to build, so the pipeline runs on Linux (on Windows, use
WSL). `scripts/setup_wsl_env.sh` uses `uv` to fetch a standalone CPython 3.12 that includes its own headers, so no
system packages are needed. It then installs `requirements.txt`, with PyRadiomics pinned to the same commit as the
notebook, and installs this package in editable mode.

```bash
bash scripts/setup_wsl_env.sh                      # creates ~/venvs/habitat2
```

## Running

```bash
# full study on the real data (defaults: data = ../1P19Q -Dataset, outputs = ./outputs)
bash scripts/run_study_wsl.sh --n-jobs 7

# or directly
python -m habitat1p19q --data-root "<folder with metadata + UTSW_processed + UCSF/images>" \
                       --output-dir outputs --mode real --n-jobs 7

# software verification on synthetic volumes (every output stamped SYNTHETIC SMOKE TEST)
python -m habitat1p19q --data-root "<data>" --output-dir smoke --mode synthetic_smoke_test

# any setting can be overridden (dotted keys, JSON values)
python -m habitat1p19q ... --set cv.n_repeats=5 --set n_label_permutations=100 --set harmonisation='"per_cohort_zscore"'

# tests
GLIOMA_DATA_ROOT="<data>" python -m pytest -q            # add -m "not slow" to skip the smoke run
```

Expected data layout (the same as in the notebook):

```
<data_root>/UTSW_Glioma_Metadata.tsv
<data_root>/UCSF-PDGM-metadata_v5.csv
<data_root>/UTSW_processed/BT####/{T1,T1_GD,T2,FLAIR,segmentation}.nii.gz
<data_root>/UCSF/images/UCSF-PDGM-####/{T1,T1_GD,T2,FLAIR,seg_mask}.nii.gz
```

**Memory.** Each extraction worker needs about 2 GB. Extraction therefore uses `extraction_n_jobs` (default 3),
which is separate from `n_jobs` (null tests). Raise it only if WSL has more memory available (see `.wslconfig`).

## Outputs

- `outputs/tables/*.csv`: one table per analysis, with a section-number prefix.
- `outputs/figures/*.png`
- `outputs/conclusion.md`
- `outputs/manifest.json`: environment, config, input hashes, patient flow and check status.
- `outputs/run.log`
- `outputs/timings.json`

Per-patient features are cached in `outputs/features/per_patient/` and keyed by an extraction fingerprint, so changing
any extraction setting invalidates the cache automatically.

## Differences from the notebook

The methods are unchanged. Only the engineering differs:

- **No hidden global state.** The notebook's `CONFIG`, `BASE`, `CHECKS` and `RESULTS` globals are now explicit
  objects (`cfg`, the base table held by `FoldFeatureBuilder`, `Reporter.checks`, `Study.results`). Joblib workers
  get everything they need as arguments.
- **QC geometry check.** It now also compares image direction, matching the `same_geometry` test used at extraction.
  Before, QC could pass a case that extraction would then resample.
- **More saved results.** The primary comparison and the spatial-null summary are written as tables
  (`13_primary_comparison.csv`, `13_spatial_arrangement_null.csv`). The conclusion also lists every discrimination
  summary and secondary comparison, not only the metadata baselines.
- **Defensive guard.** `FoldFeatureBuilder` skips habitat assignment for patients with no patches instead of
  calling k-means on an empty frame. This cannot happen in the standard analysis sets.
- `Habitat_features_iteration_2.ipynb` is superseded by v3. Its problems are listed in v3 §8–10: leakage from fitting
  habitats before CV, duplicated shape columns, and the unstable edge/core ratio.
