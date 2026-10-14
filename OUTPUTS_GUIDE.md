# Outputs guide

What every table and figure in the three result folders contains and how to read it. Headline numbers are quoted
so you can find the row that matters; the files themselves hold the full detail.

```
habitat_1p19q/outputs/   Study 1: do MRI habitats (and their spatial arrangement) predict 1p/19q?
CMDF/outputs/            Study 2: do cross-modal feature differences (patch and voxel level) predict 1p/19q?
models/                  Study 3: nine classifiers trained on the best feature sets of both approaches, compared
```

---

## 0. Shared vocabulary

**Cohorts**

| Name | Who | n (codeleted) |
|---|---|---|
| `UTSW` | UTSW IDH-mutant patients with images *and* a segmentation | 74 (36) |
| `UTSW_meta` | All UTSW IDH-mutant patients with a 1p/19q label (age/sex models only, no images needed) | 140 (63) |
| `pooled` | UTSW 74 + UCSF 76 imaging patients; a cohort indicator is available to models | 150 (44) |
| `pooled_meta` | UTSW 140 + UCSF 93, metadata only | 233 (72) |

UTSW is the clean comparison (one centre, ~50 % prevalence). Pooled numbers are inflated by site: codeletion is
49 % in UTSW but 11 % in UCSF, so a model that only learns "which hospital" already reaches AUROC ≈ 0.70.

**Models shared by both studies**

| Model | Features |
|---|---|
| M0 | cohort indicator only (pooled) |
| M1_clin | age + sex |
| M1 | age + sex + log tumour volume |
| M2 | whole-tumour radiomics (386 features) |

**Statistics used everywhere**

| Term | Meaning |
|---|---|
| AUROC (repeat-avg OOF) | Each patient's out-of-fold prediction is averaged over the 10 CV repeats, then one AUROC is computed. 0.5 = chance. The 95 % CI is a patient bootstrap. |
| AUROC per repeat | One AUROC per CV repeat: mean [min–max]. Shows how much the result moves with the fold split. |
| AUPRC | Area under the precision–recall curve. Compare it with the prevalence, which is what a random model scores. |
| ΔAUROC (fold mean), NB 95 % CI, NB p | Paired difference between two models on the same 50 folds. Nadeau–Bengio correction widens the CI because CV folds overlap; without it the CIs would be falsely narrow. |
| folds with Δ>0 | How many of the 50 folds favoured the first model. 25/50 = no consistent direction. |
| BH-adjusted p / q | Benjamini–Hochberg false-discovery-rate correction across a family of tests. |
| Cliff's δ | Rank effect size, −1 to +1. 0 = no difference; ±0.33 is medium and ±0.47 large. The sign says which group is higher. |
| Label permutation p | The whole pipeline is rerun 200 times with shuffled labels. p = share of shuffled runs that score as well as the real one. 0.005 is the floor (1/201). |
| Calibration slope | 1 = ideal. Below 1: predictions too extreme (overfit). Above 1: too timid. |
| CITL (intercept) | Calibration-in-the-large. 0 = predicted risk matches the observed rate on average. |
| Brier | Mean squared error of the probabilities. Lower is better; 0.25 = always predicting 0.5 at 50 % prevalence. |
| Sensitivity, specificity, precision, F1, balanced acc. | Computed at a threshold chosen on training data only (the one maximising balanced accuracy). |
| MCC | Matthews correlation, −1 to +1. 0 = chance. Robust to class imbalance. |

---

## 1. `habitat_1p19q/outputs/` — habitat study

**What was tested.** Each tumour is cut into 32-mm patches. Patch radiomics are clustered into K "habitats",
refitted inside every training fold. Two feature sets come out of this:

- **M3, composition**: the share of each habitat in the tumour (5 features).
- **M4, composition + spatial**: M3 plus how habitats are arranged (11 features). The spatial part covers mixing
  between neighbours, radial (margin-to-core) organisation and habitat entropy.

M5 = M4 + volume, age and sex. **Primary hypothesis H1:** AUROC(M4) > AUROC(M3) in UTSW. **Result: not supported.**

### Top-level files

| File | What it is |
|---|---|
| `conclusion.md` | Plain-text verdict on every pre-specified test. Read this first. |
| `manifest.json` | Full configuration, software versions, patient counts, extraction fingerprint `78c614a2dc75`, and the 28 PASS/FAIL integrity checks. |
| `run.log` | Timestamped log of the run. |
| `timings.json` | Seconds spent per stage. |
| `features/per_patient/` | Cached patch radiomics, one JSON per patient (156). Git-ignored; rebuilding needs WSL + PyRadiomics. |

### Tables — data and cohort (04–08)

| Table | What it shows | How to read it / key value |
|---|---|---|
| `04_missingness_by_cohort.csv` | Count of missing or "unknown" values per metadata field, per cohort | Age and sex are complete. Checks that the clinical baseline isn't biased by missing data. |
| `04_ucsf_followup_scans.csv` | The 6 UCSF rows that are follow-up scans of an existing patient | Each is matched to exactly one baseline and removed, so no patient is counted twice. |
| `05_flags_idh_mutant.csv` | Label inconsistencies, e.g. histology says oligodendroglioma but 1p/19q is intact | UTSW: 4 oligodendrogliomas without codeletion, 2 astrocytomas with codeletion. The molecular label wins; a robustness run excludes these cases. |
| `05_patient_flow_metadata.csv` | CONSORT-style flow: all rows → unique patients → IDH known → IDH-mutant → 1p/19q resolved | UTSW 625 → 140 (63 codeleted); UCSF 501 → 93 (9). Use this for the paper's flow diagram. |
| `06_prevalence_by_cohort.csv` | Codeletion rate per cohort with Wilson CI | UTSW 0.45, UCSF 0.10. This gap is why pooled models need a cohort term. |
| `06_table1_by_cohort_and_label.csv` | "Table 1": age, sex, grade, scanner by cohort × label | Codeleted patients are older on average (UTSW 43.9 vs 36.3, UCSF 43.4 vs 37.9 years), which explains why age + sex is a real baseline. |
| `06_clinical_and_acquisition_by_label.csv` | Tests whether age, sex and field strength differ by label within each cohort | UTSW: codeleted patients older (median 42 vs 35, r 0.33, p 0.0008); sex and 3 T vs 1.5 T don't differ (OR 0.82, 1.10). So the scanner isn't a proxy for the label. |
| `07_imaging_qc.csv` | One row per candidate patient: missing files, segmentation labels, spacing, size, tumour volume, QC status | 156 `ok`, 87 `missing_files`. **64 of the UTSW failures are missing only the segmentation**; these are the cases marked "not manually refined". |
| `08_exclusions_by_class.csv` | Exclusions split by label | UTSW lost 26 codeleted and 38 non-codeleted to QC. Final: UTSW 36 + 38, UCSF 8 + 68. |

### Tables — habitats (09–10)

| Table | What it shows | How to read it / key value |
|---|---|---|
| `09_k_selection_descriptive.csv` | For K = 3…8: stability (ARI over resamples), silhouette, inertia, smallest habitat share | K is chosen by stability (high ARI) with no tiny habitats. Descriptive only; the CV refits K per fold. |
| `09_habitat_phenotypes_descriptive.csv` | Per habitat: patch count, median z-scored T1, T1CE, T2 and FLAIR, median depth (0 = margin) | Names the habitats. H1 = dark T1/T1CE, very bright T2/FLAIR, a "fluid-rich" phenotype. The others differ mostly in degree. |
| `10_spatial_metric_confounding.csv` | Spearman ρ between each spatial metric and tumour volume, patch count and number of habitats present | Raw mixing correlates with size (ρ ≈ 0.34). The `_z` / `_excess` versions are the size-corrected metrics actually used in M4. |

### Tables — discrimination and tests (13–14)

| Table | What it shows | How to read it / key value |
|---|---|---|
| `13_discrimination_summary.csv` | **Main results table.** Every model × set: AUROC (CI), per-repeat spread, AUPRC, Brier, threshold metrics, confusion matrix of repeat 1 | UTSW: M1_clin 0.659, M2 0.692, M3 0.607, M4 0.597. Pooled: M2 0.828, M4+cohort 0.800, M0 0.696. |
| `13_primary_comparison.csv` | **H1 test:** M4 − M3 in UTSW | Δ −0.029, NB CI −0.146 to 0.088, p 0.62, 20/50 folds positive. Spatial arrangement adds nothing. |
| `13_secondary_comparisons.csv` | 9 pre-specified comparisons, BH-adjusted (e.g. M3 − M1, M4 − M2, pooled M3 − M0) | Only pooled M3 − M0 is significant, and it is **negative** (−0.188, BH p 0.026): habitat composition does worse than the cohort label alone. |
| `13_spatial_arrangement_null.csv` | Habitat labels shuffled across patch positions within each tumour (composition kept, arrangement destroyed); M4 − M3 recomputed 100 times | Observed Δ −0.020 vs null mean −0.016, p 0.475. Real arrangement = random arrangement. |
| `13_label_permutation.csv` | Whole pipeline on shuffled labels (200×) | UTSW M3 p 0.134, M4 p 0.204: **not distinguishable from noise**. UTSW_meta age + sex p 0.025: real. The pooled_meta M0 row is a sanity check, since the cohort label alone sits inside its own null. |
| `13_within_cohort_discrimination.csv` | Pooled models scored separately inside UTSW and inside UCSF | Shows whether a pooled AUROC survives without the site difference. Habitat M3/M4 without the cohort term score 0.47–0.49 inside UTSW (chance) and 0.57–0.61 inside UCSF. |
| `13_exploratory_univariate.csv` | Each habitat feature compared by label (Cliff's δ, BH q) | Not pre-specified. Best: share of habitat H1 (fluid-rich) lower in codeleted tumours, δ −0.31, q 0.096. Nothing survives FDR. |
| `14_leave_one_cohort_out.csv` | Train on UTSW, test on UCSF (UCSF → UTSW not run: 8 events) | M2 0.757, M3 0.890, M4 0.673, M1_clin 0.585. **Unreliable: only 8 UCSF events**, CIs ±0.2. Don't cite M3's 0.890. |

### Tables — robustness, calibration, audit trail (15–18)

| Table | What it shows | How to read it / key value |
|---|---|---|
| `15_calibration.csv` | Calibration slope, CITL and Brier per model | M1_clin slope 1.90 means age + sex predictions are too timid. Imaging models are closer to 1. |
| `15_grouped_permutation_importance_M4.csv` | AUROC lost when all composition, or all spatial, features of M4 are shuffled | Composition +0.090, spatial −0.005: M4 relies on composition alone. |
| `15_robustness_primary_delta.csv` | H1 rerun with fixed K = 4…8, without histology-discordant cases, without grade 4, with per-cohort z-scoring, and with an alternative UCSF label rule | Δ stays at or below 0 in every variant, so the null result isn't a tuning artefact. |
| `16_fold_records_*.csv` | One row per fold × repeat × model: n, AUROC, K, chosen C, threshold, TP/FP/TN/FN | Raw material for every summary above. Use it to recompute anything or plot fold spread. |
| `18_checklist.csv` | 28 automated integrity checks (unique IDs, no follow-up duplication, no patient in train and test, fingerprints) | All PASS. |

### Figures

| Figure | What it shows |
|---|---|
| `06_cohort_audit.png` | Left: codeletion prevalence per cohort with CI (UCSF 9/93 vs UTSW 63/140). Middle: age by label within each cohort (codeleted patients older). Right: UTSW field strength by label (similar mix, so scanner isn't a proxy for the label). |
| `09_k_selection_descriptive.png` | Stability (ARI) and silhouette against K, the basis for the descriptive K. |
| `09_habitat_phenotypes.png` | Heatmap of median sequence intensity per habitat, H0–H3. A visual definition of what each habitat looks like on MRI. |
| `15_calibration_UTSW.png` | Predicted vs observed risk, binned, per UTSW model. The diagonal is perfect calibration. |
| `16_auroc_forest.png` | Forest plot of every model's AUROC with 95 % CI, UTSW and pooled. The fastest overview of the study. |
| `16_null_distributions.png` | Histograms of the label-permutation nulls (M3, M4, age + sex, M0) and the spatial-arrangement null, with the observed value as an orange line. Observed inside the histogram = no signal. |
| `16_roc_curves.png` | ROC curves of the main models. |

---

## 2. `CMDF/outputs/` — cross-modal feature difference study

**What was tested.** "Cross-modal" means the same tumour location compared across two MRI sequences. Six sequence
pairs: T1–T1CE, T1–T2, T1–FLAIR, T1CE–T2, T1CE–FLAIR, T2–FLAIR. Feature sets:

| Set | Columns | Content |
|---|---|---|
| P | 744 | single-sequence patch radiomics (baseline) |
| D | 1,128 | patch-level CMDF: per pair, mean and SD of the per-patch feature difference, plus cosine similarity |
| P+D | 1,872 | both |
| D_T2FL | 188 | D for the T2–FLAIR pair only |
| V | 45 | **voxel-level CMDF**: per pair, the distribution of voxel-wise intensity differences (mean, SD, 10th/50th/90th percentile) and the Pearson/Spearman coupling, 6 × 7 = 42, plus the 3 MM features |
| MM | 3 | quantitative T2–FLAIR mismatch: mismatch fraction, FLAIR core − rim, T2 core − rim |

- **Primary hypothesis H1:** P+D > P in UTSW. **Not supported.**
- **H2:** the mismatch fraction is lower in codeleted tumours. **Confirmed in UTSW.**
- **The naive arm** reruns the *original notebooks' protocol* on the same features, to measure its optimism.

### Top-level files

| File | What it is |
|---|---|
| `conclusion.md` | Verdict on each pre-specified test (H1, H2, S1–S7, nulls). |
| `INSIGHTS.md` | Interpreted findings and what they mean for a paper. |
| `manifest.json` | Config, versions, counts, fingerprints, checks (7 PASS). |
| `run.log` | Log of the completed run (Windows venv). |
| `run_attempt1_killed.log`, `attempt1_killed/` | First WSL run, killed at stage 6 by a memory reaper. Its stage-5 tables are kept as proof that the rerun reproduced them byte for byte. Not results in their own right. |
| `features/voxel/*.json` | V and MM features, one JSON per patient (150), with an extraction fingerprint. |
| `cache/` | `data.pkl` and the `state.pkl` resume checkpoint. Git-ignored, regenerable. |

### Tables

| Table | What it shows | How to read it / key value |
|---|---|---|
| `02_voxel_status.csv` | Voxel-feature extraction per patient: status, fingerprint, voxel count, maximum depth, core and rim voxel counts, seconds | 150/150 ok. Core/rim come from distance to the tumour margin and feed the MM core − rim features. |
| `04_*`, `05_flags_idh_mutant.csv`, `05_patient_flow_metadata.csv`, `07_imaging_qc.csv`, `08_exclusions_by_class.csv` | Same cohort audit as the habitat study | Identical cohort and folds to the habitat study, so the two studies can be compared directly. |
| `04_fold_records_UTSW.csv`, `04_fold_records_pooled.csv` | Per-fold raw results, all models | As in habitat `16_fold_records_*`. |
| `05_discrimination_summary.csv` | **Main results table**, all models × sets | UTSW: P 0.662, P+D 0.628, D 0.610, D_T2FL 0.675, **V 0.765 (0.651–0.871)**, MM 0.703, M2+V 0.708. Pooled: **V+cohort 0.851**, P 0.802, V 0.800. |
| `05_primary_comparison.csv` | **H1:** P+D − P in UTSW | Δ −0.046, NB CI −0.112 to 0.020, 8/50 folds positive. Patch CMDF slightly *hurts*. |
| `05_secondary_comparisons.csv` | S1–S7 with BH adjustment | Only **S7** is significant: pooled V+cohort − M0, +0.115, BH p 0.011. S4 (V − age/sex in UTSW): +0.085, CI −0.093 to 0.263, not significant, so underpowered. |
| `05_voxel_univariate.csv` | Each of the 45 V features by label: medians, Cliff's δ (CI), p, BH q, per set | H2 row: mismatch fraction 0.024 vs 0.059, δ −0.34, p 0.012. Strongest exploratory signal: **T1–T2 Spearman coupling** (−0.48 codeleted vs −0.76), δ +0.61, q 0.0003. FLAIR core − rim pooled δ +0.42, q 0.003. |
| `05_within_cohort_discrimination.csv` | Pooled models scored within UTSW and within UCSF | V+cohort: UTSW 0.790, UCSF 0.801. It isn't just learning the site. |
| `06_label_permutation.csv` | Shuffled-label null for P+D and V (200×) | **V 0.702 vs null 0.501, p 0.005**, the only imaging set in either study to beat noise. P+D p 0.129. |
| `06_pairing_null.csv` | Sequence pairing shuffled across patches (destroys real co-location); P+D − P recomputed 100 times | Observed −0.046 vs null −0.008, p 1.00. Real co-location carries **no** extra information at patch scale. |
| `07_leave_one_cohort_out.csv` | Train UTSW → test UCSF | P+D 0.794, D 0.772, P 0.759, M2 0.757, V 0.737, M1_clin 0.585. Descriptive only (8 events): every CI is roughly 0.56–0.94, so they can't be ranked. |
| `08_calibration.csv` | Slope, CITL and Brier per model | V slope 0.77, Brier 0.199: best calibrated imaging model. |
| `08_pair_ablation.csv` | P+D AUROC with one sequence pair removed at a time | Every Δ is within ±0.02. T2–FLAIR matters most (−0.018), but no pair matters much at patch scale. |
| `08_robustness_primary.csv` | H1 rerun with a random forest, and in patients with ≥ 4 patches | Δ −0.008 and −0.030: the null holds. |
| `09_original_protocol_classifiers.csv` | The original notebook protocol: fold-mean AUROC per classifier (KNN, DT, RF, SVM, GNB, LR) in two variants | The "best" classifier changes with the variant, a sign of selection noise. |
| `09_original_protocol_summary.csv` | **Naive arm summary:** best classifier, its AUROC, the same protocol on permuted labels (mean / 95th pct / max), p, leakage-free AUROC, optimism | "notebook": 0.712, random labels 0.588 / 0.698 / 0.751. "select_all" (features picked on all patients first): 0.827, random labels **0.745 / 0.840 / 0.885**. Leakage-free 0.628, so optimism is +0.08 and +0.20. |

### Figures

| Figure | What it shows |
|---|---|
| `10_auroc_forest.png` | Forest plot of all models' AUROC (CI), UTSW and pooled. V stands out in UTSW. |
| `10_roc_curves.png` | ROC curves of the main models. |
| `10_null_distributions.png` | Top: pairing null (observed −0.046 far left of the null, so worse than random pairing), label-permutation nulls for P+D (inside the null) and V (far right, p 0.005). Bottom: the original protocol on permuted labels; its reported AUROC sits *inside* the noise distribution, which is the key methods figure. |
| `10_pair_ablation.png` | Bar chart of ΔAUROC when each sequence pair is dropped. |
| `10_t2flair_mismatch.png` | Mismatch fraction by label and cohort (H2). Codeleted tumours are lower; every codeleted UCSF tumour is below 0.06. |

---

## 3. `models/` — classifier comparison, habitat vs CMDF

**What was tested.** Four feature sets: habitat M3 and M4, CMDF V and P+D. Each was trained with 9 classifiers:

- L2 logistic regression and elastic-net logistic regression
- RBF-SVM
- random forest and extra trees
- histogram gradient boosting
- KNN, Gaussian naive Bayes and an MLP

Validation was 5 × 5 repeated CV with every fitting step inside the training folds, and thresholds were chosen on
training data. This asks whether the gap between the approaches comes from the features or from the classifier.

### Top-level files

| File | What it is |
|---|---|
| `README.md` | Key observations and caveats. Read first. |
| `comparison_report.md` | All numbers. For UTSW and pooled: best classifier per set, mean over classifiers, the L2 logistic regression row, and the paired CMDF − habitat tests. |
| `model_comparison.py` | The script (`--repeats`, `--n-jobs`, `--sets`, `--out`). |
| `manifest.json` | Config, versions, runtime. |
| `run.log` | Verbose log (git-ignored). |

### Tables

| Table | What it shows | How to read it / key value |
|---|---|---|
| `tables/comparison_all_metrics.csv` | **Master table:** 72 rows = 2 settings × 4 feature sets × 9 classifiers. AUROC (CI), fold-mean AUROC, AUPRC, Brier, precision, recall, specificity, NPV, F1, balanced accuracy, accuracy, MCC, and the SD of precision and recall over repeats | Sort or filter it to answer any "which model is best for X" question. |
| `tables/comparison_paired_tests.csv` | Per classifier: V − M4, V − M3, P+D − M4, etc. Δ, NB CI, p, folds favouring CMDF, BH p within set | UTSW V − M4 is positive for 9/9 classifiers (+0.14 to +0.23); unadjusted p < 0.05 for 7/9; BH p ≈ 0.12. Consistent direction, too few patients to be certain. |
| `habitat/tables/metrics_by_classifier.csv`, `cmdf/tables/metrics_by_classifier.csv` | The master table split by approach (36 rows each) | Same columns. |
| `habitat/tables/fold_records.csv`, `cmdf/tables/fold_records.csv` | Per fold × repeat × classifier: AUROC, threshold, TP/FP/TN/FN, chosen hyper-parameters, run time | Raw material; use it to check hyper-parameter choices. |

**Mean over 9 classifiers, UTSW:**

| Set | AUROC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|
| V | 0.758 | 0.684 | 0.731 | 0.706 | 0.413 |
| P+D | 0.672 | 0.592 | 0.660 | 0.619 | 0.222 |
| M4 | 0.604 | 0.568 | 0.560 | 0.558 | 0.154 |
| M3 | 0.599 | 0.563 | 0.575 | 0.565 | 0.150 |

**Pooled:** V 0.813, P+D 0.811, M3 0.771, M4 0.764.

### Figures

| Figure | What it shows |
|---|---|
| `figures/comparison_{auroc,precision,recall,f1,mcc}_heatmap.png` | Rows = feature sets, columns = classifiers, cell = metric; one heatmap per metric, UTSW and pooled panels. A whole V row brighter than the habitat rows means the gap is in the features, not the classifier. |
| `figures/best_roc_UTSW.png`, `best_roc_pooled.png` | ROC curve of the best classifier for each feature set. Best-of-9 is optimistic; use it for shape, not for headline numbers. |
| `figures/precision_recall_UTSW.png`, `precision_recall_pooled.png` | Precision–recall curves of the same models, with prevalence as the dashed baseline. In the pooled set (29 % prevalence) this shows the price of each extra true positive. |
| `habitat/figures/*_heatmap.png`, `cmdf/figures/*_heatmap.png` | AUROC, precision and F1 heatmaps for one approach at a time. |

**Which number to quote.** Not the best-of-9 (e.g. V + gradient boosting 0.802), which is selected on the test
folds. Quote the pre-specified L2 logistic regression (UTSW V 0.735 here, 0.765 with 10 repeats in the CMDF study)
or the nine-classifier mean.

---

## 4. One-paragraph summary across all three folders

Habitat composition and spatial arrangement do not predict 1p/19q beyond noise. Patch-level cross-modal differences
add nothing to single-sequence radiomics, and real co-location is no better than random pairing. Voxel-level
cross-modal features (V) are the only imaging representation that beats a label-permutation null. They lead the
habitat features with every one of nine classifiers. They include an automated T2–FLAIR mismatch that reproduces
the known radiological sign, and an exploratory T1–T2 coupling difference that has not been reported before. The
original notebook protocol produces AUROC 0.75–0.89 on random labels, so its reported 0.898 should not be cited. All
of this rests on 74 single-centre patients (36 events). The next step is adding the 64 UTSW patients excluded for
missing segmentations, then validating externally.
