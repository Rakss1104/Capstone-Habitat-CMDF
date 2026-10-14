# Habitat vs CMDF — classifier comparison

`model_comparison.py` trains nine classifiers on each approach's feature sets under one leakage-free protocol and
reports discrimination, precision-type metrics and paired tests. Full numbers: `comparison_report.md`.

```
habitat/   README.md (per-classifier metrics), tables/metrics_by_classifier.csv, tables/fold_records.csv, figures/
cmdf/      same, for the CMDF feature sets
tables/    comparison_all_metrics.csv (every set × classifier × metric), comparison_paired_tests.csv
figures/   comparison_{auroc,precision,recall,f1,mcc}_heatmap.png, best_roc_{UTSW,pooled}.png, precision_recall_{UTSW,pooled}.png
comparison_report.md, manifest.json, run.log
```

**Feature sets.** Habitat: M3 = composition (5 features), M4 = composition + spatial (11). CMDF: V = voxel CMDF +
T2–FLAIR mismatch (45), P+D = patch radiomics + patch CMDF (1,872). Pooled sets add the cohort indicator.
**Classifiers.** L2 and elastic-net logistic regression, RBF-SVM, random forest, extra trees, histogram gradient
boosting, KNN, Gaussian NB, MLP; small inner-CV grids. **Validation.** 5 × repeated stratified 5-fold CV on patients;
habitat model, CMDF scaling, imputation, scaling, hyper-parameters and threshold all fitted on training folds only.
The threshold maximises balanced accuracy on inner out-of-fold predictions. Run: `python models/model_comparison.py`
(Windows venv `~/venvs/habitat_win` or WSL `~/venvs/habitat2`; ~50 min on 4 workers).

## Key observations

**UTSW (single centre, 74 patients, 36 codeleted; the clean comparison)**

| feature set | mean AUROC over 9 classifiers | mean precision (PPV) | mean recall | mean F1 | mean MCC |
|---|---|---|---|---|---|
| CMDF V | **0.758** | **0.684** | **0.731** | **0.706** | **0.413** |
| CMDF P+D | 0.672 | 0.592 | 0.660 | 0.619 | 0.222 |
| Habitat M4 | 0.604 | 0.568 | 0.560 | 0.558 | 0.154 |
| Habitat M3 | 0.599 | 0.563 | 0.575 | 0.565 | 0.150 |

- **V beats both habitat sets with every one of the nine classifiers** (V − M4: +0.14 to +0.23 AUROC). The
  unadjusted p is < 0.05 for seven of nine classifiers (gradient boosting p = 0.050), but none survives BH correction across all 27 tests
  (adjusted p ≈ 0.12). The direction is unanimous; the size is uncertain at n = 74.
- At the prevalence of 0.49, V's precision is ~0.68 (≈ 2 in 3 positive calls are correct); habitat's is ~0.56, barely
  above the prevalence.
- Classifier choice matters far less than feature set: the spread across classifiers within V is 0.69–0.80, while
  habitat M3/M4 never exceeds 0.65 with any classifier.
- Tree ensembles rescue P+D somewhat (random forest 0.733 vs 0.614 for L2 logistic regression); they ignore most of
  the 1,872 features. P+D is still below V with every classifier.

**Pooled (150 patients, 44 codeleted; cohort indicator included)**

- V again leads on average (0.813) and with the pre-specified L2 logistic regression (0.849, precision 0.57 at
  prevalence 0.29, NPV 0.88).
- P+D catches up here (mean 0.811; extra trees 0.855). Treat this with caution: within UTSW alone, P+D is far weaker,
  and single-sequence intensity features also encode scanner and site differences between UTSW and UCSF. V's lead is
  the same in both settings, which is what a real biological signal should look like.
- Habitat M3/M4 reach 0.76–0.80 pooled but only ~0.60 in UTSW: most of their pooled performance is cohort, not
  tumour biology (cohort alone gives 0.70).

**Caveats**
- Best-of-nine numbers (e.g. V + gradient boosting 0.802 in UTSW) are optimistic. Quote the pre-specified L2 logistic
  regression (V 0.735 here; 0.765 with 10 repeats in the CMDF study) or the nine-classifier mean.
- This comparison uses 5 repeats; the two studies used 10. Point estimates differ slightly (± 0.03).
- No external cohort. The UTSW → UCSF transport in the CMDF study had only 8 UCSF events.
