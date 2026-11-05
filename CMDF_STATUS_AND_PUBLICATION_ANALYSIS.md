# CMDF: approach, results, status and publication analysis

**Scope.** This document covers the cross-modal differential feature (CMDF) study only. The habitat study appears
only where CMDF reuses its cohort, preprocessing, folds and baseline models, or where the nine-classifier
comparison in `models/` tests CMDF features. **Date:** 2026-11-05. **Code state:** commit `8d05a72`.

**Sources.** `CMDF/PROTOCOL.md`, `CMDF/cmdf1p19q/*.py`, `CMDF/outputs/` (conclusion, INSIGHTS, manifest, run.log,
22 tables, 5 figures, 150 voxel-feature files), `models/` (CMDF rows), the two original notebooks, and a literature
check (sources at the end).

**New numbers in this document.** Some numbers were computed for this review from cached study data. They are
not part of the pre-specified analysis and are marked **⚑ post-hoc**. Everything else is quoted from
`CMDF/outputs/`. Section 8.2 says how to reproduce the post-hoc checks.

---

## Contents

1. [Executive summary](#1-executive-summary)
2. [The CMDF approach](#2-the-cmdf-approach)
3. [Results of the pre-specified analysis](#3-results-of-the-pre-specified-analysis)
4. [Critical appraisal and post-hoc checks](#4-critical-appraisal-and-post-hoc-checks)
5. [Where the project stands](#5-where-the-project-stands)
6. [What remains: prioritised work plan](#6-what-remains-prioritised-work-plan)
7. [Publication analysis](#7-publication-analysis)
8. [Appendices](#8-appendices)

---

## 1. Executive summary

### Bottom line

The CMDF idea was that *co-located* disagreement between MRI sequences carries information about 1p/19q
codeletion beyond each sequence read alone. It was tested in a leakage-free, pre-specified study on 74 UTSW
IDH-mutant gliomas (36 codeleted), with 76 UCSF patients (8 codeleted) as a second cohort. The results split by
the scale at which the disagreement is measured:

1. **Patch-level CMDF failed, decisively.** Cross-sequence differences of 93 texture features on 32-mm patches
   added nothing to single-sequence patch radiomics (ΔAUROC −0.046, Nadeau–Bengio 95 % CI −0.112 to +0.020;
   8/50 folds positive). Real co-location did no better than randomly re-paired patches (pairing-null p = 1.00).
   The result held with a random forest, in patients with at least 4 patches, and with every sequence pair left
   out in turn. **The pre-specified primary hypothesis H1 is not supported.** The CI is tight enough to rule
   out any gain above about +0.02 AUROC, so this is an informative negative, not just an underpowered one.
2. **Voxel-level CMDF (block V, 45 features) is the only imaging representation with a clear signal.** UTSW
   AUROC 0.765 (95 % CI 0.651–0.871). Label-permutation p = 0.005, the floor for 200 permutations. It is the
   best-calibrated imaging model (slope 0.77). With all nine classifiers in `models/` it beats the habitat
   features and patch CMDF. V was pre-specified, so this is not a post-hoc discovery. **But its advantage over
   simpler comparators is not statistically established.** V − age/sex was +0.085 (BH p = 0.46). Post-hoc ⚑:
   V − P was +0.109 (p = 0.13), V − whole-tumour radiomics +0.085 (p = 0.18), and V − its own 3 mismatch
   features +0.044 (p = 0.47).

   A post-hoc ⚑ decomposition (Section 4.1) shows the 12 *purely co-location* features alone reach 0.754, so
   the signal is not an artefact of mean-intensity differences. But V adds nothing to single-sequence
   whole-tumour intensity histograms (FO 0.713; FO+V − FO = +0.005). The voxel-level cross-modal information is
   *shared with* single-sequence information, not *additional to* it.
3. **Pre-specified H2 is confirmed.** The quantitative T2–FLAIR mismatch fraction is lower in codeleted tumours
   in UTSW (Cliff's δ −0.34, p = 0.012). Post-hoc ⚑: the effect is the same direction in UCSF and significant in
   the pooled set once cohort is adjusted for (OR 0.44 per SD, p = 0.006). It matches the known radiological
   sign.
4. **The methods demonstration is robust and publishable as it stands.** The original notebooks' evaluation
   protocol gives AUROC 0.75 on average, and up to 0.885, on *randomly permuted labels* in this cohort. The
   originally reported 0.898 is therefore not credible evidence of signal.

**Net verdict on the CMDF hypothesis.** Co-located cross-sequence disagreement does not demonstrably add
information beyond reading each sequence alone. That holds at patch scale (pre-specified, strong) and at voxel
scale (post-hoc ⚑). Two things survive: a compact, interpretable voxel-coupling representation that carries the
1p/19q signal *as well as* single-sequence intensity distributions, and an automated mismatch measure that
replicates a known radiological sign.

### Scoreboard

| Question | Pre-specified? | Result | Evidence strength |
|---|---|---|---|
| H1: patch CMDF adds to single-sequence patch radiomics (UTSW) | Primary | **No.** Δ −0.046 (−0.112 to +0.020) | Strong negative |
| Co-location matters at patch scale (pairing null) | Primary (part of H1) | **No.** p = 1.00 | Strong negative |
| V carries signal (label permutation) | Yes | **Yes.** p = 0.005 | Strong |
| V beats age + sex (S4) | Yes | Δ +0.085, BH p = 0.46 | Not shown (power) |
| V + cohort beats cohort alone, pooled (S7) | Yes | Δ +0.115, BH p = 0.011 | Significant, but a weak comparator |
| V beats other imaging representations | No ⚑ | Only V − P+D significant (p = 0.024, unadjusted) | Weak |
| Voxel analogue of H1: V adds to single-sequence intensity histograms (FO) | No ⚑ | FO+V − FO +0.005 (−0.093 to 0.104); co-location-only Vc 0.754 vs FO 0.713 | Not shown: information is shared |
| H2: mismatch fraction lower in codeleted tumours | Yes (directional) | UTSW p = 0.012; pooled p = 0.08 crude, p = 0.006 cohort-adjusted ⚑ | Moderate to strong |
| T1–T2 voxel coupling differs by label | No (exploratory) | UTSW δ +0.61, q = 0.0003; UCSF δ +0.19, p = 0.39 ⚑ | Hypothesis-generating |
| FLAIR core − rim higher in codeleted tumours (pooled) | No (exploratory) | Gone after cohort adjustment (p = 0.18) ⚑ | **Artefact: withdraw** |
| Transport UTSW → UCSF | Descriptive | Ranking kept (V 0.737); V's threshold fails (specificity 0.15) | Descriptive only |
| The original protocol's numbers are optimistic | Yes (methods arm) | Null mean 0.745, maximum 0.885 | Strong |

### Where we stand

- **Engineering and analysis are complete and reproducible.** There is a pre-specified protocol with a SHA-256,
  7/7 automated integrity checks pass, 6/6 unit tests pass (re-run for this review), and a WSL → Windows rerun
  reproduced the stage-5 tables byte for byte.
- **Scientifically**, the project has one solid negative result (patch CMDF), one solid methods result (protocol
  optimism) and one promising but under-validated positive result (voxel CMDF and the quantitative mismatch).
- **The positive result is the most interesting part, and it has three open problems.** (a) It is not shown to
  beat simple comparators. (b) Post-hoc, its information turns out to be *shared with* single-sequence intensity
  histograms rather than additional to them (Section 4.1), which weakens the "cross-modal" claim. (c) It is
  strongly site-dependent, so its threshold does not transport. External validation has not been done.

### What remains (top six)

1. **Lock the V model** (features, estimator, threshold, normalisation). Register an external-validation
   protocol publicly (OSF) *before* touching new data.
2. **External validation** on an independent IDH-mutant cohort with four sequences and 1p/19q status (EGD;
   TCGA-LGG with BraTS labels).
3. **Recover the 64 UTSW patients excluded only for a missing segmentation** (26 codeleted). One automatic
   segmentation tool for every cohort would also remove the cross-site mask differences.
4. **Harmonisation and recalibration** for V (reference-tissue normalisation or ComBat; intercept recalibration),
   since V's absolute values shift strongly between sites.
5. **Handle the Section 4 checks correctly.** They have now been seen, so report them as labelled post-hoc
   analyses and make them pre-registered endpoints of the external phase. Pre-specify, *before running*, the
   checks not yet done: grade-4 and label-discordance exclusions, and white-matter-referenced normalisation.
6. **Correct over-statements in the current docs** (Section 4.9):
   - the pooled FLAIR core − rim "biology" is an artefact;
   - V's information is shared with single-sequence intensity, not additional to it;
   - prior automated mismatch work exists;
   - "pre-registered" should read "pre-specified" unless an external time-stamp exists.

### Publication verdict

| Paper option | Readiness today | What it needs |
|---|---|---|
| **A. Methods / negative-result paper**: patch CMDF fails under a pre-specified test, and a common radiomics protocol yields AUROC 0.75–0.89 on random labels | **~80 %** | Writing, reporting checklists, 2–3 sensitivity analyses |
| **B. Positive paper**: compact voxel intensity/coupling features and quantitative T2–FLAIR mismatch predict 1p/19q in IDH-mutant glioma | **~35 %** | External validation, harmonisation, a locked model, a clinical comparator. Must drop the claim that cross-modal information *adds* (4.1) |
| **C. Combined, honestly framed** (recommended if external data can't be obtained in time) | **~70 %** | As A, plus the Section 4 analyses written up as labelled post-hoc results with explicit limits |

These readiness figures are judgement calls, not measurements. Recommended path: Section 7.2.

---

## 2. The CMDF approach

### 2.1 Scientific rationale

1p/19q codeletion defines oligodendroglioma among IDH-mutant gliomas (WHO CNS5, 2021), with its prognostic and
treatment implications. The best-known imaging marker is the **T2–FLAIR mismatch sign**. In it, tumour is
homogeneously bright on T2 but relatively suppressed on FLAIR, apart from a hyperintense rim. The sign is highly
specific for IDH-mutant, 1p/19q-*intact* astrocytoma but has low sensitivity.

The mismatch sign is a **cross-modal disagreement**: two sequences disagree at the same location. CMDF
generalises this from one visual sign to *every sequence pair* and *every radiomic feature*. It then asks whether
the disagreement itself, as opposed to each sequence's own appearance, carries information.

### 2.2 Version 1: the original notebooks (superseded)

`CMDF/feature_extraction_cmdf (1).ipynb` and `CMDF/train_test_evaluate_cmdf_final.ipynb` (Google Colab, kept
unchanged for the record).

| Aspect | Original design |
|---|---|
| Data | UTSW only; 62 patients with a class split of 52 vs 10; IDH-wildtype tumours not excluded |
| Sequences | T1-Gd vs T2 only |
| Patches | 16³ voxels, ≥ 50 % tumour; aggregated mean, SD, min, max and median per patient |
| CMDF metrics | Variance `Var([a, b])` (equivalent to (a − b)²/4, so redundant with L2), L2 distance, cosine on raw features (dominated by Energy-scale features), outlier deviation (σ estimated over **all** patients, which leaks test data) |
| Features | 1,212 columns (930 patch aggregates + 282 CMDF) |
| Pipeline | Information-gain top-30 selection, VAE minority oversampling, then 1D-CNN, KNN, DT, GNB, RF and gated-attention MIL |
| Validation | **One** stratified 5-fold split (seed 42); best model chosen on the test folds; final export fitted on all data |
| Reported | Patch+CMDF best = RF, AUROC 0.898 ± 0.116, F1 0.936, accuracy 0.889. Baseline RF 0.877; patch MIL 0.854 |

Why the original numbers can't be cited:

- **Selection on the test folds.** The best of six models was picked by test-fold AUROC.
- **One split of 62 patients with 10 minority cases.** Fold AUROCs ran from 0.32 to 1.00 and SDs were 0.12–0.29.
- **Leakage.** The outlier deviation and the final export saw all patients.
- **Uninformative metrics at this prevalence.** With 84 % in the majority class, accuracy and F1 say little:
  the 1D-CNN predicted *every* patient positive (accuracy 0.8385, which equals the prevalence; recall 1.000)
  and still scored F1 0.912.
- **Confounded task.** Including IDH-wildtype tumours, which are never codeleted, partly turns the task into
  IDH prediction.

### 2.3 Version 2: the rebuilt, pre-specified study (current)

Package `CMDF/cmdf1p19q/`. Protocol `CMDF/PROTOCOL.md` (SHA-256 `97eab319…`, recorded in `manifest.json`), written
2026-10-13 *before any CMDF feature or result was computed*. At that point the habitat study's cohort and
baselines were already known.

#### Data and preprocessing (inherited unchanged from the habitat study)

- **Cohorts:** UTSW glioma dataset and UCSF-PDGM, both public, four sequences each: T1, T1-Gd, T2, FLAIR.
- **Images:** N4 bias correction, resampling to 1 mm isotropic on the segmentation grid, and per-sequence
  **whole-brain z-scoring** (non-zero voxels).
- **Mask:** whole abnormality, meaning every non-zero segmentation label (enhancing, non-enhancing, necrosis and
  oedema together).
- **Patches:** 32³ voxels, stride 16, ≥ 25 % tumour. Median 26 patches per patient (IQR 12–46); 2 patients have
  a single patch.
- **Patch features:** 93 PyRadiomics features per patch per sequence (first order, GLCM, GLRLM, GLSZM, GLDM,
  NGTDM; bin width 0.25 z-units). These are reused from the habitat cache (fingerprint `78c614a2dc75`), so no
  features were re-extracted.

#### Feature blocks

| Block | Definition | Columns | Fitted on data? |
|---|---|---|---|
| **P** single-sequence patch radiomics | per sequence × feature: mean and SD over the patient's patches | 744 | no |
| **D** patch CMDF | per pair (6) × feature: Δ = A − B **on the same patch**; mean and SD of Δ over patches. Plus per pair the cosine of the A and B feature profiles per patch (after per-sequence standardisation), mean and SD | 1,128 | cosine scaling only, **inside each training fold** |
| **D_T2FL** | D restricted to the T2–FLAIR pair | 188 | as D |
| **V** voxel CMDF | per pair, over tumour voxels, d = z_A − z_B: mean, SD, P10, P50, P90 (30); Pearson and Spearman voxel correlation of z_A and z_B (12); plus MM (3) | 45 | no |
| **MM** quantitative T2–FLAIR mismatch | fraction of tumour voxels with z_T2 − z_FLAIR > 1; FLAIR core − rim; T2 core − rim (core = normalised depth ≥ 0.5; rim = ≤ 3 mm from the boundary) | 3 | no |
| M0, M1_clin, M1, M2 | cohort; age + sex; + log volume; whole-tumour radiomics (386) | | inherited from the habitat study |

Note that V is not purely "cross-modal". `d_mean` (and to a large extent `d_p50`) equals the difference of the
two sequences' mean intensities, so it is fully determined by single-sequence information. Only `d_sd`, the
spread percentiles and the two correlations depend on voxel co-location. Section 4.1 tests what this means for
the result.

#### Modelling and validation

- **Estimator:** L2 logistic regression (median imputation, standardisation, class-balanced). C is tuned by
  inner 5-fold CV over {0.01, 0.03, 0.1, 0.3, 1, 3}.
- **Threshold:** maximises balanced accuracy on inner out-of-fold predictions of the *training* fold.
- **Outer loop:** 10 × repeated stratified 5-fold (strata = cohort × label), seed 20261012. These are the same
  folds as the habitat study.
- **Paired comparisons:** Nadeau–Bengio corrected t-test on the 50 fold-level ΔAUROCs, plus a patient bootstrap
  of the repeat-averaged out-of-fold ΔAUROC.
- **Multiplicity:** Benjamini–Hochberg across S1–S7.

#### Hypotheses

- **H1 (primary), UTSW:** AUROC(P+D) > AUROC(P). It is supported only if the NB 95 % CI lower bound is > 0
  **and** the pairing null gives p < 0.05.
- **Pairing null:** per patient, the patch order of T1-Gd, T2 and FLAIR is permuted independently (T1 fixed).
  Each sequence's own distribution, and so P, is unchanged; only co-location is destroyed. 100 replicates on
  the first-repeat folds.
- **S1–S7 (secondary):** D − P; P+D − M2; D_T2FL − M1_clin; V − M1_clin; M2+V − M2 (all UTSW);
  P+D+cohort − P+cohort; V+cohort − M0 (both pooled).
- **H2 (directional):** mismatch fraction is lower in codeleted tumours (Cliff's δ < 0), in UTSW and pooled.

#### Further pre-specified analyses

- **Label permutation:** within cohort, label-independent folds, 200 permutations, for P+D and V.
- **Leave-one-cohort-out:** UTSW → UCSF only. UCSF → UTSW is not run because there are fewer than 20 training
  events.
- **Robustness:** random forest; patients with ≥ 4 patches; leave-one-pair-out ablation of D.
- **Methodological arm:** the original protocol re-run on the same UTSW P+D features, observed and on 50
  label permutations.

#### Pipeline

```
 metadata (UTSW tsv, UCSF csv) ─► cohort: IDH-mutant, 1p/19q resolved, follow-ups removed ─► QC ─► 74 UTSW + 76 UCSF
                                                                                               │
 images ─► N4 ─► 1 mm ─► brain z-score ──┬─► 32-mm patches ─► 93 radiomics × 4 seq (habitat cache) ─► P, D, cos (fold-fitted)
                                         └─► tumour voxels ─► d = z_A − z_B, corr(z_A, z_B), mismatch, core/rim ─► V, MM (cached)
                                                                                               │
                         10 × 5-fold CV (cohort × label strata) · inner-CV C · training-only threshold
                                                                                               │
      H1 + pairing null · S1–S7 (BH) · H2 · label permutation · LOCO · robustness · calibration · original-protocol arm
```

### 2.4 Engineering and reproducibility state

| Item | State |
|---|---|
| Protocol integrity | SHA-256 of `PROTOCOL.md` stored in the manifest; resuming from a checkpoint refuses a changed protocol |
| Automated checks | 7/7 PASS: cohort sizes, block sizes, the pairing null keeps marginals, train/test disjoint and cosine scaling fitted on training data in all 100 folds, permuted runs use the same folds |
| Unit tests | `CMDF/tests/test_features.py`: 6/6 pass (re-run 2026-11-05 on the Windows venv) |
| Determinism | Fixed seeds; per-patient voxel cache keyed by a settings fingerprint (`e5937a08a21f`) |
| Cross-platform rerun | The first WSL run was killed at stage 6. The Windows rerun reproduced its stage-5 tables byte for byte (`outputs/attempt1_killed/`) |
| Runtime (Windows, caches warm) | CV 28 min, null tests 14 min, robustness 9 min, original-protocol arm 9 min: about 1 h in total |
| Environment | Python 3.12.6, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, SimpleITK 2.5.6. PyRadiomics is *not* installed on Windows, so the patch features depend on the habitat cache built under WSL |
| Data handling | Patient data and caches are git-ignored; per-patient voxel features (de-identified numbers) are committed |

---

## 3. Results of the pre-specified analysis

All values are from `CMDF/outputs/conclusion.md` and `CMDF/outputs/tables/`. AUROC is the repeat-averaged
out-of-fold value with a patient-bootstrap 95 % CI.

### 3.1 Cohort and patient flow

| Step | UTSW (codeleted) | UCSF (codeleted) |
|---|---|---|
| Rows in metadata | 625 (63) | 501 (9) |
| Unique patients (UCSF follow-ups removed) | 625 | 495 |
| IDH-mutant | 176 | 103 |
| 1p/19q resolved (primary label) | **140 (63)** | **93 (9)** |
| Excluded: imaging QC | 64 (26), **all missing only the segmentation** | 13 (0) |
| Excluded: no valid patch | 2 (1) | 4 (1) |
| **Analysis set** | **74 (36): prevalence 0.49** | **76 (8): prevalence 0.11** |

Composition of the analysis set ⚑ (from the cached study table):

| | UTSW non-codeleted / codeleted | UCSF non-codeleted / codeleted |
|---|---|---|
| Grade 2 / 3 / 4 | 19/25 · 14/11 · **5/0** | 28/8 · 23/0 · **17/0** |
| Field strength | 1.5 T 46, 3 T 27, 1.16 T 1 (balanced across labels) | standardised 3 T protocol |
| Median volume (ml) | 54.9 vs 62.3 (p = 0.71) | **74.3 vs 26.1 (p = 0.0003)** |
| Histology notes | 18 oligoastrocytomas (17 non-codeleted); 4 "oligodendroglioma but intact"; 2 "astrocytoma but codeleted" (study-wide flags) | WHO 2021 integrated diagnoses |
| Segmentation labels present | 7 different label sets (e.g. `0 3`, `0 1 2`, `0 1 2 3 4`) | BraTS-style (`0 2`, `0 1 2 4`, `0 2 4`) |

Two consequences matter later:

- Codeleted tumours are never grade 4, so grade-4 cases are label-informative "easy negatives". There are 5 in
  UTSW and 17 in UCSF.
- In UCSF, codeletion goes with much smaller tumours; in UTSW it doesn't. Any volume-dependent feature will
  therefore look label-related in pooled analyses.

### 3.2 Discrimination of every model

| Set | Model | AUROC (95 % CI) | AUROC per repeat, mean [min–max] | AUPRC | Brier |
|---|---|---|---|---|---|
| UTSW | M1_clin (age + sex) | 0.659 (0.527–0.788) | 0.651 [0.617–0.700] | 0.689 | 0.237 |
| UTSW | M1 (+ log volume) | 0.656 (0.520–0.787) | 0.645 [0.590–0.686] | 0.689 | 0.233 |
| UTSW | M2 whole-tumour radiomics | 0.692 (0.566–0.811) | 0.656 [0.614–0.700] | 0.621 | 0.231 |
| UTSW | **P** | 0.662 (0.533–0.780) | 0.639 [0.586–0.697] | 0.621 | 0.248 |
| UTSW | **D** | 0.610 (0.471–0.737) | 0.610 [0.569–0.651] | 0.527 | 0.270 |
| UTSW | **P+D** | 0.628 (0.496–0.757) | 0.600 [0.544–0.659] | 0.562 | 0.278 |
| UTSW | **D_T2FL** | 0.675 (0.550–0.799) | 0.662 [0.606–0.719] | 0.622 | 0.229 |
| UTSW | **V** | **0.765 (0.651–0.871)** | 0.744 [0.692–0.798] | 0.719 | **0.199** |
| UTSW | **MM** (3 features) | 0.703 (0.576–0.820) | 0.695 [0.662–0.730] | 0.656 | 0.222 |
| UTSW | M2+V | 0.708 (0.583–0.825) | 0.679 [0.648–0.719] | 0.651 | 0.224 |
| pooled | M0 (cohort only) | 0.696 (0.596–0.788) | 0.721 | 0.456 | 0.228 |
| pooled | M2 | 0.828 (0.762–0.893) | 0.807 [0.771–0.841] | 0.589 | 0.161 |
| pooled | P | 0.802 (0.724–0.877) | 0.790 [0.745–0.826] | 0.612 | 0.169 |
| pooled | P+D | 0.776 (0.694–0.855) | 0.759 [0.709–0.792] | 0.556 | 0.182 |
| pooled | V | 0.800 (0.716–0.879) | 0.792 [0.748–0.817] | 0.658 | 0.168 |
| pooled | **V+cohort** | **0.851 (0.783–0.916)** | 0.841 [0.798–0.858] | **0.741** | **0.147** |
| pooled | M2+cohort | 0.830 (0.763–0.896) | 0.814 [0.781–0.843] | 0.590 | 0.159 |

Threshold metrics for V in UTSW, with the threshold chosen on training data only: sensitivity 0.74,
specificity 0.66, precision 0.68, F1 0.71, balanced accuracy 0.70.

Pooled models scored within each cohort: V+cohort gives UTSW 0.790 and UCSF 0.801, so it isn't only learning the
site. M2+cohort gives UTSW 0.695 and UCSF 0.871.

### 3.3 Primary hypothesis H1: P+D vs P (UTSW)

| | Value |
|---|---|
| ΔAUROC, fold mean | **−0.046** (NB 95 % CI −0.112 to +0.020, p = 0.171) |
| ΔAUROC, repeat-averaged OOF | −0.034 (bootstrap 95 % CI −0.081 to +0.013) |
| Folds with Δ > 0 | **8/50** |
| Pairing null | observed −0.046; null mean −0.008, 95th percentile +0.010; **p = 1.00** |
| Verdict | **H1 not supported** (both pre-specified conditions fail) |

### 3.4 How to read the pairing null (p = 1.00)

The observed Δ lies *below* all 100 shuffled replicates. Real co-location made P+D slightly worse than random
co-location did. This should **not** be read as "co-location is anti-informative". The mechanics explain it:

- `D_mean` is the same under shuffling, because a mean of differences equals a difference of means, a linear
  function of P.
- With co-location destroyed, `D_sd` collapses to roughly √(SD_A² + SD_B²), a near-deterministic function of
  P's SD columns. Shuffled D is therefore **almost redundant with P**, and a penalised model loses little by
  adding it.
- Real `D_sd` adds the covariance term (−2·cov(A, B)), which makes about 560 genuinely new SD columns (6 pairs × 93 features, plus the
  cosine SDs). They are new,
  noisy and label-unrelated, so they dilute P more.

The correct reading: *at patch scale, the co-location-specific part of the signal is noise with respect to
1p/19q*. Put this explanation in the paper. Reviewers will otherwise ask why a "null" beats the observed value.

### 3.5 Secondary comparisons (BH across S1–S7)

| ID | Set | Comparison | Δ fold mean (NB 95 % CI) | Folds Δ > 0 | BH p |
|---|---|---|---|---|---|
| S1 | UTSW | D − P | −0.040 (−0.128 to 0.048) | 12/50 | 0.460 |
| S2 | UTSW | P+D − M2 | −0.070 (−0.172 to 0.032) | 13/50 | 0.411 |
| S3 | UTSW | D_T2FL − M1_clin | +0.002 (−0.167 to 0.171) | 27/50 | 0.978 |
| S4 | UTSW | **V − M1_clin** | **+0.085** (−0.093 to 0.263) | 36/50 | 0.460 |
| S5 | UTSW | M2+V − M2 | +0.022 (−0.029 to 0.073) | 33/50 | 0.460 |
| S6 | pooled | P+D+cohort − P+cohort | −0.033 (−0.068 to 0.002) | 6/50 | 0.213 |
| S7 | pooled | **V+cohort − M0** | **+0.115** (0.046 to 0.184) | 46/50 | **0.011** |

Only S7 survives, and M0 (cohort only) is the weakest possible comparator. The S4 interval is ±0.18 wide, which
reflects a power limit, not evidence of no effect.

### 3.6 H2: T2–FLAIR mismatch fraction

| Set | Median codeleted vs non-codeleted | Cliff's δ (95 % CI) | p |
|---|---|---|---|
| UTSW | 0.024 (n = 36) vs 0.059 (n = 38) | **−0.342 (−0.594 to −0.086)** | **0.012** |
| pooled | 0.012 (n = 44) vs 0.033 (n = 106) | −0.181 (−0.361 to 0.015) | 0.081 |

The UTSW part of H2 is confirmed. The pooled part misses p < 0.05 crudely, but becomes significant once cohort is
adjusted for (Section 4.4). Figure `10_t2flair_mismatch.png` shows a bimodal non-codeleted distribution: a subset
of astrocytomas has very high mismatch (> 0.25), and every codeleted UCSF tumour is below 0.06. That is exactly
the profile of the visual sign, which is specific but not sensitive.

### 3.7 Label permutation (UTSW, 200 permutations, cohort-stratified folds)

| Model | Observed AUROC | Null mean | Null 95th percentile | p |
|---|---|---|---|---|
| P+D | 0.596 | 0.494 | 0.641 | 0.129 |
| **V** | **0.702** | 0.501 | 0.633 | **0.005** (floor) |

The observed AUROCs here (0.596 and 0.702) come from one repeat on cohort-stratified folds, which is why they
differ from Section 3.2.

### 3.8 Transport: train UTSW (74/36) → test UCSF (76/8). Descriptive only

| Model | AUROC (95 % CI) | Sensitivity / specificity at the UTSW threshold | Brier |
|---|---|---|---|
| M1_clin | 0.585 (0.346–0.824) | 0.63 / 0.66 | 0.237 |
| M2 | 0.757 (0.553–0.906) | 0.88 / 0.60 | 0.331 |
| P | 0.759 (0.583–0.899) | 0.88 / 0.51 | 0.231 |
| D | 0.772 (0.599–0.899) | 0.88 / 0.43 | 0.274 |
| P+D | 0.794 (0.597–0.936) | 0.88 / 0.65 | 0.267 |
| **V** | 0.737 (0.559–0.869) | **1.00 / 0.15** (TP/FP/TN/FN 8/58/10/0) | **0.398** |

Every imaging model keeps some ranking ability across sites. V has the worst *calibration* transport: its
threshold calls 85 % of UCSF non-codeleted tumours "codeleted" (58/68). Section 4.5 explains why. With 8 events, every
CI spans about 0.35, so these models can't be ranked.

### 3.9 Robustness, ablation, calibration

**Robustness of H1.** Random forest: Δ −0.008 (−0.056 to 0.040). Patients with ≥ 4 patches (n = 72):
Δ −0.030 (−0.105 to 0.045). The null holds in both.

**Pair ablation.** Leave one pair out of D; this is the change in P+D AUROC:

| Pair dropped | T1–T1CE | T1–T2 | T1–FLAIR | T1CE–T2 | T1CE–FLAIR | T2–FLAIR |
|---|---|---|---|---|---|---|
| ΔAUROC | +0.000 | +0.001 | −0.011 | +0.007 | −0.002 | **−0.018** |

T2–FLAIR matters most, which fits the rationale, but no pair matters much.

**Calibration (UTSW, repeat-averaged OOF):**

| Model | Slope | CITL | Brier |
|---|---|---|---|
| M1_clin | 1.90 | −0.04 | 0.237 |
| P | 0.38 | −0.07 | 0.248 |
| P+D | 0.23 | −0.10 | 0.278 |
| D_T2FL | 0.69 | −0.10 | 0.229 |
| **V** | **0.77** | −0.06 | **0.199** |
| **MM** | **1.01** | −0.05 | 0.222 |
| M2+V | 0.56 | −0.12 | 0.224 |

The high-dimensional blocks are badly overfit (slopes 0.2–0.4). V is mildly overfit. The 3-feature MM is almost
perfectly calibrated.

### 3.10 Methodological arm: the original protocol on the same UTSW P+D features

| Variant | Best classifier, observed AUROC | Same protocol on 50 permuted-label runs (mean / 95th percentile / max) | p | Leakage-free AUROC | Optimism |
|---|---|---|---|---|---|
| notebook (selection inside folds, best of 5, one split) | GNB 0.712 ± 0.067 | 0.588 / 0.698 / 0.751 | 0.059 | 0.628 | +0.084 |
| select_all (scaler + top-30 MI on all patients first) | RF 0.827 ± 0.067 | **0.745 / 0.840 / 0.885** | 0.118 | 0.628 | **+0.199** |

Picking the "best" classifier changes with the variant (GNB vs RF), a sign of selection noise. The VAE
oversampling, 1D-CNN and attention-MIL arms were not replicated because TensorFlow is not installed; this is
logged as a deviation.

**Caveat for the paper.** These null distributions come from *this* cohort's features (74 IDH-mutant
patients, 1,872 features), not from the original 62-patient data. The defensible statement is: "the original
protocol produces AUROC up to 0.885 on random labels in a comparable cohort, so a reported 0.898 is not
evidence of signal". It does not prove the original model was worthless. With only 50 permutations the p-value
floor is 0.02. Use 200 or more for the paper.

### 3.11 Classifier-agnostic check (`models/`: 9 classifiers, 5 × 5 CV, UTSW)

| Feature set | Mean AUROC over 9 classifiers | Range | L2 logistic regression | Mean MCC |
|---|---|---|---|---|
| **V** | **0.758** | 0.69–0.80 | 0.735 | **0.413** |
| P+D | 0.672 | 0.61–0.73 | 0.614 | 0.222 |

V ranks above P+D with every classifier. Tree ensembles partly rescue P+D (random forest 0.733) because they
ignore most of its 1,872 columns. V's lead is therefore not an artefact of the estimator. Best-of-nine numbers
(e.g. V + gradient boosting 0.802) are selected on test folds and must not be quoted as headline numbers.

### 3.12 Exploratory voxel-feature univariate (UTSW, 45 features, BH within set)

Strongest features (all q < 0.01):

| Feature | Codeleted vs non-codeleted median | Cliff's δ | q |
|---|---|---|---|
| T1–T2 Spearman coupling | −0.48 vs −0.76 | +0.61 | 0.0003 |
| T1–T2 Pearson coupling | −0.50 vs −0.77 | +0.57 | 0.0004 |
| T1−T2 difference SD | 1.23 vs 1.70 | −0.57 | 0.0004 |
| T1−T2 difference P10 | −2.84 vs −4.21 | +0.54 | 0.0008 |
| T1CE−T2 difference P10 | −3.02 vs −4.47 | +0.52 | 0.0010 |
| T1−T2 difference mean | −1.09 vs −2.05 | +0.51 | 0.0012 |

The **T1–T2 / T1CE–T2 family dominates.** Non-codeleted tumours are darker on T1 relative to T2, more tightly
anti-coupled, and more spread out. That reads as a more uniform "fluid-like" lesion. These features are
mathematically linked: Var(d) = Var_A + Var_B − 2·Cov, so stronger anti-correlation means a larger difference
SD. They are **one finding, not six.**

---

## 4. Critical appraisal and post-hoc checks

Everything in this section is **⚑ post-hoc**. It was computed for this review from cached data with the study's
own code paths (same base table, folds, estimator and statistics). None of it is confirmatory. Because these
results have now been seen, they can't be "pre-specified" retroactively. Report them as labelled post-hoc
analyses and confirm them as pre-registered endpoints in the external phase.

### 4.1 Is V's signal actually *cross-modal*?

**Why it matters.** The CMDF hypothesis is about co-located disagreement. Of V's 45 features, `d_mean` and
`d_p50` are almost pure single-sequence information (differences of each sequence's mean or median intensity).
If V's performance came from those, the result would be "whole-tumour intensity levels predict 1p/19q": true,
but not CMDF. A reviewer will ask this first.

**Test.** Same UTSW folds and estimator. Feature sets:
- **FO:** single-sequence whole-tumour first-order statistics from the same z-scored images: mean, median, P10,
  P90, variance, skewness, kurtosis and IQR × 4 sequences = 32 features. This is the marginal-only comparator.
- **Vd:** V's 30 difference-distribution features.
- **Vc:** V's 12 voxel-coupling features (Pearson and Spearman), which are purely co-location.
- **Vsd:** Vc plus the 6 difference SDs, i.e. every co-location-dependent feature.

**Result ⚑** (UTSW, 10 × 5 CV, same folds and estimator as the study). The V row reproduces the study's 0.765
exactly, which confirms that the post-hoc setup matches the pipeline.

| Feature set | Columns | What it contains | AUROC (95 % CI) | Per repeat, mean [min–max] | Brier |
|---|---|---|---|---|---|
| V | 45 | everything | 0.765 (0.651–0.871) | 0.744 [0.692–0.798] | 0.199 |
| **FO** | 32 | **single-sequence only** (whole-tumour histograms) | 0.713 (0.586–0.832) | 0.702 [0.641–0.740] | 0.217 |
| **Vc** | 12 | **co-location only** (voxel correlations) | **0.754 (0.632–0.863)** | 0.742 [0.675–0.779] | 0.204 |
| Vsd | 18 | co-location-dependent (correlations + difference SDs) | 0.787 (0.672–0.890) | 0.766 [0.700–0.807] | 0.190 |
| Vd | 30 | difference distributions | 0.787 (0.672–0.889) | 0.756 [0.692–0.828] | 0.193 |
| MM | 3 | mismatch block | 0.703 (0.576–0.820) | 0.695 [0.662–0.730] | 0.222 |
| FO+V | 77 | both | 0.733 (0.609–0.846) | 0.719 [0.670–0.776] | 0.208 |
| FO+Vc | 44 | both | 0.743 (0.617–0.857) | 0.722 [0.666–0.745] | 0.205 |
| V+clin | 47 | V + age + sex | 0.761 (0.640–0.868) | 0.742 [0.681–0.786] | 0.198 |
| V+vol | 46 | V + log volume | 0.765 (0.648–0.872) | 0.741 [0.684–0.784] | 0.202 |

| Paired comparison (same 50 folds) | Δ fold mean | NB 95 % CI | p | Folds Δ > 0 |
|---|---|---|---|---|
| V − FO | +0.033 | −0.084 to 0.150 | 0.57 | 28/50 |
| Vc − FO | +0.031 | −0.097 to 0.159 | 0.63 | 24/50 |
| **FO+V − FO** (does cross-modal information *add*?) | **+0.005** | −0.093 to 0.104 | 0.92 | 27/50 |
| FO+Vc − FO | +0.008 | −0.086 to 0.101 | 0.87 | 27/50 |
| V+clin − V | −0.005 | −0.065 to 0.055 | 0.87 | 24/50 |
| V+vol − V | −0.007 | −0.046 to 0.033 | 0.73 | 13/50 |

**Reading.**

1. **V's signal is not an artefact of mean-intensity differences.** Twelve pure co-location features (Vc)
   contain no information about any single sequence's level. Pearson correlation is invariant to rescaling
   either sequence. Yet Vc alone reaches 0.754, essentially V's performance, with a quarter of the columns.
2. **But cross-modal information does not add to single-sequence information.** Single-sequence whole-tumour
   histograms (FO) already reach 0.713. Adding all of V gains +0.005, and adding Vc gains +0.008. The two
   descriptors carry **overlapping** information. That is plausible here: a "fluid-like" astrocytoma is at
   once very bright on T2, very dark on T1 *and* tightly anti-coupled, so the joint distribution is largely
   predictable from the two marginals. This is the **voxel-scale analogue of H1**, and it is not demonstrated
   either.
3. **V subsumes age.** V+clin − V is −0.005, consistent with several V features correlating with age (4.3).
   V's increment over age + sex (S4, +0.085) and age's increment over V (≈ 0) are two sides of one overlap.
4. **Don't quote Vd or Vsd's 0.787.** They are the best of several post-hoc subsets, and their difference from
   V is within fold noise.

**Implication for the CMDF hypothesis.** At **neither** scale does cross-modal disagreement demonstrably add
information beyond single-sequence reading. What survives is narrower and still useful: **a compact,
interpretable voxel-coupling descriptor that carries the 1p/19q signal as well as single-sequence intensity
histograms do**, together with a quantitative mismatch that replicates a known sign. Coupling would be
practically preferable only if it transported better across sites. Section 4.5 says this holds for T2–FLAIR
coupling but not T1–T2 coupling, so it can't be assumed. These results are post-hoc and must be reported that
way. The external phase can confirm them prospectively (FO+V vs FO and Vc vs FO as pre-registered secondary
endpoints).

### 4.2 Is V better than the other imaging representations?

The protocol pre-specified V only against age + sex (S4) and, with cohort, against M0 (S7). Post-hoc paired
tests on the same 50 UTSW folds:

| Comparison | Δ fold mean | NB 95 % CI | p | Folds Δ > 0 |
|---|---|---|---|---|
| V − P (single-sequence patch radiomics) | +0.109 | −0.032 to 0.250 | 0.127 | 38/50 |
| V − M2 (whole-tumour radiomics) | +0.085 | −0.041 to 0.211 | 0.184 | 39/50 |
| V − M1 (age, sex, volume) | +0.106 | −0.066 to 0.278 | 0.220 | 33/50 |
| V − D_T2FL | +0.083 | −0.053 to 0.218 | 0.226 | 34/50 |
| V − MM (its own 3 mismatch features) | +0.044 | −0.077 to 0.165 | 0.470 | 31/50 |
| V − P+D | +0.154 | 0.021 to 0.288 | **0.024** | 41/50 |
| M2+V − V | −0.063 | −0.179 to 0.054 | 0.284 | 12/50 |
| pooled: V − P | −0.005 | −0.083 to 0.073 | 0.892 | 19/50 |
| pooled: V+cohort − P+cohort | +0.043 | −0.030 to 0.115 | 0.242 | 31/50 |

**Reading.** V's point estimate is the best in UTSW, and 33–41 of 50 folds favour it. But with 36 events only the
comparison against the weakest imaging set (P+D) clears p < 0.05, and that is unadjusted. In the pooled set V is
no better than P. Two claims are defensible: "V is the only imaging representation with a permutation-validated
signal" and "V has the highest point estimate". The claim "V outperforms radiomics" is **not**. MM alone (3
features, slope 1.01) is within 0.044 of V. A parsimonious model is a credible alternative headline.

### 4.3 Confounding: volume, age, field strength, grade (UTSW, n = 74)

Spearman ρ of each leading V feature with each covariate, and p-values for differences by field strength and
grade:

| Feature | ρ log volume | ρ age | p field strength | p grade |
|---|---|---|---|---|
| T1–T2 Spearman | +0.23 (p 0.05) | +0.23 (p 0.05) | 0.86 | 0.85 |
| T1−T2 difference SD | −0.10 | **−0.39 (p 0.001)** | 0.57 | 0.50 |
| T1−T2 difference mean | +0.12 | **+0.41 (p < 0.001)** | 0.79 | 0.99 |
| Mismatch fraction | **+0.31 (p 0.007)** | **−0.32 (p 0.005)** | 0.29 | 0.77 |
| FLAIR core − rim | **−0.44 (p < 0.001)** | −0.01 | 0.21 | 0.74 |
| T2 core − rim | −0.07 | **−0.41 (p < 0.001)** | 0.07 | 0.34 |

Odds ratio for codeletion per SD of the feature (UTSW, unpenalised logistic regression):

| Feature | Crude OR (95 % CI) | Adjusted for log volume + age |
|---|---|---|
| T1–T2 Spearman | 3.92 (1.84–8.38), p 0.0004 | **3.83 (1.72–8.56), p 0.001** |
| T1−T2 difference SD | 0.24 (0.12–0.50), p 0.0001 | **0.28 (0.13–0.60), p 0.001** |
| T1−T2 difference mean | 2.80 (1.55–5.08) | 2.33 (1.24–4.39), p 0.009 |
| Mismatch fraction | 0.46 (0.25–0.87), p 0.016 | **0.47 (0.24–0.93), p 0.029** |
| T2–FLAIR Spearman | 1.79 (1.04–3.09) | 1.97 (1.06–3.64), p 0.031 |
| T2 core − rim | 0.48 (0.28–0.82) | 0.60 (0.34–1.06), p 0.078 |
| FLAIR core − rim | 1.04 (0.66–1.65), p 0.87 | 1.14 (0.67–1.94) |

**Reading.** The leading signals survive adjustment for volume and age, and none depends on field strength or
grade within UTSW. Several features correlate with age (codeleted patients are older), so the paper should
report V's increment over age (V+clin vs clin) as well as V alone.

### 4.4 Site effects: pooled univariate claims need cohort adjustment

The pre-specified pooled univariate tests ignore cohort. Prevalence differs (0.49 vs 0.11) and V features shift
strongly between sites, so crude pooled tests are confounded. Re-testing with cohort, and then cohort + volume +
age, as covariates (pooled n = 150):

| Feature | Crude OR per SD | + cohort | + cohort + volume + age |
|---|---|---|---|
| **Mismatch fraction (H2)** | 0.55, p 0.024 | **0.44, p 0.006** | **0.50, p 0.023** |
| T1–T2 Spearman | 1.83, p 0.002 | 2.39, p 0.0003 | 2.41, p 0.0004 |
| T1−T2 difference SD | 0.74, p 0.13 | 0.23, p < 0.0001 | 0.26, p 0.0001 |
| T2–FLAIR Spearman | 2.04, p 0.003 | 2.14, p 0.004 | 2.09, p 0.009 |
| **FLAIR core − rim** | 1.93, p 0.002 | **1.34, p 0.18** | **1.27, p 0.34** |

**Two corrections follow.**

1. **H2 is stronger than the conclusion states.** The pooled crude p = 0.08 comes from the site difference
   diluting the effect. Adjusted for cohort, the mismatch fraction is significant in the pooled set too. The
   mismatch-fraction OR above (0.55, p 0.024) comes from a logistic model; the protocol's rank test gave
   p = 0.08.
2. **The pooled "FLAIR core − rim is higher in codeleted" finding is an artefact.** It has no effect in UTSW
   (δ +0.095, p 0.49; OR 1.04). It appears only in UCSF, where codeleted tumours are far smaller (26 vs 74 ml),
   and core − rim correlates with volume (ρ −0.44). Its pooled q = 0.003 is cohort and volume confounding.
   `CMDF/outputs/INSIGHTS.md` §3 and `OUTPUTS_GUIDE.md` present it as biology ("astrocytomas more often have the
   classic pattern"). **That interpretation should be withdrawn.**

### 4.5 Site shift explains V's transport failure

Leading V features in *non-codeleted* patients only, UCSF vs UTSW, where the label is held fixed:

| Feature | UTSW median | UCSF median | Site Cliff's δ | p |
|---|---|---|---|---|
| T1−T2 difference SD | 1.70 | 1.10 | **−0.77** | < 0.001 |
| T2 core − rim | 1.85 | 0.76 | −0.76 | < 0.001 |
| T1CE−T2 difference P10 | −4.47 | −2.89 | +0.54 | < 0.001 |
| T1–T2 Spearman | −0.76 | −0.56 | +0.51 | < 0.001 |
| FLAIR core − rim | 1.01 | 0.32 | −0.50 | < 0.001 |
| Mismatch fraction | 0.059 | 0.014 | −0.29 | 0.013 |
| T2–FLAIR Spearman | 0.55 | 0.55 | +0.03 | 0.83 |

UCSF non-codeleted tumours "look codeleted" on most V features. Possible reasons: a single 3 T protocol, a
different sequence implementation, BraTS-style masks rather than UTSW's mixed label sets, and more grade-4 cases.
A model trained on UTSW therefore shifts every UCSF patient towards "codeleted". The ranking within UCSF is
partly preserved (AUROC 0.737) but the threshold isn't (specificity 0.15). Whole-brain z-scoring doesn't remove
sequence-protocol contrast differences. **Any clinical claim for V needs harmonisation, site-specific
recalibration, or features that are invariant to intensity scale.**

Among the leading features, T2–FLAIR Spearman coupling is the one with no site shift (δ 0.03) that is still
label-related in both cohorts. It is a candidate for a robust feature.

### 4.6 Does the T1–T2 coupling finding replicate within UCSF?

Within UCSF, codeleted (8) vs non-codeleted (68). Direction is compared with UTSW:

| Feature | UTSW δ | UCSF δ | UCSF p | Same direction? |
|---|---|---|---|---|
| T1–T2 Spearman | +0.61 | +0.19 | 0.39 | yes, weak |
| T1−T2 difference SD | −0.57 | −0.53 | **0.012** | **yes** |
| T1−T2 difference mean | +0.51 | +0.32 | 0.15 | yes |
| T1CE−T2 difference P10 | +0.52 | +0.46 | **0.033** | **yes** |
| Mismatch fraction | −0.34 | −0.39 | 0.074 | **yes** |
| T2–FLAIR Spearman | +0.34 | +0.56 | **0.008** | **yes** |
| T2 core − rim | −0.38 | −0.06 | 0.78 | yes, null |

**Reading.** The T1–T2 *relationship* replicates in direction, and partly in significance despite only 8
events. The coupling *statistic itself* is the weakest replicator. The paper should present the whole T1/T2
joint-intensity family as one exploratory finding, not lead with the Spearman coefficient.

### 4.7 The quantitative mismatch as a clinical-style rule

Thresholds below were chosen after looking at the data, so they are illustrative only:

| Cohort | Rule | Non-codeleted flagged (sensitivity for astrocytoma) | Codeleted flagged | Specificity for astrocytoma | PPV for astrocytoma |
|---|---|---|---|---|---|
| UTSW | mismatch > 0.10 | 16/38 (0.42) | 5/36 | 0.86 | 0.76 |
| UTSW | mismatch > 0.20 | 11/38 (0.29) | 2/36 | 0.94 | 0.85 |
| UCSF | mismatch > 0.10 | 20/68 (0.29) | 0/8 | 1.00 | 1.00 |
| UCSF | mismatch > 0.20 | 11/68 (0.16) | 0/8 | 1.00 | 1.00 |

This high-specificity, low-sensitivity profile matches the visual sign. It also matches published automated
quantifications: Cho et al. (AJNR 2024) report 100 % specificity with 20–32 % sensitivity at a 42 % mismatch
cut-off. The two codeleted UTSW tumours above 0.20 are BT0182, histologically an *oligoastrocytoma* (mismatch
0.55), and BT0602, an anaplastic oligodendroglioma (0.29). These could be label noise or true exceptions. A
pre-specified threshold should be fixed on UTSW and tested externally.

### 4.8 Other threats to validity

| Threat | Why it matters | Status / mitigation |
|---|---|---|
| **Whole-abnormality masks** | V and MM mix tumour core and oedema; "core" and "rim" are geometric, not anatomical | Pre-specified limitation. Sub-region masks are needed (Section 6, T2.4) |
| **Different segmentation protocols by site** | UTSW has 7 label configurations; UCSF is BraTS-style. Mask extent drives the voxel statistics | Re-segment all patients with one tool and report V's stability (ICC) between manual and automatic masks |
| **Normalisation** | Whole-brain z-score per sequence: tumour burden and protocol contrast change the reference | Sensitivity analysis with a normal-appearing white-matter reference |
| **Grade-4 "easy negatives"** | Codeleted tumours are never grade 4 (5 UTSW, 17 UCSF grade-4 non-codeleted) | Re-run V without grade 4 (the habitat study did this; CMDF did not) |
| **Label noise** | 18 UTSW oligoastrocytomas; 4 + 2 histology/molecular discordances; BT0182 above | Re-run V without discordant cases (done in the habitat study, not in CMDF) |
| **Pre-registration provenance** | `PROTOCOL.md` was committed in the *same* commit as the outputs (`8035989`, 2026-10-14). The SHA proves it wasn't changed *during* the run, not that it predates the results. A timing probe also printed P and P+D AUROCs before the full run (declared deviation) | Say "pre-specified analysis plan", not "pre-registered". Register the next phase on OSF before touching external data |
| **Baselines known in advance** | The protocol was written after the habitat study, so M1/M2 performance was known | Declared in `PROTOCOL.md`. Harmless for H1 (P vs P+D) but should be stated |
| **Multiplicity in the exploratory univariate** | 45 features × 2 sets; BH within set only | Present it as exploratory; the coupling family counts as one finding |
| **Power** | 36 events give a ±0.11 AUROC CI and a ±0.18 ΔAUROC CI | See below |
| **Original-arm null uses 50 permutations** | p floor 0.02; coarse tails | Re-run with 200 or more |

**Power, back of the envelope.** The S4 interval (±0.18) implies an SE of about 0.09 for ΔAUROC. Detecting
Δ ≈ 0.085 with 80 % power needs the SE to fall to about 0.03, roughly 8–9 times the current UTSW sample. That is
out of reach. A **single-model external validation** with a pre-specified lower bound is achievable: for
AUROC ≈ 0.75 with 60 events and 60 non-events, the Hanley–McNeil SE is about 0.045 (95 % CI about ±0.09).
Design the confirmatory phase around absolute external discrimination and calibration, not around beating
age + sex.

### 4.9 Corrections needed in existing project documents

| Document | Current statement | Correction |
|---|---|---|
| `CMDF/outputs/INSIGHTS.md` §3; `OUTPUTS_GUIDE.md` | "FLAIR core-minus-rim is higher in codeleted tumours (pooled δ +0.42, q = 0.003)… astrocytomas more often have the classic pattern" | Confounded by site and volume (4.4); no effect in UTSW. Remove or relabel as an artefact |
| `INSIGHTS.md` §2 | "Voxel-level CMDF is the strongest imaging signal in either study" | Keep, but add that it isn't significantly better than P, M2 or MM (4.2) |
| `INSIGHTS.md` §3; `OUTPUTS_GUIDE.md` §4 | T1–T2 coupling "has not been reported before" | Unverified. One literature search found no direct precedent, so say "to our knowledge" only after a systematic search. It replicates weakly in UCSF (4.6) |
| `INSIGHTS.md` §2 | "The two features together reproduce the visual sign quantitatively, without a radiologist" | Automated quantification already exists (Cho et al. 2024; Eur Radiol 2025). Present it as consistent with prior work, not as a first |
| `INSIGHTS.md` §6; `OUTPUTS_GUIDE.md` §4 | V presented as a "cross-sequence" signal | Add that its information is shared with single-sequence intensity histograms (FO+V − FO +0.005; 4.1) |
| `CMDF/README.md` | "pre-registered" | "pre-specified (internally time-stamped)" |
| `conclusion.md` (H2) | pooled p = 0.081 | Add the cohort-adjusted result as a declared post-hoc analysis |

---

## 5. Where the project stands

### 5.1 Finished

- **The cohort** is fully audited: patient-flow table, flags, exclusions by class, and folds identical to the
  habitat study.
- **All five feature blocks** are implemented, unit-tested and cached. Voxel features exist for 150/150
  patients.
- **The whole pre-specified analysis has run**: H1, pairing null, S1–S7, H2, label permutation, LOCO,
  robustness, ablation, calibration and the original-protocol arm.
- **The cross-classifier check** is done (`models/`, 9 classifiers).
- **The reproducibility package exists**: manifest, checks, deviations log and a byte-identical rerun.
- **Documentation exists**: `PROTOCOL.md`, `conclusion.md`, `INSIGHTS.md`, `OUTPUTS_GUIDE.md` and this document.

### 5.2 Claims ledger: what the evidence supports today

| Claim | Supported? | Wording to use |
|---|---|---|
| Patch-level cross-sequence radiomic differences add no information over single-sequence patch radiomics | **Yes, strongly** | "did not improve discrimination (ΔAUROC −0.046, 95 % CI −0.112 to 0.020); co-located pairing was no better than random pairing" |
| Voxel-level cross-sequence features carry 1p/19q information in IDH-mutant glioma | **Yes** | "the only imaging representation exceeding a label-permutation null (p = 0.005)" |
| V outperforms clinical or radiomic models | **No** | "highest point estimate; differences not significant at n = 74" |
| Voxel cross-modal features add information beyond single-sequence intensity | **No** ⚑ (FO+V − FO +0.005) | "co-location-only features carried the signal (AUROC 0.75) as well as, but not beyond, single-sequence intensity histograms" |
| The quantitative T2–FLAIR mismatch fraction is lower in codeleted tumours | **Yes** (pre-specified, UTSW; cohort-adjusted pooled post-hoc) | "consistent with the visual T2–FLAIR mismatch sign" |
| T1/T2 joint-intensity relationship differs by codeletion | **Exploratory** | "hypothesis-generating; same direction in an independent cohort with 8 events" |
| The common radiomics protocol is optimistic | **Yes** | "on permuted labels the protocol reported AUROC 0.75 on average (maximum 0.885)" |
| A clinically usable V model | **No** | Needs harmonisation, recalibration and external validation |

### 5.3 Readiness by component

| Component | Status |
|---|---|
| Data, cohort, QC | ✅ Complete (64 UTSW cases recoverable with segmentation) |
| Feature engineering | ✅ Complete (a sub-region version is pending) |
| Pre-specified analysis | ✅ Complete |
| Sensitivity analyses (grade 4, discordant labels, normalisation, segmentation) | ⚠️ Not done for CMDF |
| Marginal-vs-coupling decomposition | ⚠️ Done post-hoc (4.1); report as such, confirm externally |
| Cohort-adjusted pooled inference | ⚠️ Done post-hoc (4.4); report as such, confirm externally |
| External validation | ❌ Not started |
| Harmonisation / recalibration | ❌ Not started |
| Clinical comparator (radiologist-read mismatch sign) | ❌ Not available |
| Manuscript, checklists, public registration | ❌ Not started |

---

## 6. What remains: prioritised work plan

Effort estimates assume the existing pipeline and caches. "Days" means focused working days.

### Tier 1: needed for any submission (paper A or C)

| # | Task | Why | How | Effort |
|---|---|---|---|---|
| T1.1 | **Protocol addendum for analyses *not yet run*, plus a labelled post-hoc stage** | Section 4 results have been seen and can only be reported as post-hoc. Analyses not yet run can still be pre-specified | (a) Add the Section 4 checks to the pipeline as a stage labelled "post-hoc". (b) Write `PROTOCOL_ADDENDUM.md` for grade-4 and discordance exclusions, NAWM normalisation and V+clin vs clin. Time-stamp it publicly (OSF) *before* running | 1–2 d |
| T1.2 | **Sensitivity runs for V and H2** | Grade-4 easy negatives and label noise are standard reviewer questions | Reuse the habitat robustness variants (no grade 4, no discordant cases) for V, MM and H2 | 1 d |
| T1.3 | **Increase permutations in the original-protocol arm** | 50 permutations give a coarse tail and a p floor of 0.02 | `--set naive_permutations=200` (about 35 min) | 0.5 d |
| T1.4 | **Correct the documents** (Section 4.9) | Avoid publishing an artefact | Edit `INSIGHTS.md`, `OUTPUTS_GUIDE.md`, `README.md`; regenerate `conclusion.md` with the addendum | 0.5 d |
| T1.5 | **Report V's increment over age + sex** (V+clin vs clin) | Several V features correlate with age | Add the model to the addendum | included in T1.1 |
| T1.6 | **Reporting checklists** | Journals require them | TRIPOD+AI, CLAIM 2024, METRICS / CLEAR (Section 7.4) | 2 d |

### Tier 2: needed for the positive V paper (paper B)

| # | Task | Why | How | Effort |
|---|---|---|---|---|
| T2.1 | **Recover the 64 UTSW patients excluded for a missing segmentation** (26 codeleted, 38 non) | Raises UTSW from 74 to about 138 patients and 36 to about 62 events; narrows CIs by about 25 % | Automatic segmentation (BraTS-trained nnU-Net or HD-GLIO) for *all* patients; Dice against existing masks; V re-extraction (about 20 s per patient) | 4–6 d |
| T2.2 | **One segmentation tool for all cohorts** | Removes cross-site mask-protocol differences (4.5, 4.8) | Same tool as T2.1; report V's ICC between manual and automatic masks on the 150 current patients | with T2.1 |
| T2.3 | **Harmonisation and recalibration** | V's threshold fails across sites (specificity 0.15) | (a) NAWM-referenced z-scoring; (b) ComBat fitted on training sites only; (c) intercept-only recalibration on a small local sample; (d) prefer scale-invariant features (correlations, rank-based quantities) | 3–5 d |
| T2.4 | **Sub-region masks** (non-enhancing core vs oedema) | The mismatch sign is defined on the non-enhancing tumour, not oedema | BraTS labels from T2.1's tool; recompute MM on the non-enhancing core | 2 d |
| T2.5 | **Lock the model** | External validation needs a frozen model | Freeze: V (or a reduced V; see 4.1 and 4.2), L2 logistic regression with a fixed C, normalisation, threshold, recalibration rule. Store the coefficients. Also lock FO and Vc so that "FO+V vs FO" and "Vc vs FO" can be pre-registered external secondary endpoints | 1 d |
| T2.6 | **External validation** | The main gap; reviewers in this field expect it | Candidates: **EGD** (Erasmus Glioma Database: 774 patients, 4 sequences, 1p/19q status for 259; manual or automatic segmentations; data-use agreement through Health-RI XNAT). **TCGA-LGG** on TCIA with expert BraTS segmentations and TCGA molecular labels. Pre-register the endpoint: AUROC and calibration of the locked model. Count the IDH-mutant subset with four sequences before committing | 2–4 weeks including access |
| T2.7 | **Clinical comparator** | The real question is the value over a radiologist reading the mismatch sign | A blinded neuroradiologist (or two, with kappa) reads the visual sign on the 74 UTSW cases (half a day of reading). Compare and combine with MM and V | 1 week elapsed |

### Tier 3: strengthening and novelty

| # | Task | Why |
|---|---|---|
| T3.1 | **Scale sweep** | Explains why voxel works and patches fail. Compute the same intensity-coupling statistic within windows of 4, 8, 16 and 32 mm. Today the patch-vs-voxel contrast confounds *scale* with *feature type* (texture differences vs intensity coupling), so INSIGHTS's "patches average the disagreement away" is untested |
| T3.2 | **Voxel-level pairing null** | The direct co-location test for V: shuffle voxel correspondence within the mask. Lower priority now, because 4.1 already answers the practical question at model level (FO+V vs FO). It requires reloading images (about 50 min per replicate set) |
| T3.3 | **Joint-histogram / mutual-information features per pair** | A richer cross-modal descriptor than mean, SD and correlation |
| T3.4 | **Decision-curve analysis** | Clinical utility framing (e.g. deciding on biopsy or extent of resection) |
| T3.5 | **Deep-learning comparator** | Reviewers may ask; a small 3D CNN under the same CV is enough to show the representation question isn't trivially solved |
| T3.6 | **UCSF → UTSW transport after T2.1** | Still impossible because UCSF has only 8 events. It needs an external cohort with codeleted cases (EGD) |

### Suggested sequence

```
Week 1   T1.1–T1.5 (addendum, sensitivity, corrections)  ──►  submit paper A/C if the timeline is tight
Week 2-3 T2.1–T2.2 (segment everyone; UTSW to ~138)  +  start the EGD/TCGA data-access requests
Week 3-4 T2.3–T2.5 (harmonise, sub-regions, lock the model)  +  T2.7 radiologist read
Week 5-8 T2.6 external validation (pre-registered)  ──►  paper B
```

---

## 7. Publication analysis

### 7.1 Contribution and novelty in context

| Element | Prior work | What this project adds | Novelty |
|---|---|---|---|
| MRI prediction of 1p/19q | Many radiomics and DL studies. A reference point with external validation is van der Voort et al. 2019 (Clin Cancer Res): SVM on T1-Gd + T2 + age + sex, external AUC 0.72 on 129 TCIA LGG patients; neuroradiologists averaged 0.81 | A leakage-free, pre-specified evaluation restricted to **IDH-mutant** tumours, where the clinical question actually sits | Moderate. Rigour, not the task |
| Automated or quantitative T2–FLAIR mismatch | Cho et al. 2024 (AJNR; digital subtraction): 100 % specificity, 20–32 % sensitivity at a 42 % cut-off. A 2025 Eur Radiol study computed an automated T2FM ratio with HD-GLIO masks | A simple z-difference mismatch fraction; pre-specified H2 confirmed; same direction in a second cohort | **Low as a standalone**, useful as a replication inside a broader framework |
| Cross-modal **voxel coupling** (T1–T2 joint intensity) | The search found voxel-wise *cross-modality* correlation work (e.g. PET–MRI by molecular subtype) and joint T1/T2 variables for IDH, but no direct report of T1–T2 voxel coupling for 1p/19q | Strongest exploratory univariate signal; robust to volume and age; same direction in UCSF. Twelve coupling features match single-sequence histograms (0.754 vs 0.713) but don't add to them (4.1) | **Potentially novel as a descriptor**, not as new information; needs a systematic search and replication |
| **Patch-level CMDF** (cross-sequence radiomic differences) | The original notebooks (unpublished) | A rigorous negative with a co-location null | Novel as a *negative* result. Negative results are publishable when the test is rigorous, as here |
| **Protocol optimism on random labels** | Leakage and selection bias are well documented in radiomics (e.g. Kocak et al.'s METRICS/CLEAR work) | A concrete, reproducible demonstration on real glioma data: AUROC up to 0.885 with no signal | Moderate; an effective teaching figure |

### 7.2 Paper options

**Option A: methods / negative-result paper.**
*"Cross-sequence radiomic differences do not predict 1p/19q codeletion: a pre-specified, leakage-controlled
evaluation, and how a common protocol produces AUROC 0.89 on random labels."*
- **For:** almost ready; strong evidence (tight CI, pairing null, robustness); useful to the field; low reviewer
  risk because the claims are conservative.
- **Against:** negative results get lower-tier venues, and it doesn't use the most interesting finding (V).

**Option B: positive V paper.**
*"Compact voxel-level intensity and cross-sequence coupling descriptors with quantitative T2–FLAIR mismatch
predict 1p/19q codeletion in IDH-mutant glioma: development and external validation."*
- **For:** the most clinically relevant story, with higher-impact venues.
- **Against:** it is **not defensible without** external validation (T2.6) and harmonisation (T2.3). After 4.1
  it can't claim that cross-modal information *adds* to single-sequence intensity. It must sell compactness,
  interpretability and (if shown) transportability instead. This is weeks to months of work with data-access
  risk.

**Option C (recommended if external data can't be obtained quickly): one combined, honestly framed paper.**
*"Where is the cross-modal signal? A pre-specified evaluation of cross-sequence MRI features for 1p/19q codeletion
in IDH-mutant glioma."*
- **Structure:**
  1. Primary: patch CMDF negative (H1 and pairing null).
  2. Pre-specified secondary: voxel CMDF signal (permutation p = 0.005) and H2 confirmed.
  3. Post-hoc, labelled:
     - the voxel signal is carried by pure co-location features but is shared with single-sequence
       histograms (4.1);
     - the T1/T2 coupling family is robust to confounders and directionally replicated in UCSF;
     - cohort-adjusted pooled tests.
  4. Methods: optimism of the common protocol.
- **Coherent message:** "the cross-modal disagreement idea is intuitive, but it doesn't add information beyond
  single-sequence intensity. Here is where its apparent signal comes from, and here is how a common protocol
  would have told a different story."
- **Limitations stated upfront:** single-centre primary analysis, transport is descriptive, no external
  validation, a site-dependent threshold.
- **For:** uses everything; the honest framing pre-empts reviewer objections; feasible in about 2–3 weeks
  (Tier 1).
- **Against:** reviewers in clinical journals will still ask for external validation. Expect
  "major revision" or a mid-tier venue.

**Recommendation.** Do Tier 1 now and request EGD/TCGA access in parallel. If external data arrive within about
6 weeks, write Option B with the Option A material as a secondary section. Otherwise submit Option C, and keep
external validation as a short follow-up paper.

### 7.3 Proposed manuscript (Option C)

**Working title.** *Does cross-sequence disagreement add information? A pre-specified evaluation of cross-modal MRI
features for 1p/19q codeletion in IDH-mutant glioma.*

**Abstract skeleton (≈ 300 words).**

- **Purpose:** test whether co-located disagreement between MRI sequences predicts 1p/19q codeletion beyond
  single-sequence radiomics.
- **Methods:**
  - Cohort: 74 IDH-mutant gliomas from UTSW (36 codeleted), plus 76 from UCSF-PDGM (8 codeleted).
  - Features: patch-level cross-sequence radiomic differences (1,128 features), voxel-level cross-sequence
    features (45), and a quantitative T2–FLAIR mismatch.
  - Validation: pre-specified analysis plan; 10 × 5-fold CV with training-only fitting; co-location (pairing)
    and label-permutation nulls; Nadeau–Bengio tests.
- **Results:**
  - Patch-level differences did not improve on single-sequence radiomics (ΔAUROC −0.046, 95 % CI −0.112 to
    0.020; pairing null p = 1.00).
  - Voxel-level features reached AUROC 0.77 (0.65–0.87), the only imaging set exceeding a permutation null
    (p = 0.005).
  - The mismatch fraction was lower in codeleted tumours (δ −0.34, p = 0.012).
  - In post-hoc analysis, 12 co-location-only features reached AUROC 0.75, but adding voxel features to
    single-sequence intensity statistics did not improve discrimination (ΔAUROC +0.005).
  - The common radiomics protocol reported AUROC up to 0.885 on permuted labels.
- **Conclusion:** voxel-level cross-sequence coupling is a compact carrier of the 1p/19q signal. However,
  cross-modal disagreement did not add information beyond single-sequence intensity at either patch or voxel
  scale. External validation is required.

**Figures (6) and tables (4):**

| # | Content | Source |
|---|---|---|
| Fig 1 | Concept: the mismatch sign generalises to six pairs; patch D vs voxel V | new schematic |
| Fig 2 | Patient-flow diagram | `05_patient_flow_metadata.csv`, `08_exclusions_by_class.csv` |
| Fig 3 | AUROC forest plot, UTSW and pooled | `10_auroc_forest.png` |
| Fig 4 | Null distributions: pairing, label permutation, original protocol | `10_null_distributions.png` |
| Fig 5 | Mismatch fraction by label and cohort, plus T1–T2 coupling by label and cohort | `10_t2flair_mismatch.png` + new panel |
| Fig 6 | Decomposition of V: FO vs Vc vs V vs FO+V (labelled post-hoc) | Section 4.1 |
| Table 1 | Cohort characteristics by label and site (age, sex, grade, field strength, volume) | habitat table 06 + Section 3.1 |
| Table 2 | Feature-block definitions | Section 2.3 |
| Table 3 | Primary, secondary and post-hoc comparisons (with labels) | Sections 3.3, 3.5, 4.2 |
| Table 4 | Original protocol vs leakage-free | Section 3.10 |
| Supplement | Calibration, pair ablation, robustness, voxel univariate, nine-classifier heatmap, LOCO, site-shift table | existing tables and figures |

### 7.4 Reporting-guideline gap analysis

| Item (TRIPOD+AI / CLAIM / METRICS) | Status |
|---|---|
| Data source, eligibility, flow diagram | ✅ |
| Outcome definition (1p/19q; label resolution rules) | ✅ (document the UCSF WHO-2021 label exclusion) |
| Sample-size justification | ⚠️ Fixed by data availability. State it, with a precision-based argument (Section 4.8) |
| Missing data handling | ✅ Median imputation in-fold; single-patch SD imputed |
| Preprocessing and feature definitions | ✅ (also report bin width, patch geometry, PyRadiomics commit) |
| Segmentation method and reliability | ⚠️ Masks are dataset-provided with different protocols; no inter-reader or ICC analysis (T2.2) |
| Feature robustness / reproducibility (METRICS) | ⚠️ No test–retest or segmentation-perturbation analysis |
| Leakage control, all fitting in-fold | ✅ Asserted for every fold |
| Hyper-parameter tuning | ✅ Nested |
| Model selection / no test-set selection | ✅ One pre-specified estimator |
| Performance with CIs; calibration | ✅ |
| Internal validation | ✅ Repeated CV |
| External validation | ❌ |
| Fairness / subgroup performance | ⚠️ Not reported by sex, age band, field strength or grade (easy to add) |
| Comparison to a clinical standard | ❌ No radiologist read (T2.7) |
| Code and data availability | ✅ Code ready; data public (TCIA / UTSW release) |
| Protocol registration | ⚠️ Internal only. Register the addendum and the external phase on OSF |

### 7.5 Anticipated reviewer critiques and prepared responses

| Likely critique | Response available now | Still to do |
|---|---|---|
| "n = 74 is small; the results are unstable" | 10 × 5 CV, per-repeat ranges, bootstrap and NB CIs, permutation null; the negative result's CI is tight | External cohort, UTSW expansion |
| "V is just intensity, not cross-modal" | Partly right, and we say so. Co-location-only features match V (0.754), so the signal isn't mean intensity, but V adds nothing to single-sequence histograms (+0.005) (4.1) | Confirm both as pre-registered endpoints externally |
| "V doesn't beat simple baselines" | It's the only permutation-significant imaging set; we say "highest point estimate", not "better" | V+clin vs clin; external validation |
| "The pooled results reflect site" | Cohort term, within-cohort AUROCs, cohort-adjusted univariate (4.4) | Harmonisation |
| "Masks differ between sites" | Acknowledged | One segmentation tool (T2.2) |
| "Why did the null beat the observed value?" | Section 3.4 | |
| "Grade-4 cases inflate the AUROC" | Only 5 in UTSW | Sensitivity run (T1.2) |
| "The T2–FLAIR mismatch is already known and automated" | Yes; we replicate it inside a pre-specified test and generalise it to six pairs | Cite Cho 2024 and Eur Radiol 2025 |
| "Was it really pre-registered?" | Protocol SHA and declared deviations | Use "pre-specified"; OSF for the next phase |
| "The original-protocol comparison is a straw man" | It is the authors' own earlier pipeline, reproduced faithfully (two variants) | 200 or more permutations |
| "No comparison with deep learning" | Out of scope; the representation question is the point | Optional small CNN (T3.5) |

### 7.6 Venues

| Paper | Suitable venue types (examples) |
|---|---|
| A (methods / negative) | Open methods-friendly journals (Scientific Reports, PLOS ONE, European Radiology Experimental, Insights into Imaging); imaging-informatics journals; MICCAI workshops (e.g. BrainLes); SPIE Medical Imaging |
| C (combined) | Neuro-oncology imaging (Neuro-Oncology Advances, Neuroradiology, Journal of Neuro-Oncology); Insights into Imaging; European Radiology Experimental |
| B (validated positive) | AJNR, European Radiology, Radiology: Imaging Cancer, Neuro-Oncology Advances |

Check each journal's current scope, article types (several have "negative results" or "technical note"
formats) and open-access fees before choosing.

### 7.7 Data, code, ethics, authorship

- **Data:** both cohorts are public de-identified releases (UCSF-PDGM on TCIA; the UTSW glioma dataset of 625
  patients with four sequences, molecular markers and segmentations). Cite the dataset papers and follow their
  data-use terms. Retrospective public data usually needs no new ethics approval; confirm with your institution.
- **Code:** `CMDF/` and `habitat_1p19q/` are ready to release. Add a licence, a frozen environment file for the
  Windows and WSL venvs, and the PyRadiomics commit hash, which is already in `requirements.txt`.
- **Shareable derived data:** the per-patient V features (`CMDF/outputs/features/voxel/*.json`) can be shared
  as supplementary data. This strengthens the paper.
- **Authorship and AI-assistance disclosure:** follow the target journal's policy on AI-assisted analysis and
  writing.

---

## 8. Appendices

### 8.1 File map (CMDF)

```
CMDF/PROTOCOL.md                         pre-specified plan (SHA in manifest)
CMDF/cmdf1p19q/features.py               P, D, cosine (fold-fitted), pairing null
CMDF/cmdf1p19q/voxel.py                  V and MM (cached per patient)
CMDF/cmdf1p19q/cv.py                     repeated CV, pairing null, label permutation, LOCO
CMDF/cmdf1p19q/naive.py                  original-protocol arm
CMDF/cmdf1p19q/study.py                  10-stage pipeline
CMDF/outputs/conclusion.md               generated verdicts
CMDF/outputs/INSIGHTS.md                 interpretation (needs the corrections in 4.9)
CMDF/outputs/tables/05_*                 main results;   06_* nulls;   07_* LOCO and QC;   08_* robustness and calibration;   09_* original arm
CMDF/outputs/figures/10_*                forest, ROC, nulls, ablation, mismatch
models/                                  nine-classifier comparison including V and P+D
```

### 8.2 Reproducing the post-hoc checks (⚑)

The checks in Section 4 were run with a stand-alone script that imports the study's own modules. It uses the same
base table, `run_cv`, `paired_comparison` and Nadeau–Bengio code, and reads only `CMDF/outputs/cache/data.pkl`,
`CMDF/outputs/features/voxel/*.json` and the fold-record tables. Runtime is a few minutes, plus about 15–20
minutes for the single-process CV in 4.1. The FO set is the 32 whole-tumour first-order PyRadiomics features
(mean, median, P10, P90, variance, skewness, kurtosis, IQR × 4 sequences) already in M2, computed from the same
preprocessed images. The script is not part of the pipeline; adding it as a labelled post-hoc stage is task
T1.1(a).

### 8.3 Glossary of statistics

- **AUROC (repeat-averaged OOF):** each patient's prediction is averaged over 10 repeats, then one AUROC is
  computed, with a patient-bootstrap 95 % CI.
- **ΔAUROC, Nadeau–Bengio:** a paired fold-level difference with the corrected resampled t-test for overlapping
  training sets.
- **Cliff's δ:** a rank effect size from −1 to 1; about 0.33 is medium and about 0.47 large.
- **Calibration slope:** 1 is ideal; below 1 means overfit (predictions too extreme).
- **Label-permutation p:** (1 + #null ≥ observed) / (1 + n), so the floor is 0.005 for n = 200.

### 8.4 Literature consulted for the publication analysis

- van der Voort SR et al. *Predicting the 1p/19q codeletion status of presumed low-grade glioma with an externally
  validated machine learning algorithm.* Clin Cancer Res 2019;25(24):7455–62. https://www.pubmed.gov/31548344/
- Cho NS et al. *Quantification of T2-FLAIR mismatch in nonenhancing diffuse gliomas using digital subtraction.*
  AJNR 2024;45(2):188. https://www.ajnr.org/content/ajnr/45/2/188.full.pdf
- Automated T2-FLAIR mismatch ratio with HD-GLIO segmentation, Eur Radiol 2025.
  https://link.springer.com/article/10.1007/s00330-025-11475-7
- van der Voort SR et al. *The Erasmus Glioma Database (EGD): structural MRI scans, WHO 2016 subtypes, and
  segmentations of 774 patients with glioma.* Data in Brief 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8203723/
- *The University of Texas Southwestern Glioma Dataset: MRI, molecular markers and segmentations.*
  https://utsouthwestern.elsevierpure.com/en/publications/the-university-of-texas-southwestern-glioma-dataset-mri-molecular/
- Voxel-wise PET–MRI correlation by molecular subtype (J Nucl Med 2021).
  https://jnm.snmjournals.org/content/jnumed/62/3/319.full.pdf

Reporting guidelines to cite: TRIPOD+AI (Collins et al., BMJ 2024), CLAIM 2024 update, METRICS (Kocak et al.,
Insights into Imaging 2024), CLEAR (Kocak et al., 2023).
