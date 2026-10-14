# CMDF study protocol (fixed before any CMDF result was computed)

**Title.** Cross-modal differential features (CMDF) of multiparametric MRI for 1p/19q codeletion in IDH-mutant glioma.

**Status when written (2026-10-13).** The habitat study (`habitat_1p19q/`) had been run, so the cohort, the baseline
models M0–M2 and their performance were known. No CMDF feature, model or result had been computed. This file's
SHA-256 is stored in `outputs/manifest.json` at run time.

## 1. Question

Do *co-located* disagreements between MRI sequences inside the tumour (T1, T1-Gd, T2, FLAIR) tell 1p/19q-codeleted
from non-codeleted IDH-mutant gliomas better than the same radiomic features read one sequence at a time?

Rationale: the radiological T2–FLAIR mismatch sign — T2-hyperintense tumour that is relatively suppressed on FLAIR — is
a cross-modal disagreement that is highly specific for IDH-mutant, 1p/19q-*intact* astrocytoma. CMDF generalises this
idea from one visual sign to every sequence pair and every radiomic feature, and asks whether the disagreement itself
carries information beyond each sequence's own feature distribution.

## 2. Cohort

Identical to the habitat study: IDH-mutant, primary 1p/19q label resolved, imaging QC passed, at least one valid
32-mm patch. Expected: UTSW 74 patients (36 codeleted), UCSF 76 (8), pooled 150. UCSF patients whose label is only
resolvable from the WHO 2021 diagnosis are excluded from every analysis here.

## 3. Features

Images: N4-corrected, brain z-scored, 1-mm isotropic; whole-abnormality mask (all non-zero labels). Patches: 32³ voxels,
stride 16, ≥ 25 % tumour. Per patch and sequence: 93 PyRadiomics intensity/texture features (bin width 0.25 z-units).
These are the habitat study's cached features, unchanged.

| Block | Definition | Columns |
|---|---|---|
| **P** single-sequence patch radiomics | per sequence × feature: mean and SD across the patient's patches | 744 |
| **D** patch CMDF | per sequence pair (A, B) × feature: Δ = A − B on the *same* patch; mean and SD of Δ across patches. Per pair: cosine similarity of the A and B feature profiles per patch (each feature standardised per sequence with **training-fold** means/SDs); mean and SD across patches | 1,128 |
| **D_T2FL** | D restricted to the T2–FLAIR pair | 188 |
| **V** voxel CMDF | per pair, over tumour voxels, d = z_A − z_B: mean, SD, P10, P50, P90; Pearson and Spearman voxel correlation of A and B (co-localisation coupling); plus MM | 45 |
| **MM** quantitative T2–FLAIR mismatch | fraction of tumour voxels with z_T2 − z_FLAIR > 1; FLAIR core − rim; T2 core − rim (core: normalised depth ≥ 0.5; rim: ≤ 3 mm from the boundary) | 3 |
| M0, M1_clin, M1, M2 | cohort; age + sex; + log volume; whole-tumour radiomics — as in the habitat study | — |

SDs from a single patch are missing (median-imputed inside the training fold). Only the cosine block is fitted
(training-fold standardisation); every other block is a fixed per-patient transform.

## 4. Models and validation

L2-penalised logistic regression, class-balanced, C tuned by inner 5-fold CV over {0.01 … 3}, threshold chosen on inner
out-of-fold predictions of the training fold only — identical to the habitat study. Outer: 10 × repeated stratified
5-fold (strata cohort × label), seed 20261012, same patient order as the habitat study, so folds are identical.

## 5. Primary hypothesis (H1)

Within UTSW: AUROC(P+D) > AUROC(P). **Supported only if** the Nadeau–Bengio 95 % CI of the fold-level ΔAUROC has a
lower bound > 0 **and** the pairing null gives p < 0.05.

**Pairing null.** For each patient, the patch order of T1-Gd, T2 and FLAIR is permuted independently, leaving T1
fixed, before D is computed. Each sequence's own distribution (and therefore P) is unchanged; only the co-location of
features across sequences is destroyed. P+D is refitted on the first-repeat folds; 100 replicates;
p = (1 + #{Δ_null ≥ Δ_obs}) / (1 + 100).

## 6. Secondary analyses (Benjamini–Hochberg across S1–S7)

| | Set | Comparison |
|---|---|---|
| S1 | UTSW | D − P |
| S2 | UTSW | P+D − M2 |
| S3 | UTSW | D_T2FL − M1_clin |
| S4 | UTSW | V − M1_clin |
| S5 | UTSW | M2+V − M2 |
| S6 | pooled | P+D+cohort − P+cohort |
| S7 | pooled | V+cohort − M0 |

**H2 (directional, from the literature).** The mismatch fraction is *lower* in codeleted tumours (Cliff's δ < 0),
in UTSW and in the pooled set.

Also reported: leave-one-cohort-out UTSW → UCSF (UCSF has 8 codeleted patients: descriptive only; UCSF → UTSW is not
run because fewer than 20 training events); label-permutation test (within cohort, label-independent folds, 200
permutations) for P+D and V in UTSW; calibration.

## 7. Methodological arm (pre-specified)

The original CMDF notebooks' protocol is replicated on the same UTSW P+D features: standardisation and top-30
mutual-information feature selection on **all** patients before splitting, one 5-fold split, five classifiers
(KNN, decision tree, Gaussian NB, random forest, logistic regression), best classifier reported. The same protocol is
re-run on 50 within-cohort label permutations to show the AUROC it produces when there is no signal.

## 8. Robustness (descriptive)

Random forest instead of logistic regression; patients with ≥ 4 patches only; leave-one-pair-out ablation of D.

## 9. Deviations

Any deviation from this file is listed in `outputs/conclusion.md` under "Deviations from protocol".
