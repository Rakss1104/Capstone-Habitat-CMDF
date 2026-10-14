# CMDF study — insights and observations

Numbers come from `conclusion.md` and `tables/`. Cohort: IDH-mutant gliomas, UTSW 74 (36 codeleted), UCSF 76
(8 codeleted). Leakage-free 10 × 5-fold CV with folds identical to the habitat study. The run was reproduced
byte-for-byte after a move from WSL to a Windows environment with identical package versions.

## 1. Patch-level CMDF does not help (primary hypothesis H1 not supported)

| UTSW | AUROC (95% CI) |
|---|---|
| P — single-sequence patch radiomics | 0.662 (0.533–0.780) |
| P+D — adds patch CMDF | 0.628 (0.496–0.757) |
| Δ (P+D − P) | −0.046 (NB 95% CI −0.112 to 0.020); 8/50 folds favour P+D |

- **Co-location carries no signal at patch scale.** With co-location shuffled, P+D loses only 0.008 against P; with
  real co-location it loses 0.046 (pairing-null p = 1.00). Real cross-sequence disagreement is no more informative
  than random pairing, and the 1,128 extra features only dilute P.
- The result holds with random forests (Δ −0.008) and in patients with ≥ 4 patches (Δ −0.030).
- Leave-one-pair-out: removing T2–FLAIR costs the most (−0.018), removing T1CE–T2 gains slightly (+0.007).
  T2–FLAIR is the most useful pair, matching the radiological rationale, but no pair matters much at patch scale.
- Interpretation: 32-mm patches average the voxel-level disagreement away. The second-order differences of 93 texture
  features are noisy and high-dimensional for 59 training patients.

## 2. Voxel-level CMDF is the strongest imaging signal in either study

| UTSW | AUROC (95% CI) | label-permutation p |
|---|---|---|
| **V — voxel CMDF (45 features)** | **0.765 (0.651–0.871)** | **0.005** (floor for 200 permutations) |
| MM — 3-feature quantitative T2–FLAIR mismatch | 0.703 (0.576–0.820) | — |
| M2 — whole-tumour radiomics (386 features) | 0.692 (0.566–0.811) | — |
| habitat M3 / M4 (habitat study) | 0.607 / 0.597 | 0.13 / 0.20 |
| M1_clin — age + sex | 0.659 (0.527–0.788) | 0.025 (habitat study, 140-patient UTSW metadata set) |

- V is the only imaging representation across both studies that clearly beats a label-permutation null. It is also
  the best calibrated imaging model (slope 0.77, Brier 0.199) and has the best threshold metrics in UTSW: sensitivity
  0.74, specificity 0.66, precision 0.68, F1 0.71, balanced accuracy 0.70.
- **Caveat:** V's gain over age + sex (S4, +0.085) is not significant after correction (NB 95% CI −0.093 to 0.263).
  With 36 events the interval is too wide; this is a power limit, not evidence of no effect.
- Pooled: V + cohort reaches 0.851 (0.783–0.916), the best pooled model, and significantly better than cohort alone
  (S7, Δ +0.115, BH p = 0.011). Its within-cohort AUROCs are UTSW 0.790 and UCSF 0.801 (UCSF has 8 events), so it is
  not simply learning the site.
- Adding V to whole-tumour radiomics (M2+V) helps less than V alone (0.708): 386 radiomics features dilute 45 focused ones.

## 3. Biology: the T2–FLAIR mismatch sign, quantified

- **H2 confirmed in UTSW.** The mismatch fraction (share of tumour voxels with z_T2 − z_FLAIR > 1) is lower in
  codeleted tumours: median 0.024 vs 0.059, Cliff's δ −0.34 (95% CI −0.59 to −0.09), p = 0.012. Pooled: δ −0.18,
  p = 0.08. UCSF shows the same direction (figure `10_t2flair_mismatch.png`): every codeleted UCSF tumour is below 0.06.
- FLAIR core-minus-rim is *higher* in codeleted tumours (pooled δ +0.42, q = 0.003). Astrocytomas more often have the
  classic pattern: FLAIR-suppressed core with a hyperintense rim. The two features together reproduce the visual sign
  quantitatively, without a radiologist.
- **Strongest exploratory signal: T1–T2 voxel coupling.** Codeleted tumours show a weaker T1/T2 anticorrelation
  (Spearman −0.48 vs −0.76, δ +0.61, q = 0.0003) and a narrower spread of T1 − T2 differences (δ −0.57). Non-codeleted
  tumours behave more like a "pure" fluid-rich lesion (dark T1, bright T2, tightly coupled); codeleted tumours are more
  heterogeneous, which fits calcification, haemorrhage and the cortical involvement typical of oligodendroglioma.
  This was not pre-specified: treat it as hypothesis-generating until replicated.

## 4. Cross-cohort transport (UTSW → UCSF, descriptive: 8 UCSF events)

All imaging models transport with AUROC 0.74–0.79 (V 0.737, P+D 0.794, M2 0.757), against 0.585 for age + sex. Every
confidence interval spans roughly 0.56–0.94, so the models cannot be ranked here. UCSF → UTSW was not run
(8 training events < 20).

## 5. Methodological finding: the original notebooks' numbers are not credible

The original protocol, re-run on the same UTSW P+D features:

| Protocol | Reported AUROC | Same protocol, random labels (mean / 95th pct / max) | Leakage-free |
|---|---|---|---|
| notebook (in-fold selection, best of 5 classifiers, one split) | 0.712 ± 0.067 | 0.588 / 0.698 / 0.751 | 0.628 |
| selection on all patients before splitting | 0.827 ± 0.067 | **0.745 / 0.840 / 0.885** | 0.628 |

- Choosing the best of five classifiers on one split adds about +0.09 optimism. Selecting features on all patients
  first adds another ≈ +0.15: on pure noise this protocol reports 0.75 on average and up to 0.89.
- The original notebooks reported AUROC 0.898 (CMDF_RF) on 62 patients, 10 of them in the minority class, including
  IDH-wildtype tumours. That figure sits inside the range this protocol produces on random labels, so it should not be
  cited.

## 6. What this means for a paper

**Defensible contributions:**
1. A negative result: patch-level cross-modal radiomic differences add nothing (pre-registered, pairing null p = 1.0).
2. A positive, permutation-validated imaging signal from 45 voxel-level cross-sequence features, including an
   automated, quantitative T2–FLAIR mismatch that reproduces a known radiological sign (H2 pre-specified and confirmed).
3. A methods demonstration: the common radiomics protocol produces AUROC 0.75–0.89 on random labels in this cohort.

**Still needed:**
- Power and external validation: 36 events gives CIs of ±0.11. An external IDH-mutant cohort with 1p/19q labels
  (EGD, or TCGA-LGG on TCIA) would test V and the T1–T2 coupling finding.
- Tumour sub-region masks (core vs oedema) to sharpen the mismatch features.
- A pre-registered replication of the T1–T2 coupling result, which was exploratory here.
