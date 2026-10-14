# CMDF study: conclusion

Protocol: `PROTOCOL.md` (SHA-256 97eab31935b7735a…). Cohort: UTSW 74 patients (36 codeleted), UCSF 76 (8); repeated 10 × 5-fold CV, folds identical to the habitat study.

## Primary hypothesis (H1: P+D > P within UTSW)

- AUROC P = 0.662 (0.533–0.780); P+D = 0.628 (0.496–0.757).
- ΔAUROC (fold mean) = -0.046, Nadeau–Bengio 95% CI -0.112 to 0.020, p = 0.171; folds with Δ > 0: 8/50.
- Pairing null (co-location shuffled, 100 replicates): observed Δ (repeat 1) -0.046, null mean -0.008, p = 1.000.
- By the pre-specified rule, H1 is **NOT SUPPORTED**.

## Secondary comparisons (Benjamini–Hochberg across S1–S7)

- S1 UTSW: D − P = -0.040 (NB 95% CI -0.128 to 0.048; BH-adjusted p = 0.460).
- S2 UTSW: P+D − M2 = -0.070 (NB 95% CI -0.172 to 0.032; BH-adjusted p = 0.411).
- S3 UTSW: D_T2FL − M1_clin = 0.002 (NB 95% CI -0.167 to 0.171; BH-adjusted p = 0.978).
- S4 UTSW: V − M1_clin = 0.085 (NB 95% CI -0.093 to 0.263; BH-adjusted p = 0.460).
- S5 UTSW: M2+V − M2 = 0.022 (NB 95% CI -0.029 to 0.073; BH-adjusted p = 0.460).
- S6 pooled: P+D+cohort − P+cohort = -0.033 (NB 95% CI -0.068 to 0.002; BH-adjusted p = 0.213).
- S7 pooled: V+cohort − M0 = 0.115 (NB 95% CI 0.046 to 0.184; BH-adjusted p = 0.011).

## AUROC by model (repeat-averaged out-of-fold, 95% bootstrap CI)

| set | model | n | codeleted | AUROC |
|---|---|---|---|---|
| UTSW | M1_clin | 74 | 36 | 0.659 (0.527–0.788) |
| UTSW | M1 | 74 | 36 | 0.656 (0.520–0.787) |
| UTSW | M2 | 74 | 36 | 0.692 (0.566–0.811) |
| UTSW | P | 74 | 36 | 0.662 (0.533–0.780) |
| UTSW | D | 74 | 36 | 0.610 (0.471–0.737) |
| UTSW | P+D | 74 | 36 | 0.628 (0.496–0.757) |
| UTSW | D_T2FL | 74 | 36 | 0.675 (0.550–0.799) |
| UTSW | V | 74 | 36 | 0.765 (0.651–0.871) |
| UTSW | MM | 74 | 36 | 0.703 (0.576–0.820) |
| UTSW | M2+V | 74 | 36 | 0.708 (0.583–0.825) |
| pooled | M0 | 150 | 44 | 0.696 (0.596–0.788) |
| pooled | M1_clin | 150 | 44 | 0.648 (0.537–0.754) |
| pooled | M2 | 150 | 44 | 0.828 (0.762–0.893) |
| pooled | P | 150 | 44 | 0.802 (0.724–0.877) |
| pooled | D | 150 | 44 | 0.769 (0.691–0.848) |
| pooled | P+D | 150 | 44 | 0.776 (0.694–0.855) |
| pooled | V | 150 | 44 | 0.800 (0.716–0.879) |
| pooled | P+cohort | 150 | 44 | 0.805 (0.729–0.880) |
| pooled | P+D+cohort | 150 | 44 | 0.777 (0.694–0.857) |
| pooled | V+cohort | 150 | 44 | 0.851 (0.783–0.916) |
| pooled | M2+cohort | 150 | 44 | 0.830 (0.763–0.896) |

## H2: T2–FLAIR mismatch fraction lower in codeleted tumours

- UTSW: median 0.024 (codeleted, n=36) vs 0.059 (non-codeleted, n=38); Cliff's δ = -0.342 (95% CI -0.594 to -0.086), p = 0.0116 — lower in codeleted.
- pooled: median 0.012 (codeleted, n=44) vs 0.033 (non-codeleted, n=106); Cliff's δ = -0.181 (95% CI -0.361 to 0.015), p = 0.0812 — lower in codeleted.

## Null tests and transport

- Label permutation, UTSW P+D: observed 0.596 vs null mean 0.494 (95th pct 0.641), p = 0.129.
- Label permutation, UTSW V: observed 0.702 vs null mean 0.501 (95th pct 0.633), p = 0.005.
- Leave-one-cohort-out UTSW (n=74, 36 codel.) → UCSF (n=76, 8 codel.), M1_clin: AUROC 0.585 (0.346–0.824).
- Leave-one-cohort-out UTSW (n=74, 36 codel.) → UCSF (n=76, 8 codel.), M2: AUROC 0.757 (0.553–0.906).
- Leave-one-cohort-out UTSW (n=74, 36 codel.) → UCSF (n=76, 8 codel.), P: AUROC 0.759 (0.583–0.899).
- Leave-one-cohort-out UTSW (n=74, 36 codel.) → UCSF (n=76, 8 codel.), D: AUROC 0.772 (0.599–0.899).
- Leave-one-cohort-out UTSW (n=74, 36 codel.) → UCSF (n=76, 8 codel.), P+D: AUROC 0.794 (0.597–0.936).
- Leave-one-cohort-out UTSW (n=74, 36 codel.) → UCSF (n=76, 8 codel.), V: AUROC 0.737 (0.559–0.869).
- Leave-one-cohort-out UCSF → UTSW: NOT RUN: 8 codeleted training patients < 20.

## Robustness

- random forest: Δ(P+D − P) = -0.008 (NB 95% CI -0.056 to 0.040).
- >= 4 patches (n=72, 36 codel.): Δ(P+D − P) = -0.030 (NB 95% CI -0.105 to 0.045).
- Ablation without T1–T1CE: AUROC 0.628 (Δ +0.000).
- Ablation without T1–T2: AUROC 0.629 (Δ +0.001).
- Ablation without T1–FLAIR: AUROC 0.617 (Δ -0.011).
- Ablation without T1CE–T2: AUROC 0.635 (Δ +0.007).
- Ablation without T1CE–FLAIR: AUROC 0.626 (Δ -0.002).
- Ablation without T2–FLAIR: AUROC 0.610 (Δ -0.018).

## Methodological arm: the original notebooks' protocol on the same UTSW P+D features

- notebook: best classifier GNB AUROC 0.712 ± 0.067; the same protocol on permuted labels gives 0.588 on average (95th pct 0.698, max 0.751), p = 0.059. Leakage-free estimate: 0.628.
- select_all: best classifier RF AUROC 0.827 ± 0.067; the same protocol on permuted labels gives 0.745 on average (95th pct 0.840, max 0.885), p = 0.118. Leakage-free estimate: 0.628.

## Deviations from protocol

- After PROTOCOL.md was fixed and before the full run, a one-repeat timing probe on UTSW printed fold-mean AUROCs for P (0.657) and P+D (0.601). Nothing in the protocol, features or models was changed afterwards.
- Methodological arm: the protocol described one variant (selection on all patients); a second, 'notebook' variant (selection inside each fold, exactly as pipeline_cmdf.py) was added so the original notebooks are reproduced faithfully. Their VAE oversampling, 1D-CNN and attention-MIL models were not replicated (TensorFlow is not installed).

## Limitations

- Single-centre primary analysis (UTSW, 36 events); UCSF has 8 codeleted patients, so transport is descriptive.
- No untouched external cohort.
- Whole-abnormality masks; the mismatch features do not separate tumour core from oedema.
