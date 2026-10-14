# Habitat vs CMDF — classifier comparison

Nine classifiers × four feature sets (habitat M3, M4; CMDF V, P+D), 5 × repeated stratified 5-fold CV on patients, every learned step (habitat model, CMDF scaling, imputation, scaling, hyper-parameters, threshold) fitted on the training fold only. Pooled analyses include the cohort indicator. Runtime 50 min.

**Reading the numbers.** Precision (PPV) depends on prevalence: 0.49 codeleted in UTSW, 0.29 pooled. Threshold-dependent metrics come from a training-fold cut-off maximising balanced accuracy, so they are honest out-of-sample values, not the best achievable cut-off. Choosing the best of 9 classifiers after the fact is optimistic; the pre-specified estimator in both studies is L2 logistic regression.

## UTSW (n = 74, 36 codeleted)

### Best classifier per feature set (by AUROC)

| feature_set | classifier | n_features | AUROC | AUROC 95% CI | AUPRC | precision (PPV) | recall (sensitivity) | specificity | NPV | F1 | MCC | balanced accuracy | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V | GradientBoosting | 45 | 0.802 | 0.694–0.901 | 0.790 | 0.671 | 0.689 | 0.679 | 0.701 | 0.678 | 0.370 | 0.684 | 0.192 |
| P+D | RandomForest | 1872 | 0.733 | 0.610–0.845 | 0.676 | 0.684 | 0.611 | 0.726 | 0.665 | 0.642 | 0.343 | 0.669 | 0.212 |
| M4 | GradientBoosting | 11 | 0.652 | 0.515–0.786 | 0.671 | 0.602 | 0.628 | 0.595 | 0.626 | 0.609 | 0.225 | 0.611 | 0.233 |
| M3 | SVM-RBF | 5 | 0.615 | 0.482–0.743 | 0.569 | 0.534 | 0.539 | 0.553 | 0.568 | 0.522 | 0.097 | 0.546 | 0.240 |

### Mean over the nine classifiers

| approach | feature_set | AUROC | AUPRC | precision (PPV) | recall (sensitivity) | specificity | F1 | MCC | balanced accuracy |
|---|---|---|---|---|---|---|---|---|---|
| cmdf | P+D | 0.672 | 0.624 | 0.592 | 0.660 | 0.557 | 0.619 | 0.222 | 0.609 |
| cmdf | V | 0.758 | 0.721 | 0.684 | 0.731 | 0.680 | 0.706 | 0.413 | 0.706 |
| habitat | M3 | 0.599 | 0.569 | 0.563 | 0.575 | 0.573 | 0.565 | 0.150 | 0.574 |
| habitat | M4 | 0.604 | 0.590 | 0.568 | 0.560 | 0.591 | 0.558 | 0.154 | 0.576 |

### Pre-specified estimator (L2 logistic regression)

| feature_set | n_features | AUROC | AUROC 95% CI | AUPRC | precision (PPV) | recall (sensitivity) | specificity | NPV | F1 | MCC | balanced accuracy | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 5 | 0.596 | 0.462–0.729 | 0.562 | 0.563 | 0.594 | 0.563 | 0.595 | 0.577 | 0.158 | 0.579 | 0.246 |
| M4 | 11 | 0.578 | 0.440–0.719 | 0.552 | 0.578 | 0.556 | 0.621 | 0.601 | 0.565 | 0.178 | 0.588 | 0.249 |
| V | 45 | 0.735 | 0.613–0.849 | 0.692 | 0.674 | 0.700 | 0.679 | 0.707 | 0.686 | 0.380 | 0.689 | 0.212 |
| P+D | 1872 | 0.614 | 0.484–0.739 | 0.564 | 0.575 | 0.656 | 0.542 | 0.628 | 0.612 | 0.200 | 0.599 | 0.279 |

### Paired CMDF − habitat ΔAUROC per classifier (Nadeau–Bengio, BH within set)

| classifier | comparison | ΔAUROC (fold mean) | NB 95% CI | NB p | NB p (BH, within set) | folds CMDF better |
|---|---|---|---|---|---|---|
| LogReg-L2 | V − M4 | 0.142 | 0.006 to 0.277 | 0.041 | 0.115 | 21/25 |
| LogReg-L2 | V − M3 | 0.107 | -0.038 to 0.253 | 0.141 | 0.201 | 19/25 |
| LogReg-L2 | P+D − M4 | 0.021 | -0.139 to 0.180 | 0.793 | 0.824 | 17/25 |
| LogReg-ElasticNet | V − M4 | 0.172 | 0.012 to 0.332 | 0.036 | 0.115 | 22/25 |
| LogReg-ElasticNet | V − M3 | 0.090 | -0.041 to 0.222 | 0.170 | 0.229 | 18/25 |
| LogReg-ElasticNet | P+D − M4 | 0.029 | -0.147 to 0.205 | 0.736 | 0.795 | 16/25 |
| SVM-RBF | V − M4 | 0.207 | 0.007 to 0.408 | 0.043 | 0.115 | 23/25 |
| SVM-RBF | V − M3 | 0.178 | -0.019 to 0.374 | 0.074 | 0.135 | 21/25 |
| SVM-RBF | P+D − M4 | 0.048 | -0.220 to 0.317 | 0.713 | 0.795 | 16/25 |
| RandomForest | V − M4 | 0.177 | 0.022 to 0.333 | 0.027 | 0.115 | 21/25 |
| RandomForest | V − M3 | 0.181 | 0.019 to 0.343 | 0.030 | 0.115 | 22/25 |
| RandomForest | P+D − M4 | 0.123 | -0.042 to 0.287 | 0.137 | 0.201 | 22/25 |
| ExtraTrees | V − M4 | 0.163 | 0.025 to 0.302 | 0.023 | 0.115 | 21/25 |
| ExtraTrees | V − M3 | 0.154 | 0.008 to 0.300 | 0.039 | 0.115 | 20/25 |
| ExtraTrees | P+D − M4 | 0.114 | -0.039 to 0.267 | 0.139 | 0.201 | 22/25 |
| GradientBoosting | V − M4 | 0.175 | 0.000 to 0.350 | 0.050 | 0.115 | 21/25 |
| GradientBoosting | V − M3 | 0.186 | -0.023 to 0.396 | 0.078 | 0.135 | 21/25 |
| GradientBoosting | P+D − M4 | 0.066 | -0.146 to 0.279 | 0.526 | 0.617 | 16/25 |
| KNN | V − M4 | 0.185 | 0.049 to 0.320 | 0.010 | 0.115 | 22/25 |
| KNN | V − M3 | 0.188 | 0.002 to 0.374 | 0.048 | 0.115 | 22/25 |
| KNN | P+D − M4 | 0.001 | -0.203 to 0.204 | 0.995 | 0.995 | 14/25 |
| GaussianNB | V − M4 | 0.227 | 0.023 to 0.431 | 0.031 | 0.115 | 22/25 |
| GaussianNB | V − M3 | 0.110 | -0.014 to 0.235 | 0.080 | 0.135 | 19/25 |
| GaussianNB | P+D − M4 | 0.101 | -0.065 to 0.267 | 0.220 | 0.283 | 20/25 |
| MLP | V − M4 | 0.177 | -0.020 to 0.374 | 0.076 | 0.135 | 19/25 |
| MLP | V − M3 | 0.166 | -0.001 to 0.332 | 0.051 | 0.115 | 23/25 |
| MLP | P+D − M4 | 0.069 | -0.130 to 0.267 | 0.483 | 0.592 | 15/25 |

- P+D − M4: 9/9 classifiers favour CMDF; 0 significant after BH
- V − M3: 9/9 classifiers favour CMDF; 0 significant after BH
- V − M4: 9/9 classifiers favour CMDF; 0 significant after BH

## pooled (n = 150, 44 codeleted)

### Best classifier per feature set (by AUROC)

| feature_set | classifier | n_features | AUROC | AUROC 95% CI | AUPRC | precision (PPV) | recall (sensitivity) | specificity | NPV | F1 | MCC | balanced accuracy | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V | LogReg-ElasticNet | 46 | 0.857 | 0.791–0.919 | 0.741 | 0.602 | 0.727 | 0.796 | 0.876 | 0.656 | 0.500 | 0.762 | 0.149 |
| P+D | ExtraTrees | 1873 | 0.855 | 0.791–0.915 | 0.719 | 0.551 | 0.800 | 0.726 | 0.898 | 0.651 | 0.486 | 0.763 | 0.163 |
| M4 | LogReg-L2 | 12 | 0.800 | 0.722–0.873 | 0.559 | 0.506 | 0.764 | 0.689 | 0.876 | 0.608 | 0.416 | 0.726 | 0.179 |
| M3 | MLP | 6 | 0.787 | 0.706–0.861 | 0.573 | 0.482 | 0.782 | 0.651 | 0.879 | 0.596 | 0.395 | 0.716 | 0.163 |

### Mean over the nine classifiers

| approach | feature_set | AUROC | AUPRC | precision (PPV) | recall (sensitivity) | specificity | F1 | MCC | balanced accuracy |
|---|---|---|---|---|---|---|---|---|---|
| cmdf | P+D | 0.811 | 0.624 | 0.528 | 0.742 | 0.719 | 0.614 | 0.429 | 0.730 |
| cmdf | V | 0.813 | 0.690 | 0.562 | 0.702 | 0.770 | 0.622 | 0.447 | 0.736 |
| habitat | M3 | 0.771 | 0.544 | 0.473 | 0.768 | 0.642 | 0.584 | 0.374 | 0.705 |
| habitat | M4 | 0.764 | 0.526 | 0.456 | 0.735 | 0.632 | 0.561 | 0.337 | 0.684 |

### Pre-specified estimator (L2 logistic regression)

| feature_set | n_features | AUROC | AUROC 95% CI | AUPRC | precision (PPV) | recall (sensitivity) | specificity | NPV | F1 | MCC | balanced accuracy | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M3 | 6 | 0.780 | 0.694–0.855 | 0.573 | 0.496 | 0.768 | 0.675 | 0.875 | 0.603 | 0.406 | 0.722 | 0.188 |
| M4 | 12 | 0.800 | 0.722–0.873 | 0.559 | 0.506 | 0.764 | 0.689 | 0.876 | 0.608 | 0.416 | 0.726 | 0.179 |
| V | 46 | 0.849 | 0.782–0.915 | 0.737 | 0.567 | 0.741 | 0.764 | 0.877 | 0.642 | 0.474 | 0.753 | 0.151 |
| P+D | 1873 | 0.783 | 0.704–0.863 | 0.551 | 0.535 | 0.664 | 0.760 | 0.845 | 0.592 | 0.402 | 0.712 | 0.180 |

### Paired CMDF − habitat ΔAUROC per classifier (Nadeau–Bengio, BH within set)

| classifier | comparison | ΔAUROC (fold mean) | NB 95% CI | NB p | NB p (BH, within set) | folds CMDF better |
|---|---|---|---|---|---|---|
| LogReg-L2 | V − M4 | 0.064 | -0.045 to 0.173 | 0.239 | 0.460 | 21/25 |
| LogReg-L2 | V − M3 | 0.067 | -0.026 to 0.160 | 0.151 | 0.454 | 21/25 |
| LogReg-L2 | P+D − M4 | -0.009 | -0.116 to 0.099 | 0.870 | 0.870 | 9/25 |
| LogReg-ElasticNet | V − M4 | 0.080 | -0.021 to 0.180 | 0.114 | 0.441 | 21/25 |
| LogReg-ElasticNet | V − M3 | 0.076 | -0.004 to 0.157 | 0.062 | 0.297 | 21/25 |
| LogReg-ElasticNet | P+D − M4 | 0.038 | -0.047 to 0.123 | 0.367 | 0.496 | 17/25 |
| SVM-RBF | V − M4 | 0.090 | -0.003 to 0.183 | 0.057 | 0.297 | 21/25 |
| SVM-RBF | V − M3 | 0.078 | -0.001 to 0.158 | 0.053 | 0.297 | 21/25 |
| SVM-RBF | P+D − M4 | 0.089 | 0.002 to 0.176 | 0.045 | 0.297 | 22/25 |
| RandomForest | V − M4 | 0.054 | -0.065 to 0.172 | 0.358 | 0.496 | 19/25 |
| RandomForest | V − M3 | 0.042 | -0.059 to 0.144 | 0.398 | 0.511 | 17/25 |
| RandomForest | P+D − M4 | 0.087 | -0.006 to 0.179 | 0.066 | 0.297 | 21/25 |
| ExtraTrees | V − M4 | 0.062 | -0.036 to 0.160 | 0.202 | 0.454 | 20/25 |
| ExtraTrees | V − M3 | 0.057 | -0.038 to 0.152 | 0.231 | 0.460 | 17/25 |
| ExtraTrees | P+D − M4 | 0.079 | -0.000 to 0.159 | 0.051 | 0.297 | 22/25 |
| GradientBoosting | V − M4 | 0.084 | -0.044 to 0.212 | 0.187 | 0.454 | 18/25 |
| GradientBoosting | V − M3 | 0.069 | -0.032 to 0.169 | 0.171 | 0.454 | 17/25 |
| GradientBoosting | P+D − M4 | 0.081 | -0.039 to 0.201 | 0.175 | 0.454 | 23/25 |
| KNN | V − M4 | 0.073 | -0.063 to 0.208 | 0.279 | 0.496 | 19/25 |
| KNN | V − M3 | 0.027 | -0.103 to 0.158 | 0.669 | 0.752 | 14/25 |
| KNN | P+D − M4 | 0.065 | -0.071 to 0.200 | 0.334 | 0.496 | 17/25 |
| GaussianNB | V − M4 | 0.085 | -0.089 to 0.258 | 0.323 | 0.496 | 18/25 |
| GaussianNB | V − M3 | 0.034 | -0.096 to 0.164 | 0.593 | 0.697 | 14/25 |
| GaussianNB | P+D − M4 | 0.056 | -0.065 to 0.177 | 0.348 | 0.496 | 18/25 |
| MLP | V − M4 | 0.030 | -0.075 to 0.135 | 0.561 | 0.689 | 17/25 |
| MLP | V − M3 | 0.019 | -0.091 to 0.129 | 0.721 | 0.779 | 15/25 |
| MLP | P+D − M4 | 0.009 | -0.081 to 0.100 | 0.832 | 0.864 | 12/25 |

- P+D − M4: 8/9 classifiers favour CMDF; 0 significant after BH
- V − M3: 9/9 classifiers favour CMDF; 0 significant after BH
- V − M4: 9/9 classifiers favour CMDF; 0 significant after BH

## Full per-classifier tables

- `habitat/README.md`, `cmdf/README.md`, `tables/comparison_all_metrics.csv`
