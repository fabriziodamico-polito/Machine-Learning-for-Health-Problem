# Chronic Kidney Disease - Classification with Missing Data

## Objective

Predict chronic kidney disease (CKD) from clinical and laboratory measurements while comparing two ways to handle missing values. The experiment is designed so the held-out patients cannot influence imputation, feature preprocessing or model fitting.

## Dataset

| Property | Value |
| --- | --- |
| Source | [UCI Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) |
| Samples | 400 patients |
| Features | 24 numeric and categorical clinical variables |
| Target | `classk` (CKD / not CKD) |
| License | CC BY 4.0; see [DATASETS.md](../DATASETS.md) |

## Leakage-safe pipeline

```text
Load and map values -> stratified 70/30 patient split
-> fit each imputer on training predictors only
-> transform train and test separately
-> train Decision Tree and Random Forest
-> evaluate once on the held-out patients
```

The target is never supplied to an imputer. Median imputation and multivariate iterative regression imputation are fitted separately. Predicted categorical values are snapped to categories observed in the training partition.

## Verified holdout results

| Imputation | Classifier | Accuracy | Balanced accuracy | F1 |
| --- | --- | ---: | ---: | ---: |
| Median | Decision Tree | 0.983 | 0.987 | 0.986 |
| Median | Random Forest | 1.000 | 1.000 | 1.000 |
| Iterative regression | Decision Tree | 0.992 | 0.993 | 0.993 |
| Iterative regression | Random Forest | 0.992 | 0.993 | 0.993 |

These strong scores describe one deterministic split of a small public dataset; they are not evidence of clinical readiness. External validation, calibration analysis and prospective evaluation would be required before any real-world use.

## Run

```bash
cd 05_kidney_disease_classification
python kidney_classification.py
```

| File | Description |
| --- | --- |
| `kidney_classification.py` | Imputation, classification and evaluation pipeline |
| `data/chronic_kidney_disease.arff` | UCI dataset in ARFF format |
