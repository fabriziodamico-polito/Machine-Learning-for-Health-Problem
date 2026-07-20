# Chronic Kidney Disease — Classification with Missing Data

## Objective

Build a classification system to predict **Chronic Kidney Disease (CKD)** from clinical and laboratory features, addressing the real-world challenge of **missing data** through two imputation strategies: **regression-based (LLS)** and **median imputation**.

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [UCI Machine Learning Repository — Chronic Kidney Disease](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease) |
| **Samples** | 400 patients |
| **Features** | 24 (11 numeric + 13 categorical) |
| **Target** | `classk` (binary: CKD / not CKD) |
| **Missing Values** | Significant — up to 6 missing features per row |

## Techniques

| Method | Description |
|--------|-------------|
| **LLS Regression Imputation** | For each incomplete row, train a local LLS model on complete rows using available features to predict missing ones |
| **Median Imputation** | Replace each missing value with the median from the complete training subset |
| **Categorical Rounding** | After regression, map continuous predictions back to the nearest valid categorical value |
| **CART Decision Tree** | Entropy-based splitting criterion for interpretable classification |
| **Random Forest** | Ensemble of 100/1000 decision trees for robust predictions |
| **Evaluation** | Accuracy, confusion matrix, feature importance ranking |

## Pipeline

```
Load ARFF → Map Categoricals → Analyze Missing Patterns
→ Extract Complete Rows (Training) → Regression Imputation (x_new)
→ Median Imputation (y_new) → Train Decision Tree & Random Forest
→ Compare Imputation Strategies → Split Experiment on y_new
```

## Key Results

- Regression imputation (**x_new**) produces distributions closer to the original data compared to median imputation
- **Random Forest (1000 trees)** achieves the highest accuracy (~98%) on the regression-imputed dataset
- Decision Tree provides interpretable rules: key features include `hemoglobin`, `specific gravity`, and `albumin`
- Median imputation (**y_new**) performs comparably due to the dataset's relatively simple separability

## How to Run

```bash
cd 05_kidney_disease_classification
python kidney_classification.py
```

## Files

| File | Description |
|------|-------------|
| `kidney_classification.py` | Main classification pipeline |
| `kidney_classification.ipynb` | Jupyter notebook with visual outputs |
| `data/chronic_kidney_disease.arff` | Dataset in ARFF format |
