# Parkinson's Disease Progression — Linear Regression

## Objective

Predict the **total UPDRS score** (Unified Parkinson's Disease Rating Scale) from voice biomarkers using linear regression methods, and compare the performance of **closed-form LLS** versus **Steepest Descent**.

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [UCI Machine Learning Repository — Parkinsons Telemonitoring](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring) |
| **Samples** | ~5,875 voice recordings |
| **Features** | 16 voice measures (jitter, shimmer, HNR, etc.) |
| **Target** | `total_UPDRS` (continuous, 7–55) |

## Techniques

| Method | Description |
|--------|-------------|
| **Data Normalization** | Z-score standardization (mean=0, std=1) computed on training set only |
| **Feature Selection** | Removal of collinear features (`Jitter:DDP`, `Shimmer:DDA`) and identifiers (`subject#`) |
| **Linear Least Squares (LLS)** | Closed-form solution for the weight vector |
| **Steepest Descent (SD)** | Iterative optimization with adaptive learning rate |
| **Evaluation Metrics** | MSE, R², Pearson correlation coefficient |

## Pipeline

```
Load CSV → Correlation Analysis → Shuffle → Split (50/50)
→ Normalize (training stats only) → Drop Features
→ Solve (LLS & SD) → De-normalize → Evaluate & Compare
```

## Key Results

- Both LLS and SD converge to **virtually identical** weight vectors
- R² ≈ 0.58 on the test set, indicating moderate predictive power
- `motor_UPDRS` is the most influential predictor when included
- Error distributions are approximately Gaussian and centered near zero

## How to Run

```bash
cd 02_parkinson_regression
python parkinson_regression.py
```

## Files

| File | Description |
|------|-------------|
| `parkinson_regression.py` | Main analysis script (OOP-based) |
| `parkinson_regression.ipynb` | Jupyter notebook with inline outputs |
| `data/parkinsons_updrs.csv` | Dataset |
