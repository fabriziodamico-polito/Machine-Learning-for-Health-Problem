# Parkinson's Disease Progression — KNN-LLS Regression

## Objective

Improve upon standard linear regression by implementing a **K-Nearest Neighbors combined with Local Linear Regression (KNN-LLS)** approach for predicting `total_UPDRS`. This method fits a local linear model for each test point using only its K nearest training neighbors, capturing non-linear relationships in the data.

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
| **KNN-LLS** | For each test point, find K nearest neighbors in feature space, then solve a local LLS regression on that subset |
| **Ridge Regularization** | Small $\varepsilon I$ term added to $A^T A$ to ensure invertibility |
| **K Optimization** | Validation set used to sweep K values and select the optimal one (minimum MSE) |
| **Train/Validation/Test Split** | 40% / 20% / 40% partitioning |
| **Evaluation Metrics** | MSE, R², Pearson correlation coefficient |

## Pipeline

```
Load CSV → Correlation Analysis → Shuffle → Split (40/20/40)
→ Normalize → Drop Features → Optimize K on Validation Set
→ Merge Train+Validation → Test KNN-LLS vs Standard LLS → Evaluate
```

## Key Results

- **Optimal K** found via grid search on the validation set
- KNN-LLS achieves **lower MSE** and **higher R²** than standard LLS on the test set, confirming it captures local non-linear patterns
- The improvement is especially visible in the scatter plot (tighter clustering around the diagonal)

## How to Run

```bash
cd 03_parkinson_knn_regression
python knn_regression.py
```

## Files

| File | Description |
|------|-------------|
| `knn_regression.py` | Main analysis script with KNN-LLS implementation |
| `knn_regression.ipynb` | Jupyter notebook with inline outputs |
| `data/parkinsons_updrs.csv` | Dataset |
