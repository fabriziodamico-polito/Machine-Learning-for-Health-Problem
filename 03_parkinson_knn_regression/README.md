# Parkinson's Disease Progression - Local KNN-LLS Regression

## Objective

Test whether a local model can predict `total_UPDRS` better than one global linear model. For each observation, KNN-LLS finds nearby training samples and fits a regularized linear regression only on that neighborhood.

## Dataset

| Property | Value |
| --- | --- |
| Source | [UCI Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinson) |
| Repository subset | 990 recordings from 42 subjects |
| Predictors | Voice measures plus age and sex |
| Target | `total_UPDRS` |
| License | CC BY 4.0; see [DATASETS.md](../DATASETS.md) |

The 990-row repository copy is a subset of the 5,875-recording UCI dataset. Its derivation is currently unknown.

## Method

```text
Split subjects into train / validation / test (16 / 8 / 18)
-> fit normalization on training subjects
-> choose K and ridge strength on validation subjects
-> evaluate KNN-LLS and global LLS on untouched test subjects
```

- Patient groups, rather than individual recordings, define every split.
- Every local model includes an intercept; a ridge term stabilizes its slopes.
- `motor_UPDRS`, identifiers and collinear derived measures are excluded.
- The validation set selects `K=20` and ridge strength `1.0`; the test set is used only for final evaluation.

## Verified results

| Model | Test MSE | Test R-squared |
| --- | ---: | ---: |
| KNN-LLS | 210.42 | -0.68 |
| Global LLS | 244.94 | -0.95 |

KNN-LLS reduces test MSE relative to global LLS on this split, but both test R-squared values remain negative. The local model therefore improves this baseline without becoming a useful unseen-patient predictor. Because only 42 subjects are available, the comparison is evidence for this deterministic split rather than a general conclusion about KNN.

## Run

```bash
cd 03_parkinson_knn_regression
python knn_regression.py
```

| File | Description |
| --- | --- |
| `knn_regression.py` | Patient-isolated KNN-LLS experiment and global baseline |
| `data/parkinsons_updrs.csv` | Course subset used by the script |
