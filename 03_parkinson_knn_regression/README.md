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
-> choose K on validation subjects
-> evaluate KNN-LLS and global LLS on untouched test subjects
```

- Patient groups, rather than individual recordings, define every split.
- A small ridge term stabilizes each local least-squares system.
- `motor_UPDRS`, identifiers and collinear derived measures are excluded.
- The validation set selects K; the test set is used only for final evaluation.

## Verified results

| Model | Test MSE | Test R-squared |
| --- | ---: | ---: |
| KNN-LLS | 332.46 | -1.65 |
| Global LLS | 246.54 | -0.97 |

KNN-LLS fits the training observations more closely (training R-squared `0.80`) but performs worse than global LLS on unseen patients. The gap is evidence that local similarity among repeated recordings does not automatically translate into patient-level generalization. Reporting that limitation is more informative than preserving the high scores produced by a leaked row-level split.

## Run

```bash
cd 03_parkinson_knn_regression
python knn_regression.py
```

| File | Description |
| --- | --- |
| `knn_regression.py` | Patient-isolated KNN-LLS experiment and global baseline |
| `data/parkinsons_updrs.csv` | Course subset used by the script |
