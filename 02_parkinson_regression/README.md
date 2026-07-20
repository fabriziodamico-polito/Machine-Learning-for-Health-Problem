# Parkinson's Disease Progression - Linear Regression

## Objective

Predict `total_UPDRS`, a measure of Parkinson's disease severity, from remotely collected voice biomarkers. The lab compares a closed-form linear least-squares solution with an iterative steepest-descent solver.

The central evaluation question is whether the relationship learned from some patients transfers to people the model has never seen.

## Dataset

| Property | Value |
| --- | --- |
| Source | [UCI Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinson) |
| Repository subset | 990 recordings from 42 subjects |
| Predictors | Voice measures plus age and sex |
| Target | `total_UPDRS` |
| License | CC BY 4.0; see [DATASETS.md](../DATASETS.md) |

The upstream UCI dataset contains 5,875 recordings. The origin of the 990-row course subset stored here is not documented, so results should be interpreted as specific to this repository copy.

## Leakage-safe pipeline

```text
Load data -> split by subject (21 train / 21 test)
-> fit normalization on training patients only
-> remove identifiers, collinear features and target proxy
-> fit LLS and Steepest Descent -> evaluate on unseen patients
```

- A subject's repeated recordings remain in exactly one partition.
- `motor_UPDRS` is excluded because it is a closely related clinical score and would act as a target proxy.
- Normalization statistics are learned from the training partition only.
- A fixed seed makes the experiment reproducible.

## Verified results

| Model | Test MSE | Test R-squared | Pearson correlation |
| --- | ---: | ---: | ---: |
| Linear Least Squares | 251.67 | -1.29 | 0.07 |
| Steepest Descent | 242.31 | -1.20 | 0.06 |

The two solvers reach similar solutions, but neither generalizes well to unseen subjects. The negative test R-squared means that, on this split, both models perform worse than predicting the test-set mean. This is a useful result: a row-level random split would mix recordings from the same patients and make performance look substantially more optimistic.

## Run

```bash
cd 02_parkinson_regression
python parkinson_regression.py
```

| File | Description |
| --- | --- |
| `parkinson_regression.py` | Patient-isolated regression experiment |
| `data/parkinsons_updrs.csv` | Course subset used by the script |
