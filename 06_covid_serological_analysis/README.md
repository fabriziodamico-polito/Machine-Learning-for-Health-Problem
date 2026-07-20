# COVID-19 Serological Test Evaluation - ROC Analysis

## Objective

Compare two IgG serological measurements against a PCR swab reference using ROC curves, area under the curve (AUC) and Youden's J threshold selection. Bootstrap confidence intervals make the sampling uncertainty around each AUC visible.

## Dataset

| Property | Value |
| --- | --- |
| Retained observations | 829 after filtering uncertain labels and DBSCAN outliers |
| Measurements | `IgG_Test1_titre`, `IgG_Test2_titre` |
| Reference | `COVID_swab_res` (0 negative, 1 positive) |
| Provenance | Original study and redistribution terms are not documented; see [DATASETS.md](../DATASETS.md) |

## Pipeline

```text
Load table -> remove uncertain swab results
-> normalize each measurement -> DBSCAN outlier filtering
-> calculate ROC curve and AUC -> bootstrap 95% AUC interval
-> select an exploratory threshold with Youden's J
```

## Verified results

| Measurement | AUC | Bootstrap 95% CI | Youden threshold |
| --- | ---: | ---: | ---: |
| Test 1 | 0.948 | 0.910-0.978 | 7.59 |
| Test 2 | 0.938 | 0.900-0.968 | 0.30 |

Manual trapezoidal integration is checked against scikit-learn's AUC implementation. Thresholds are selected and evaluated on the same retrospective table, so they are exploratory rather than deployment-ready cutoffs. Independent cohorts and a prespecified protocol would be needed for clinical validation.

## Run

```bash
cd 06_covid_serological_analysis
python covid_roc_analysis.py
```

| File | Description |
| --- | --- |
| `covid_roc_analysis.py` | ROC analysis, bootstrap uncertainty and plots |
| `data/covid_serological_results.csv` | Serological measurement table |
