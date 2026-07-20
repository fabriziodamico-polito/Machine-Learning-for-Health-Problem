# COVID-19 Serological Test Evaluation - ROC Analysis

## Objective

Compare two IgG serological measurements against a PCR swab reference using ROC curves, area under the curve (AUC) and Youden's J threshold selection. Bootstrap confidence intervals make the sampling uncertainty around each AUC visible.

## Dataset

| Property | Value |
| --- | --- |
| Primary observations | 862 with known binary swab labels |
| Sensitivity cohort | 829 after DBSCAN outlier filtering |
| Measurements | `IgG_Test1_titre`, `IgG_Test2_titre` |
| Reference | `COVID_swab_res` (0 negative, 1 positive) |
| Provenance | Original study and redistribution terms are not documented; see [DATASETS.md](../DATASETS.md) |

## Pipeline

```text
Load table -> remove uncertain swab results
-> calculate primary ROC/AUC on every known-label sample
-> stratified bootstrap 95% AUC interval -> exploratory Youden threshold
-> report DBSCAN-filtered AUC separately as a sensitivity analysis
```

## Verified results

| Measurement | Primary AUC | Bootstrap 95% CI | Youden threshold | DBSCAN-filtered AUC |
| --- | ---: | ---: | ---: | ---: |
| Test 1 | 0.943 | 0.913-0.968 | 7.71 | 0.948 |
| Test 2 | 0.936 | 0.906-0.961 | 0.32 | 0.938 |

Manual trapezoidal integration is checked against scikit-learn's AUC implementation. ROC operating points use the same inclusive threshold convention as scikit-learn. Thresholds are still selected and evaluated on the same retrospective table, so they remain exploratory rather than deployment-ready cutoffs. Independent cohorts and a prespecified protocol would be needed for clinical validation.

## Run

```bash
cd 06_covid_serological_analysis
python covid_roc_analysis.py
```

| File | Description |
| --- | --- |
| `covid_roc_analysis.py` | ROC analysis, bootstrap uncertainty and plots |
| `data/covid_serological_results.csv` | Serological measurement table |
