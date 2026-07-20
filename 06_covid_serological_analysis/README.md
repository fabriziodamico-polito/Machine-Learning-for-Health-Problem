# COVID-19 Serological Test Evaluation — ROC Analysis

## Objective

Evaluate the diagnostic performance of two **IgG serological tests** for COVID-19 by computing **ROC curves**, **AUC scores**, and determining the **optimal classification threshold** using Youden's J statistic.

## Dataset

| Property | Value |
|----------|-------|
| **Source** | COVID-19 serological study |
| **Samples** | ~300 patients (after outlier removal) |
| **Features** | `IgG_Test1_titre`, `IgG_Test2_titre` |
| **Ground Truth** | `COVID_swab_res` (PCR swab result: 0=negative, 1=positive) |

## Techniques

| Method | Description |
|--------|-------------|
| **DBSCAN Outlier Removal** | Density-based filtering on normalized IgG values to remove measurement anomalies |
| **Sensitivity & Specificity** | Computed across all possible thresholds |
| **ROC Curve** | Trade-off visualization: True Positive Rate vs False Positive Rate |
| **AUC** | Area Under the ROC Curve — computed both manually (trapezoidal rule) and via scikit-learn |
| **Youden's J Statistic** | $J = \text{Sensitivity} - \text{FPR}$, maximized to find the optimal threshold |

## Pipeline

```
Load CSV → Remove Uncertain Swab Results → DBSCAN Outlier Removal
→ Compute Sensitivity & Specificity for All Thresholds
→ Plot ROC Curve → Compute AUC → Optimize Threshold (Youden's J)
→ Compare Test 1 vs Test 2
```

## Key Results

- Both tests show strong discriminative ability (AUC > 0.9)
- **Test 2** generally achieves a higher AUC than Test 1
- The optimal threshold balances sensitivity and specificity according to Youden's criterion
- Manual AUC computation matches scikit-learn's implementation, validating the methodology

## How to Run

```bash
cd 06_covid_serological_analysis
python covid_roc_analysis.py
```

## Files

| File | Description |
|------|-------------|
| `covid_roc_analysis.py` | Main ROC analysis pipeline |
| `covid_roc_analysis.ipynb` | Jupyter notebook with inline plots |
| `data/covid_serological_results.csv` | Serological test results |
