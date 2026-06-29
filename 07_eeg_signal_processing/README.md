# EEG Signal Processing — Gaussianity Tests & Blind Source Separation

## Objective

This lab covers two fundamental topics in biomedical signal processing:

1. **Gaussianity Testing**: Generate pseudo-Gaussian samples using the **Central Limit Theorem** and **Box-Muller** methods, then rigorously verify their Gaussianity through statistical tests
2. **Blind Source Separation (BSS)**: Recover independent source signals from their linear mixtures using **FastICA**, and compare against **PCA**

## Techniques

### Part 1 — Gaussianity Testing

| Method | Description |
|--------|-------------|
| **Central Limit Theorem (CLT)** | Sum of N uniform random variables approximates a Gaussian |
| **Box-Muller Transform** | Exact method to generate Gaussian samples from uniform ones |
| **Normal Probability Plot** | Q-Q plot to visually assess Gaussianity |
| **t-Score Test** | Tests whether the sample mean matches the expected mean |
| **Excess Kurtosis** | Measures deviation from Gaussian tail behavior (should be ≈ 0) |
| **Anderson-Darling Test** | Powerful goodness-of-fit test for normality |
| **p-Value Estimation** | Monte Carlo simulation (10,000 experiments) to estimate p-value curves |

### Part 2 — Blind Source Separation

| Method | Description |
|--------|-------------|
| **FastICA** | Independent Component Analysis using deflation algorithm; maximizes non-Gaussianity |
| **PCA** | Principal Component Analysis; finds orthogonal components (decorrelation only) |

## Pipeline

### Part 1
```
Generate Uniform Samples → Sum (CLT) or Transform (Box-Muller)
→ Histogram vs Theoretical PDF → CDF Comparison
→ Normal Probability Plot → t-Score, Kurtosis, Anderson-Darling
→ Monte Carlo p-Value Estimation
```

### Part 2
```
Generate 4 Source Signals (sin, square, sawtooth, triangular)
→ Mix with Random Matrix A → Apply FastICA → Apply PCA
→ Compare Recovered vs Original Signals
→ Compare Estimated vs True Unmixing Matrix W
```

## Key Results

- **CLT** produces increasingly Gaussian samples as N grows; N=10 already passes most tests
- **Box-Muller** generates exact Gaussian samples (passes all tests)
- **FastICA** successfully recovers all 4 independent sources (up to sign/permutation ambiguity)
- **PCA** fails to separate non-orthogonal sources — it only decorrelates, not makes independent
- The estimated unmixing matrix $\hat{W}$ closely matches the true $W = A^{-1}$

## How to Run

```bash
cd 07_eeg_signal_processing

# Part 1: Gaussianity tests
python central_limit_theorem.py

# Part 2: Blind Source Separation
python fastica_bss.py
```

## Files

| File | Description |
|------|-------------|
| `central_limit_theorem.py` | CLT, Box-Muller, and statistical tests |
| `central_limit_theorem.ipynb` | Notebook with inline visualizations |
| `fastica_bss.py` | FastICA vs PCA for source separation |
| `fastica_bss.ipynb` | Notebook with inline visualizations |
