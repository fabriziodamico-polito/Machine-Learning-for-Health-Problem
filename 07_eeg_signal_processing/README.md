# Signal Processing - Gaussianity and Blind Source Separation

## Objective

Explore two foundations of biomedical signal processing with reproducible synthetic data:

1. generate and test Gaussian samples with the Central Limit Theorem and Box-Muller transform;
2. separate linearly mixed sources with FastICA and compare the result with PCA.

## Part 1: Gaussianity

| Method | Role |
| --- | --- |
| Central Limit Theorem | Approximates a Gaussian by summing uniform variables |
| Box-Muller transform | Generates Gaussian samples directly from uniform variables |
| Q-Q plot and CDF | Visual distribution checks |
| t-score and excess kurtosis | Checks center and tail shape |
| Anderson-Darling statistic | Goodness-of-fit test for normality |
| Monte Carlo simulation | Estimates reference p-value behavior |

## Part 2: Blind source separation

Three non-Gaussian waveforms and an independent Laplace-noise source are standardized and mixed through a seeded random matrix. FastICA searches for statistically independent components, while PCA provides a decorrelation baseline. Recovered components are optimally aligned before scoring to resolve permutation, sign and scale ambiguities.

```text
Generate sources -> mix through matrix A
-> apply FastICA and PCA -> align and compare recovered components
```

| Method | Aligned absolute source correlations | Mean |
| --- | --- | ---: |
| FastICA | `0.9994, 1.0000, 0.9997, 0.9999` | `0.9998` |
| PCA | `0.8345, 0.7320, 0.6490, 0.9260` | `0.7854` |

The experiment quantitatively illustrates why decorrelation alone is insufficient for separating these non-Gaussian sources. Because the inputs are synthetic, the result is a methods demonstration rather than validation on recorded EEG.

## Run

```bash
cd 07_eeg_signal_processing
python central_limit_theorem.py
python fastica_bss.py
```

| File | Description |
| --- | --- |
| `central_limit_theorem.py` | Gaussian generation and statistical checks |
| `fastica_bss.py` | FastICA/PCA source-separation comparison |
