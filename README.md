# Machine Learning for Health Labs

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.5-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![CI](https://github.com/fabriziodamico-polito/Machine-Learning-for-Health-Problem/actions/workflows/ci.yml/badge.svg)](https://github.com/fabriziodamico-polito/Machine-Learning-for-Health-Problem/actions/workflows/ci.yml)

> Laboratory projects for the **Machine Learning for Health** course at [Politecnico di Torino](https://www.polito.it/), covering core machine-learning techniques applied to biomedical data, medical images and signal-processing problems.

---

## 📋 Overview

This repository contains seven laboratory projects spanning the machine-learning pipeline, from numerical optimization to classification, image segmentation and blind source separation. Each lab is self-contained and includes dedicated documentation, reproducible Python code and its required data.

| # | Lab | Domain | Techniques | Dataset |
| --- | --- | --- | --- | --- |
| 01 | [Optimization Methods](./01_optimization_methods/) | Fundamentals | Least Squares, Gradient Descent, Steepest Descent | Synthetic |
| 02 | [Parkinson Regression](./02_parkinson_regression/) | Neurology | Linear Regression, Feature Selection, Patient-Level Validation | [Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinson) |
| 03 | [Parkinson KNN Regression](./03_parkinson_knn_regression/) | Neurology | KNN-LLS, Ridge Regularization, Model Selection | [Parkinsons Telemonitoring](https://archive.ics.uci.edu/dataset/189/parkinson) |
| 04 | [Mole Segmentation](./04_mole_segmentation/) | Dermatology | K-Means, DBSCAN, Sobel Edge Detection | Dermoscopic Images |
| 05 | [Kidney Disease Classification](./05_kidney_disease_classification/) | Nephrology | Decision Trees, Random Forest, Missing-Data Imputation | [Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) |
| 06 | [COVID Serological Analysis](./06_covid_serological_analysis/) | Epidemiology | ROC Curves, AUC, Youden's J, Bootstrap | COVID-19 Serological Study |
| 07 | [Signal Processing](./07_eeg_signal_processing/) | Neuroscience | CLT, Box-Muller, FastICA, PCA, Hypothesis Testing | Synthetic Signals |

---

## 🔬 ML Techniques Map

### Regression and Optimization

- **Linear Least Squares (LLS)** — stable least-squares estimation through an SVD-based numerical solver
- **Gradient Descent** — fixed-step iterative minimization
- **Steepest Descent** — iterative optimization with an adaptive step size
- **KNN-LLS** — local linear regression on nearest neighbors with ridge stabilization

### Classification

- **Decision Trees (CART)** — entropy-based splitting for interpretable classification
- **Random Forest** — ensemble classification for chronic kidney disease prediction
- **ROC Analysis** — sensitivity/specificity trade-off, AUC estimation and threshold exploration

### Clustering and Segmentation

- **K-Means** — grayscale quantization for candidate lesion segmentation
- **DBSCAN** — spatial clustering for mole detection and serology sensitivity analysis
- **Sobel Filters** — gradient-based border extraction

### Signal Processing and Statistics

- **FastICA** — Independent Component Analysis for blind source separation
- **PCA** — decorrelation baseline for source recovery
- **Central Limit Theorem** — approximate Gaussian generation from uniform random variables
- **Box-Muller Transform** — direct Gaussian sampling
- **Anderson-Darling Test** — goodness-of-fit testing for normality
- **Youden's J Statistic** — exploratory threshold selection for binary tests

### Data Preprocessing

- **Z-Score Normalization** — training-set mean and standard-deviation scaling
- **Missing-Data Imputation** — median and iterative regression strategies
- **Feature Selection** — identifier, target-proxy and collinearity removal

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/fabriziodamico-polito/Machine-Learning-for-Health-Problem.git
cd Machine-Learning-for-Health-Problem

# Create a virtual environment
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the tested dependencies:

```bash
python -m pip install -r requirements.txt
```

### Running a Lab

Each lab can be executed from its folder:

```bash
cd 02_parkinson_regression
python parkinson_regression.py
```

Run the complete validation suite from the repository root:

```bash
python -m unittest discover -s tests -v
python scripts/run_all_labs.py
python scripts/check_secrets.py
```

---

## 📁 Project Structure

```text
Machine-Learning-for-Health-Problem/
├── README.md
├── DATASETS.md
├── requirements.txt
├── utils/
│   └── minimization.py
├── scripts/
│   ├── check_secrets.py
│   └── run_all_labs.py
├── tests/
│   └── test_data_validation.py
├── 01_optimization_methods/
├── 02_parkinson_regression/
├── 03_parkinson_knn_regression/
├── 04_mole_segmentation/
├── 05_kidney_disease_classification/
├── 06_covid_serological_analysis/
└── 07_eeg_signal_processing/
```

---

## 🛠️ Shared Utilities

The [`utils/minimization.py`](./utils/minimization.py) module provides a reusable object-oriented framework for linear minimization problems:

| Class | Method | Description |
| --- | --- | --- |
| `SolveMinProbl` | Base class | Common plotting and result utilities |
| `SolveLLS` | Least Squares | Stable SVD-based solution |
| `SolveGrad` | Gradient Descent | Fixed learning rate |
| `SolveSteepDesc` | Steepest Descent | Adaptive step size |

---

## ✅ Validation

- Repeated Parkinson recordings are separated by patient across data partitions.
- Normalization and missing-data imputation are learned from training data only.
- Numerical convergence, preprocessing boundaries and source recovery are covered by automated tests.
- GitHub Actions compiles the code, runs the test suite and executes every laboratory script.

Detailed assumptions, results and limitations are documented inside each laboratory folder.

---

## 📚 Data and Responsible Use

Dataset sources, citations and known redistribution terms are documented in [DATASETS.md](./DATASETS.md). The provenance of the dermoscopic images and COVID-19 table still requires confirmation.

This repository is an educational project. It is not clinically validated and must not be used for diagnosis or treatment decisions. The repository currently does not grant a software license.
