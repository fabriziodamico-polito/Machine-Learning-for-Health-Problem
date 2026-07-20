# Machine Learning for Health Labs

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> Laboratory assignments for the **Machine Learning for Health** course at [Politecnico di Torino](https://www.polito.it/), covering core Machine Learning techniques applied to real-world biomedical and signal processing problems.

---

## 📋 Overview

This repository contains 7 laboratory projects spanning the full ML pipeline — from optimization fundamentals to advanced signal processing. Each lab applies established ML techniques to a specific domain problem, with reproducible code and detailed documentation.

| #  | Lab                                                                 | Domain       | Techniques                                        | Dataset                                                                                       |
| -- | ------------------------------------------------------------------- | ------------ | ------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 01 | [Optimization Methods](./01_optimization_methods/)                   | Fundamentals | LLS, Gradient Descent, Steepest Descent           | Synthetic                                                                                     |
| 02 | [Parkinson Regression](./02_parkinson_regression/)                   | Neurology    | Linear Regression (LLS & SD), Feature Selection   | [Parkinsons Telemonitoring](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring) |
| 03 | [Parkinson KNN Regression](./03_parkinson_knn_regression/)           | Neurology    | KNN-LLS, Ridge Regression, Model Selection        | [Parkinsons Telemonitoring](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring) |
| 04 | [Mole Segmentation](./04_mole_segmentation/)                         | Dermatology  | K-Means, DBSCAN, Sobel Edge Detection             | Dermoscopic Images                                                                            |
| 05 | [Kidney Disease Classification](./05_kidney_disease_classification/) | Nephrology   | Decision Trees, Random Forest, LLS Imputation     | [Chronic Kidney Disease](https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease)       |
| 06 | [COVID Serological Analysis](./06_covid_serological_analysis/)       | Epidemiology | ROC Curves, AUC, Youden's J, DBSCAN               | COVID-19 Serological Study                                                                    |
| 07 | [EEG Signal Processing](./07_eeg_signal_processing/)                 | Neuroscience | CLT, Box-Muller, FastICA, PCA, Hypothesis Testing | Synthetic Signals                                                                             |

---

## 🔬 ML Techniques Map

### Regression

- **Linear Least Squares (LLS)** — closed-form solution via normal equations
- **Steepest Descent** — iterative optimization with adaptive step size
- **Gradient Descent** — fixed learning rate iterative minimization
- **KNN-LLS** — local linear regression on K nearest neighbors with ridge regularization

### Classification

- **Decision Trees (CART)** — entropy-based splitting for interpretable models
- **Random Forest** — ensemble learning with 100–1000 estimators
- **ROC Analysis** — sensitivity/specificity trade-off, AUC computation, threshold optimization

### Clustering & Segmentation

- **K-Means** — color quantization for image segmentation
- **DBSCAN** — density-based spatial clustering for mole detection and outlier removal

### Signal Processing

- **FastICA** — Independent Component Analysis for blind source separation
- **PCA** — Principal Component Analysis for dimensionality reduction
- **Sobel Filters** — gradient-based edge detection

### Statistical Methods

- **Central Limit Theorem** — approximate Gaussian generation from uniform RVs
- **Box-Muller Transform** — exact Gaussian sampling
- **Anderson-Darling Test** — goodness-of-fit for normality
- **Youden's J Statistic** — optimal threshold selection for binary classifiers

### Data Preprocessing

- **Z-Score Normalization** — mean/std standardization
- **Missing Data Imputation** — regression-based (LLS) and median imputation
- **Feature Selection** — correlation analysis, collinearity removal

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/fabriziodamico-polito/Machine-Learning-for-Health-Problem.git
cd Machine-Learning-for-Health-Problem

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running a Lab

Each lab is self-contained. Navigate to the desired folder and run:

```bash
cd 02_parkinson_regression
python parkinson_regression.py
```



---

## 📁 Project Structure

```
Machine-Learning-for-Health-Problem/
├── README.md
├── DATASETS.md
├── .gitignore
├── requirements.txt
├── utils/
│   ├── __init__.py
│   └── minimization.py            # Shared optimization solvers (LLS, GD, SD)
├── 01_optimization_methods/
│   └── optimization_demo.py
├── 02_parkinson_regression/
│   ├── parkinson_regression.py
│   └── data/
├── 03_parkinson_knn_regression/
│   ├── knn_regression.py
│   └── data/
├── 04_mole_segmentation/
│   ├── mole_segmentation.py
│   └── data/images/
├── 05_kidney_disease_classification/
│   ├── kidney_classification.py
│   └── data/
├── 06_covid_serological_analysis/
│   ├── covid_roc_analysis.py
│   └── data/
└── 07_eeg_signal_processing/
    ├── central_limit_theorem.py
    └── fastica_bss.py
```

---

## 🛠️ Shared Utilities

The [`utils/minimization.py`](./utils/minimization.py) module provides a reusable OOP framework for solving linear minimization problems:

| Class              | Method           | Description                                                     |
| ------------------ | ---------------- | --------------------------------------------------------------- |
| `SolveMinProbl`  | —               | Abstract base class with common plotting and printing           |
| `SolveLLS`       | Closed-form      | $(A^T A)^{-1} A^T y$                                          |
| `SolveGrad`      | Gradient Descent | Fixed learning rate$\gamma$                                   |
| `SolveSteepDesc` | Steepest Descent | Adaptive$\gamma_k = \|\nabla J\|^2 / (\nabla J^T H \nabla J)$ |

---

## 📚 Data and Responsible Use

Dataset sources, citations and known redistribution terms are documented in [DATASETS.md](./DATASETS.md).

This repository is an educational project. It is not clinically validated and must not be used for diagnosis or treatment decisions. The repository currently does not grant a software license.
