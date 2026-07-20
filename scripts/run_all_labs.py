"""Run every repository experiment in headless mode as a smoke test."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LABS = [
    ("Optimization", "01_optimization_methods", "optimization_demo.py"),
    ("Parkinson linear", "02_parkinson_regression", "parkinson_regression.py"),
    ("Parkinson KNN", "03_parkinson_knn_regression", "knn_regression.py"),
    ("Mole segmentation", "04_mole_segmentation", "mole_segmentation.py"),
    ("Kidney classification", "05_kidney_disease_classification", "kidney_classification.py"),
    ("COVID ROC", "06_covid_serological_analysis", "covid_roc_analysis.py"),
    ("Central limit theorem", "07_eeg_signal_processing", "central_limit_theorem.py"),
    ("FastICA", "07_eeg_signal_processing", "fastica_bss.py"),
]


def main():
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(ROOT), existing_pythonpath) if item
    )
    failures = []

    for name, directory, script in LABS:
        result = subprocess.run(
            [sys.executable, script],
            cwd=ROOT / directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            print(f"PASS: {name}")
        else:
            failures.append(name)
            print(f"FAIL: {name}")
            print(result.stderr[-2000:])

    if failures:
        print(f"Failed labs: {', '.join(failures)}")
        return 1
    print(f"All {len(LABS)} experiment scripts passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
