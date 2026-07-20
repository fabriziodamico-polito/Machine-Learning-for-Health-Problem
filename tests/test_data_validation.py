import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np


os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParkinsonSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.linear = load_module(
            "parkinson_regression",
            "02_parkinson_regression/parkinson_regression.py",
        )
        cls.knn = load_module(
            "parkinson_knn_regression",
            "03_parkinson_knn_regression/knn_regression.py",
        )

    def test_linear_regression_uses_disjoint_patients(self):
        lab = self.linear.ParkinsonUPDRS()
        lab.load_and_explore(plot=False)
        lab.prepare_data(include_motor=False)
        self.assertFalse(lab.train_subjects & lab.test_subjects)
        self.assertNotIn("motor_UPDRS", lab.regressors)

    def test_knn_uses_disjoint_patient_partitions(self):
        lab = self.knn.UPDRS()
        lab.load_and_explore(plot=False)
        lab.prepare_data()
        self.assertFalse(lab.train_subjects & lab.validation_subjects)
        self.assertFalse(lab.train_subjects & lab.test_subjects)
        self.assertFalse(lab.validation_subjects & lab.test_subjects)
        self.assertNotIn("motor_UPDRS", lab.regressors)


class KidneyPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kidney = load_module(
            "kidney_classification",
            "05_kidney_disease_classification/kidney_classification.py",
        )

    def test_imputation_is_fit_after_a_disjoint_split(self):
        path = ROOT / "05_kidney_disease_classification/data/chronic_kidney_disease.arff"
        lab = self.kidney.ChronicKidneyDiseaseLab(path)
        lab.load_and_preprocess()
        lab.split_data()
        self.assertFalse(lab.train_ids & lab.test_ids)

        for strategy in ("median", "regression"):
            train, test = lab.impute(strategy)
            self.assertTrue(np.isfinite(train).all())
            self.assertTrue(np.isfinite(test).all())
            self.assertEqual(train.shape[1], len(lab.feature_names))


if __name__ == "__main__":
    unittest.main()
