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
        self.assertNotIn("test_time", lab.regressors)

    def test_knn_uses_disjoint_patient_partitions(self):
        lab = self.knn.UPDRS()
        lab.load_and_explore(plot=False)
        lab.prepare_data()
        self.assertFalse(lab.train_subjects & lab.validation_subjects)
        self.assertFalse(lab.train_subjects & lab.test_subjects)
        self.assertFalse(lab.validation_subjects & lab.test_subjects)
        self.assertNotIn("motor_UPDRS", lab.regressors)


class NumericalMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optimization = load_module(
            "optimization_demo",
            "01_optimization_methods/optimization_demo.py",
        )
        cls.knn = load_module(
            "parkinson_knn_intercept",
            "03_parkinson_knn_regression/knn_regression.py",
        )

    def test_all_optimization_solvers_recover_the_noiseless_solution(self):
        result = self.optimization.run_demo(plot=False)
        for name, error in result['coefficient_errors'].items():
            self.assertLess(error, 1e-6, msg=f'{name} did not converge')

    def test_local_regression_includes_an_intercept(self):
        lab = self.knn.UPDRS()
        lab.eps = 1e-8
        lab.X_tr_norm = np.arange(8, dtype=float).reshape(-1, 1)
        lab.y_tr_norm = 5 + 2 * lab.X_tr_norm[:, 0]
        prediction = lab.local_prediction(np.array([3.5]), K=8)
        self.assertAlmostEqual(prediction, 12.0, places=6)


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


class AnalysisConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.covid = load_module(
            "covid_roc_analysis",
            "06_covid_serological_analysis/covid_roc_analysis.py",
        )
        cls.fastica = load_module(
            "fastica_bss",
            "07_eeg_signal_processing/fastica_bss.py",
        )
        cls.moles = load_module(
            "mole_segmentation",
            "04_mole_segmentation/mole_segmentation.py",
        )

    def test_roc_operating_points_use_exhaustive_threshold_rules(self):
        analysis = self.covid.Covid()
        analysis.DataAnalysis('Test1')
        analysis.Sensitivity_Specificity()
        finite = np.flatnonzero(np.isfinite(analysis.thresholds))
        best = finite[np.argmax((analysis.sensitivity - analysis.FPR)[finite])]
        threshold = analysis.thresholds[best]
        expected_sensitivity = np.mean(
            analysis.Test_value[analysis.swab == 1] >= threshold
        )
        expected_specificity = np.mean(
            analysis.Test_value[analysis.swab == 0] < threshold
        )
        self.assertAlmostEqual(analysis.sensitivity[best], expected_sensitivity)
        self.assertAlmostEqual(analysis.specificity[best], expected_specificity)

    def test_fastica_recovers_the_synthetic_sources_after_alignment(self):
        result = self.fastica.run_experiment(n_samples=5000, plot=False)
        self.assertGreater(result['ica_correlations'].min(), 0.90)
        self.assertGreater(
            result['ica_correlations'].mean(),
            result['pca_correlations'].mean(),
        )

    def test_mole_pipeline_returns_a_finite_nonempty_border(self):
        lab = self.moles.Moles(plotfig=False, image_title='medium_risk_4.jpg')
        border = lab.run()
        self.assertGreater(border.size, 0)
        self.assertTrue(np.isfinite(border).all())
        self.assertGreater(float(border.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
