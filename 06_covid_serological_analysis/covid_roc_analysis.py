"""Evaluate two serological measurements against a binary swab reference."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler


RANDOM_SEED = 42


class Covid:
    def __init__(self):
        self.data = os.path.join(
            os.path.dirname(__file__),
            'data',
            'covid_serological_results.csv',
        )
        self.results = {}

    def DataAnalysis(self, test_name):
        """Prepare all known-label observations and a filtered sensitivity set."""
        self.Test = test_name
        data = pd.read_csv(self.data)
        data = data[data.COVID_swab_res != 1].copy()
        data.loc[data.COVID_swab_res == 2, 'COVID_swab_res'] = 1

        value_column = {
            'Test1': 'IgG_Test1_titre',
            'Test2': 'IgG_Test2_titre',
        }.get(test_name)
        if value_column is None:
            raise ValueError(f'Unknown test name: {test_name}')

        self.swab = data.COVID_swab_res.to_numpy(dtype=int)
        self.Test_value = data[value_column].to_numpy(dtype=float)

        # DBSCAN is retained as a sensitivity analysis, not as the primary
        # reported cohort. This prevents an arbitrary outlier rule from being
        # hidden inside the headline diagnostic result.
        measurement_columns = ['IgG_Test1_titre', 'IgG_Test2_titre']
        normalized = MinMaxScaler().fit_transform(data[measurement_columns])
        inlier_mask = DBSCAN(eps=0.05, min_samples=5).fit_predict(normalized) != -1
        self.filtered_swab = self.swab[inlier_mask]
        self.filtered_values = self.Test_value[inlier_mask]

        print(f'{self.Test}: primary cohort contains {len(data)} known-label samples')
        print(
            f'{self.Test}: DBSCAN sensitivity cohort retains '
            f'{int(inlier_mask.sum())} samples'
        )

    def Sensitivity_Specificity(self):
        """Compute internally consistent ROC operating points."""
        fpr, tpr, thresholds = metrics.roc_curve(
            self.swab,
            self.Test_value,
            pos_label=1,
            drop_intermediate=False,
        )
        self.FPR = fpr
        self.sensitivity = tpr
        self.specificity = 1 - fpr
        self.thresholds = thresholds

        finite = np.isfinite(thresholds)
        order = np.argsort(thresholds[finite])
        plt.figure()
        plt.plot(
            thresholds[finite][order],
            tpr[finite][order],
            label='Sensitivity',
        )
        plt.plot(
            thresholds[finite][order],
            (1 - fpr[finite])[order],
            label='Specificity',
        )
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title(self.Test)
        plt.grid()
        plt.legend()

    def ROC(self):
        """Report primary and outlier-filtered AUC values."""
        auc = roc_auc_score(self.swab, self.Test_value)
        auc_manual = np.trapezoid(self.sensitivity, self.FPR)
        filtered_auc = roc_auc_score(self.filtered_swab, self.filtered_values)
        ci_low, ci_high = self.bootstrap_auc_ci()

        print(f'Manual AUC: {auc_manual:.4f}')
        print(f'Primary AUC: {auc:.4f} (stratified bootstrap 95% CI: {ci_low:.4f}-{ci_high:.4f})')
        print(f'DBSCAN-filtered sensitivity AUC: {filtered_auc:.4f}')

        self.results[self.Test] = {
            'sample_count': len(self.swab),
            'filtered_sample_count': len(self.filtered_swab),
            'auc': float(auc),
            'auc_manual': float(auc_manual),
            'filtered_auc': float(filtered_auc),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
        }

        plt.figure()
        plt.plot(self.FPR, self.sensitivity, label=f'ROC (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve - {self.Test}')
        plt.grid()
        plt.legend()

    def bootstrap_auc_ci(self, n_bootstrap=2000):
        """Stratified bootstrap interval conditional on the primary cohort."""
        rng = np.random.default_rng(RANDOM_SEED)
        positive_values = self.Test_value[self.swab == 1]
        negative_values = self.Test_value[self.swab == 0]
        scores = np.empty(n_bootstrap, dtype=float)

        labels = np.concatenate(
            [np.ones(len(positive_values)), np.zeros(len(negative_values))]
        )
        for index in range(n_bootstrap):
            values = np.concatenate(
                [
                    rng.choice(positive_values, len(positive_values), replace=True),
                    rng.choice(negative_values, len(negative_values), replace=True),
                ]
            )
            scores[index] = roc_auc_score(labels, values)
        return tuple(np.percentile(scores, [2.5, 97.5]))

    def SetThreshold(self):
        """Choose an exploratory threshold with Youden's J statistic."""
        youden = self.sensitivity - self.FPR
        valid_indices = np.flatnonzero(np.isfinite(self.thresholds))
        best_index = valid_indices[np.argmax(youden[valid_indices])]
        threshold = float(self.thresholds[best_index])
        sensitivity = float(self.sensitivity[best_index])
        specificity = float(self.specificity[best_index])

        self.results[self.Test].update(
            {
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
            }
        )
        print(
            f'Exploratory Youden threshold: {threshold:.3f} '
            f'(sensitivity={sensitivity:.3f}, specificity={specificity:.3f})'
        )

        plt.figure()
        plt.plot(self.FPR, self.sensitivity, label='ROC curve')
        plt.scatter(
            self.FPR[best_index],
            self.sensitivity[best_index],
            color='red',
            label='Youden operating point',
        )
        plt.grid()
        plt.legend()
        return threshold, sensitivity, specificity


if __name__ == '__main__':
    analysis = Covid()
    for test_name in ('Test1', 'Test2'):
        print(f'\n--- {test_name} ---')
        analysis.DataAnalysis(test_name)
        analysis.Sensitivity_Specificity()
        analysis.ROC()
        analysis.SetThreshold()

    print('\n--- Comparison ---')
    for test_name, result in analysis.results.items():
        print(
            f"{test_name}: AUC={result['auc']:.3f}, "
            f"threshold={result['threshold']:.3f}, "
            f"sensitivity={result['sensitivity']:.3f}, "
            f"specificity={result['specificity']:.3f}"
        )
    plt.show()
