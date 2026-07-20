import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42


class ChronicKidneyDiseaseLab:
    """Leakage-safe comparison of two missing-data strategies for CKD."""

    def __init__(self, filepath, seed=RANDOM_SEED):
        self.filepath = filepath
        self.seed = seed
        self.feature_names = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
            'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
            'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane',
        ]
        self.column_names = [*self.feature_names, 'classk']
        self.categorical_features = {
            'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm',
            'cad', 'appet', 'pe', 'ane',
        }
        self.target_names = ['notckd', 'ckd']
        self.mapping = {
            'normal': 0,
            'abnormal': 1,
            'present': 1,
            'notpresent': 0,
            'yes': 1,
            ' yes': 1,
            'no': 0,
            '\tno': 0,
            '\tyes': 1,
            'ckd': 1,
            'notckd': 0,
            'poor': 1,
            'good': 0,
            'ckd\t': 1,
        }
        self.results = []

    def load_and_preprocess(self):
        data = pd.read_csv(
            self.filepath,
            sep=',',
            skiprows=29,
            names=self.column_names,
            header=None,
            na_values=['?', '\t?'],
        )
        data = data.replace(self.mapping)
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.dropna(subset=['classk']).reset_index(drop=True)
        data['classk'] = data['classk'].astype(int)
        self.data = data

        print(f"Patients: {len(data)}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Missing predictor values: {int(data[self.feature_names].isna().sum().sum())}")

    def split_data(self, test_size=0.3):
        row_ids = np.arange(len(self.data))
        train_ids, test_ids = train_test_split(
            row_ids,
            test_size=test_size,
            random_state=self.seed,
            stratify=self.data['classk'],
        )
        self.train_ids = set(train_ids.tolist())
        self.test_ids = set(test_ids.tolist())
        if self.train_ids & self.test_ids:
            raise RuntimeError("Train/test leakage detected")

        train = self.data.iloc[train_ids]
        test = self.data.iloc[test_ids]
        self.X_train = train[self.feature_names]
        self.X_test = test[self.feature_names]
        self.y_train = train['classk'].to_numpy()
        self.y_test = test['classk'].to_numpy()
        print(f"Leakage-free split: {len(train)} train / {len(test)} test patients")

    def _snap_categorical_values(self, train_values, test_values):
        for column in self.categorical_features:
            index = self.feature_names.index(column)
            valid_values = np.sort(self.X_train[column].dropna().unique())
            if len(valid_values) == 0:
                continue
            for values in (train_values, test_values):
                distances = np.abs(values[:, index, None] - valid_values[None, :])
                values[:, index] = valid_values[np.argmin(distances, axis=1)]
        return train_values, test_values

    def impute(self, strategy):
        if strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif strategy == 'regression':
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                initial_strategy='median',
                max_iter=20,
                random_state=self.seed,
                skip_complete=True,
            )
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")

        # Fit only on the training partition, then transform the untouched test set.
        X_train = imputer.fit_transform(self.X_train)
        X_test = imputer.transform(self.X_test)
        return self._snap_categorical_values(X_train, X_test)

    def evaluate(self, model_name, model, strategy, X_train, X_test):
        model.fit(X_train, self.y_train)
        predictions = model.predict(X_test)
        result = {
            'imputation': strategy,
            'model': model_name,
            'accuracy': accuracy_score(self.y_test, predictions),
            'balanced_accuracy': balanced_accuracy_score(self.y_test, predictions),
            'f1': f1_score(self.y_test, predictions),
            'confusion_matrix': confusion_matrix(self.y_test, predictions),
        }
        self.results.append(result)

        print(f"\n{strategy.title()} imputation + {model_name}")
        print(f"Accuracy:          {result['accuracy']:.3f}")
        print(f"Balanced accuracy: {result['balanced_accuracy']:.3f}")
        print(f"F1:                {result['f1']:.3f}")
        print(result['confusion_matrix'])
        return model

    def plot_feature_importance(self, model, strategy):
        order = np.argsort(model.feature_importances_)[::-1]
        plt.figure(figsize=(12, 6))
        plt.title(f"Random Forest feature importance — {strategy} imputation")
        plt.bar(range(len(order)), model.feature_importances_[order])
        plt.xticks(
            range(len(order)),
            [self.feature_names[index] for index in order],
            rotation=90,
        )
        plt.tight_layout()
        plt.savefig(f"feature_importance_{strategy}.png")
        plt.close()

    def run(self):
        self.load_and_preprocess()
        self.split_data()

        for strategy in ('median', 'regression'):
            X_train, X_test = self.impute(strategy)
            decision_tree = tree.DecisionTreeClassifier(
                criterion='entropy',
                max_depth=5,
                random_state=self.seed,
            )
            self.evaluate(
                'Decision Tree',
                decision_tree,
                strategy,
                X_train,
                X_test,
            )

            random_forest = RandomForestClassifier(
                n_estimators=500,
                class_weight='balanced',
                random_state=self.seed,
            )
            random_forest = self.evaluate(
                'Random Forest',
                random_forest,
                strategy,
                X_train,
                X_test,
            )
            self.plot_feature_importance(random_forest, strategy)

        return self.results


if __name__ == '__main__':
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        'data',
        'chronic_kidney_disease.arff',
    )
    ChronicKidneyDiseaseLab(dataset_path).run()
