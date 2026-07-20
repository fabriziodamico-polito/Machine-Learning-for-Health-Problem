import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import minimization as mymin

RANDOM_SEED = 42
csv_path = os.path.join(os.path.dirname(__file__), "data", "parkinsons_updrs.csv")

class UPDRS:
    def __init__(self, seed=RANDOM_SEED):
        self.seed = seed
        self.eps = 1e-8
        self.metrics = {}
        
        # Output settings
        pd.set_option('display.precision', 3)
        plt.close('all')

    # Loads data and plots the covariance matrix 
    def load_and_explore(self, plot=True):
        # Read the dataset
        self.X = pd.read_csv(csv_path) # read the dataset; x is a Pandas dataframe
        features = list(self.X.columns) # list of features in the dataset
        subj = pd.unique(self.X['subject#']) # existing values of patient ID
        
        print(f"Original dataset shape: {self.X.shape}")
        print(f"Distinct patients: {len(subj)}")
        print(f"Features: {len(features)}")
        #print(self.X.describe().T)
        #print(self.X.info())

        if not plot:
            return

        # Measure and show covariance matrix
        Xnorm = (self.X - self.X.mean()) / self.X.std() # normalized/standardized/scaled data
        c = Xnorm.cov() 
        
        plt.figure()
        plt.matshow(np.abs(c.values), fignum=0)
        plt.xticks(np.arange(len(features)), features, rotation=90)
        plt.yticks(np.arange(len(features)), features, rotation=0)
        plt.colorbar()
        plt.title('Correlation coefficients (Original Data)')
        plt.tight_layout()
        plt.savefig('./corr_coeff.png')
        plt.draw()

        plt.figure()
        c.total_UPDRS.plot()
        plt.grid()
        plt.xticks(np.arange(len(features)), features, rotation=90)
        plt.title('Corr. coeff. total_UPDRS vs other features')
        plt.tight_layout()
        plt.savefig('./UPDRS_corr_coeff.png')
        plt.draw()

    # Split by patient into train/validation/test, then normalize using training data only.
    def prepare_data(self):
        groups = self.X['subject#'].to_numpy()
        first_split = GroupShuffleSplit(
            n_splits=1,
            train_size=0.4,
            random_state=self.seed,
        )
        train_idx, remaining_idx = next(first_split.split(self.X, groups=groups))

        remaining = self.X.iloc[remaining_idx]
        remaining_groups = remaining['subject#'].to_numpy()
        second_split = GroupShuffleSplit(
            n_splits=1,
            train_size=1 / 3,
            random_state=self.seed + 1,
        )
        val_rel_idx, test_rel_idx = next(
            second_split.split(remaining, groups=remaining_groups)
        )
        val_idx = remaining_idx[val_rel_idx]
        test_idx = remaining_idx[test_rel_idx]

        X_tr = self.X.iloc[train_idx].copy()
        X_va = self.X.iloc[val_idx].copy()
        X_te = self.X.iloc[test_idx].copy()

        self.train_subjects = set(X_tr['subject#'].unique())
        self.validation_subjects = set(X_va['subject#'].unique())
        self.test_subjects = set(X_te['subject#'].unique())
        if (
            self.train_subjects & self.validation_subjects
            or self.train_subjects & self.test_subjects
            or self.validation_subjects & self.test_subjects
        ):
            raise RuntimeError("Patient leakage detected across data partitions")

        print(
            "Patient-level split: "
            f"{len(self.train_subjects)} train / "
            f"{len(self.validation_subjects)} validation / "
            f"{len(self.test_subjects)} test subjects (overlap: 0)"
        )

        # Evaluate normalization parameters from training data only.
        self.mm = X_tr.mean()
        self.ss = X_tr.std()
        self.my = self.mm['total_UPDRS'] # mean of regressand/total UPDRS (for later use)
        self.sy = self.ss['total_UPDRS'] # st.dev of regressand/total UPDRS (for later use)

        # Exclude identifiers, collinear measures and motor_UPDRS, which is a
        # closely related clinical score and would make total_UPDRS prediction
        # unrealistically easy for a voice-biomarker experiment.
        drop_list = [
            'total_UPDRS',
            'motor_UPDRS',
            'subject#',
            'test_time',
            'Jitter:DDP',
            'Shimmer:DDA',
        ]
        self.regressors = [column for column in self.X.columns if column not in drop_list]
        self.Nf = len(self.regressors) # number of regressors
        print("After dropping, the new regressors are: ", len(self.regressors))
        print(self.regressors)

        self.X_tr_norm = ((X_tr[self.regressors] - self.mm[self.regressors]) / self.ss[self.regressors]).to_numpy()
        self.X_va_norm = ((X_va[self.regressors] - self.mm[self.regressors]) / self.ss[self.regressors]).to_numpy()
        self.X_te_norm = ((X_te[self.regressors] - self.mm[self.regressors]) / self.ss[self.regressors]).to_numpy()
        self.y_tr_norm = ((X_tr['total_UPDRS'] - self.my) / self.sy).to_numpy()
        self.y_va_norm = ((X_va['total_UPDRS'] - self.my) / self.sy).to_numpy()
        self.y_te_norm = ((X_te['total_UPDRS'] - self.my) / self.sy).to_numpy()

        print('The training set shape is {}, The validation set shape is {}, The test set shape is {}'.format(self.X_tr_norm.shape, self.X_va_norm.shape, self.X_te_norm.shape))
    
    #% Find optimal K
    def euclidean_distance(self, x0, X):
        return np.sum((X - x0) ** 2, axis=1)

    def local_prediction(
        self,
        x,
        K,
        X_train=None,
        y_train=None,
        exclude_index=None,
    ):
        """Fit a local ridge model with an unpenalized intercept."""
        if X_train is None:
            X_train = self.X_tr_norm
        if y_train is None:
            y_train = self.y_tr_norm

        distances = self.euclidean_distance(x, X_train)
        if exclude_index is not None:
            distances[exclude_index] = np.inf
        neighbor_idx = np.argsort(distances)[:K]

        local_X = np.column_stack(
            [np.ones(len(neighbor_idx)), X_train[neighbor_idx, :]]
        )
        local_y = y_train[neighbor_idx].reshape(-1, 1)
        ridge = np.eye(local_X.shape[1])
        ridge[0, 0] = 0.0
        weights = np.linalg.solve(
            local_X.T @ local_X + self.eps * ridge,
            local_X.T @ local_y,
        )
        x_with_intercept = np.concatenate(([1.0], x))
        return float((x_with_intercept @ weights).item())
    

    def fixed_k(self, K, eps, verbose=True):
        self.eps = eps
        n = self.X_va_norm.shape[0]
        y_hat_va_norm = np.zeros(n, dtype=float)
        for i in range(n):
            y_hat_va_norm[i] = self.local_prediction(
                self.X_va_norm[i, :],
                K,
            )
            
        mse_val = float(np.mean((self.y_va_norm - y_hat_va_norm) ** 2))
        if verbose:
            print(
                f"[K={K}, eps={eps:g}] Validation MSE "
                f"(normalized): {mse_val:.6f}"
            )
        return mse_val

    def optimized_k(self, k_min, k_max, step, eps_values=None):
        if eps_values is None:
            eps_values = (1e-6, 1e-4, 1e-2, 1e-1, 1.0)
        K_values = np.arange(int(k_min), int(k_max) + 1, int(step), dtype=int)
        mse_values = np.empty((len(eps_values), len(K_values)), dtype=float)
        for eps_index, eps in enumerate(eps_values):
            for k_index, k in enumerate(K_values):
                mse_values[eps_index, k_index] = self.fixed_k(
                    K=k,
                    eps=eps,
                    verbose=False,
                )

        best_eps_index, best_k_index = np.unravel_index(
            np.argmin(mse_values),
            mse_values.shape,
        )
        self.K_opt = int(K_values[best_k_index])
        self.eps = float(eps_values[best_eps_index])
        mse_min = float(mse_values[best_eps_index, best_k_index])
    
        # plot opzionale
        plt.figure()
        for eps_index, eps in enumerate(eps_values):
            plt.plot(
                K_values,
                mse_values[eps_index],
                '-o',
                label=f'eps={eps:g}',
            )
        plt.xlabel('K')
        plt.ylabel('Validation MSE (normalized)')
        plt.title('KNN-LLS validation search')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./K_optimization.png')
        plt.draw()
    
        print(
            f"[optimized_k] Best K = {self.K_opt}, eps = {self.eps:g} "
            f"| MSE_val (norm) = {mse_min:.6f}"
        )

    #% Test phase
    def test(self):
        self.X_tr_norm = np.vstack([self.X_tr_norm, self.X_va_norm]) # to rebuild the training set (true training set + validation set)
        self.y_tr_norm = np.concatenate([self.y_tr_norm, self.y_va_norm]) # to rebuild the regressand
        
        # --- KNN-LLS on Test Set ---
        n = self.X_te_norm.shape[0]
        y_hat_te_norm = np.zeros(n, dtype=float)
        for i in range(n):
            y_hat_te_norm[i] = self.local_prediction(
                self.X_te_norm[i, :],
                self.K_opt,
            )
        
        # De-normalization
        sy = self.sy
        my = self.my
        y_te = self.y_te_norm * sy + my

        y_hat_knn = y_hat_te_norm * sy + my
            
        e_knn = self.calculate_metrics(y_te, y_hat_knn, "KNN-LLS (Test)")
        self.plot_results(y_te, y_hat_knn, e_knn, "KNN-LLS (Test)")
        
        # --- Standard LLS on Test Set ---
        X_train = np.column_stack(
            [np.ones(self.X_tr_norm.shape[0]), self.X_tr_norm]
        )
        y_train = self.y_tr_norm.reshape(-1, 1)
        
        solver = mymin.SolveLLS(y=y_train, A=X_train)
        solver.run()
        w_lls = solver.what
        X_test = np.column_stack(
            [np.ones(self.X_te_norm.shape[0]), self.X_te_norm]
        )
        y_hat_lls_norm = (X_test @ w_lls).flatten()
        
        # De-normalization
        y_hat_lls = y_hat_lls_norm * sy + my
        
        e_lls = self.calculate_metrics(y_te, y_hat_lls, "Standard LLS (Test)")
        self.plot_results(y_te, y_hat_lls, e_lls, "Standard LLS (Test)")
        
        # --- KNN-LLS on Training Set ---
        n_tr = self.X_tr_norm.shape[0]
        y_hat_tr_knn_norm = np.zeros(n_tr, dtype=float)
        for i in range(n_tr):
            y_hat_tr_knn_norm[i] = self.local_prediction(
                self.X_tr_norm[i, :],
                self.K_opt,
                exclude_index=i,
            )
            
        y_hat_tr_knn = y_hat_tr_knn_norm * sy + my
        y_tr = self.y_tr_norm * sy + my
        self.calculate_metrics(y_tr, y_hat_tr_knn, "KNN-LLS (Training)")
        
    def calculate_metrics(self, y_true, y_pred, label):
        e = y_true - y_pred
        mean_e = np.mean(e)
        std_e = np.std(e)
        mse = np.mean(e**2)
        R2 = 1 - np.sum(e**2) / np.sum((y_true - np.mean(y_true))**2)
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        self.metrics[label] = {
            'mean_error': float(mean_e),
            'std_error': float(std_e),
            'mse': float(mse),
            'r2': float(R2),
            'correlation': float(corr),
        }
        
        print(f"\nMetrics for {label}:")
        print(f"Mean error: {mean_e:.4f}")
        print(f"Std error: {std_e:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R2: {R2:.4f}")
        print(f"Correlation coefficient: {corr:.4f}")
        return e

    def plot_results(self, y_true, y_pred, e, label):
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Regression Line ({label})')
        plt.grid()
        plt.savefig(f'regression_line_{label.replace(" ", "_")}.png')
        
        plt.figure()
        plt.hist(e, bins=20, edgecolor='black')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Histogram ({label})')
        plt.grid()
        plt.savefig(f'error_hist_{label.replace(" ", "_")}.png')
    
#% main
if __name__ == "__main__":
    lab = UPDRS()
    lab.load_and_explore()
    lab.prepare_data()
    lab.optimized_k(20, 150, 10)
    lab.test()
