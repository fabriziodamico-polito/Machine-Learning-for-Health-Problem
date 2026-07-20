import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import minimization as mymin

RANDOM_SEED = 42
csv_path = os.path.join(os.path.dirname(__file__), "data", "parkinsons_updrs.csv")

class ParkinsonUPDRS:
    def __init__(self, seed=RANDOM_SEED):
        self.seed = seed
        self.results = {} #To store w_hat, errors, etc.
        
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
        print(self.X.describe().T)
        print(self.X.info())

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

    # Split by patient, normalize from training statistics, and drop features.
    def prepare_data(self, include_motor=False, test_size=0.5):
        groups = self.X['subject#'].to_numpy()
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=self.seed,
        )
        train_idx, test_idx = next(splitter.split(self.X, groups=groups))
        X_tr_raw = self.X.iloc[train_idx].copy()
        X_te_raw = self.X.iloc[test_idx].copy()

        self.train_subjects = set(X_tr_raw['subject#'].unique())
        self.test_subjects = set(X_te_raw['subject#'].unique())
        overlap = self.train_subjects & self.test_subjects
        if overlap:
            raise RuntimeError(f"Patient leakage detected: {sorted(overlap)}")

        print(
            f"Patient-level split: {len(self.train_subjects)} train subjects / "
            f"{len(self.test_subjects)} test subjects (overlap: 0)"
        )

        # Evaluate normalization parameters from training data only.
        self.mm = X_tr_raw.mean()
        self.ss = X_tr_raw.std()
        self.my = self.mm['total_UPDRS']
        self.sy = self.ss['total_UPDRS']

        drop_list = ['total_UPDRS', 'subject#', 'Jitter:DDP', 'Shimmer:DDA']
        if not include_motor:
            drop_list.append('motor_UPDRS')
        print(f"Motor UPDRS: {'INCLUDED' if include_motor else 'EXCLUDED'}")

        self.regressors = [column for column in self.X.columns if column not in drop_list]
        print(f"New regressors ({len(self.regressors)}): {self.regressors}")

        self.X_tr_norm = ((X_tr_raw[self.regressors] - self.mm[self.regressors]) / self.ss[self.regressors]).to_numpy()
        self.X_te_norm = ((X_te_raw[self.regressors] - self.mm[self.regressors]) / self.ss[self.regressors]).to_numpy()
        self.y_tr_norm = ((X_tr_raw['total_UPDRS'] - self.my) / self.sy).to_numpy()
        self.y_te_norm = ((X_te_raw['total_UPDRS'] - self.my) / self.sy).to_numpy()

    # Runs the solver (LLS or SD), de-normalizes and computes statistics
    def solve_and_evaluate(self, method):
        # 1. Select Solver
        # Ensure y is a column vector (N, 1) to prevent broadcasting errors
        y_input = self.y_tr_norm.reshape(-1, 1)
        if method == "LLS":
            solver = mymin.SolveLLS(y_input, self.X_tr_norm)
        elif method == "SD":
            solver = mymin.SolveSteepDesc(y_input, self.X_tr_norm)
        
        print(f"\n--- Running {method} ---")
        solver.run() # Run optimization
        
        # 2. Predictions (Normalized)
        y_hat_tr_norm = self.X_tr_norm @ solver.what
        y_hat_te_norm = self.X_te_norm @ solver.what
        
        # 3. De-normalize
        y_tr = self.y_tr_norm * self.sy + self.my
        y_te = self.y_te_norm * self.sy + self.my
        y_hat_tr = y_hat_tr_norm.flatten() * self.sy + self.my
        y_hat_te = y_hat_te_norm.flatten() * self.sy + self.my
        
        # 4. Store results for comparison between LLS and SD
        self.results[method] = {
            'w': solver.what,
            'y_te': y_te,
            'y_hat_te': y_hat_te,
            'y_tr': y_tr,
            'y_hat_tr': y_hat_tr
        }

        # 5. Plots (Specific to the method)
        # Plot Weights
        
        solver.plot_what(title=f'{method}-Optimized weights')
        
        # Plot Histograms
        E_tr = (y_tr - y_hat_tr)
        E_te = (y_te - y_hat_te)
        
        M = np.max([np.max(E_tr), np.max(E_te)])
        m = np.min([np.min(E_tr), np.min(E_te)])
        common_bins = np.arange(m, M, (M - m) / 50)
        
        plt.figure(figsize=(6, 4))
        plt.hist([E_tr, E_te], bins=common_bins, density=True, histtype='bar', label=['training', 'test'])
        plt.xlabel(r'$e=y-\^y$')
        plt.ylabel(r'$P(e$ in bin$)$')
        plt.legend()
        plt.grid()
        plt.title(f'{method}-Error histograms')
        plt.tight_layout()
        plt.savefig(f'./{method}-hist.png')
        plt.draw()

        # Plot Regression Line
        plt.figure(figsize=(4, 4))
        plt.plot(y_te, y_hat_te, '.', label='all')
        plt.legend()
        v = plt.axis()
        plt.plot([v[0], v[1]], [v[0], v[1]], 'r', linewidth=2)
        plt.xlabel(r'$y$')
        plt.ylabel(r'$\^y$')
        plt.axis('square')
        plt.grid()
        plt.title(f'{method}-test')
        plt.tight_layout()
        plt.savefig(f'./{method}-yhat_vs_y.png')
        plt.draw()

        # 6. Statistics Table (Professor's Code)
        self.print_statistics(E_tr, y_tr, y_hat_tr, E_te, y_te, y_hat_te, method)

    def print_statistics(self, E_tr, y_tr, y_hat_tr, E_te, y_te, y_hat_te, method):
        # Training stats
        E_tr_MSE = np.mean(E_tr**2)
        R2_tr = 1 - E_tr_MSE / (np.var(y_tr))
        c_tr = np.mean((y_tr - y_tr.mean()) * (y_hat_tr - y_hat_tr.mean())) / (y_tr.std() * y_hat_tr.std())
        
        # Test stats
        E_te_MSE = np.mean(E_te**2)
        R2_te = 1 - E_te_MSE / (np.var(y_te))
        c_te = np.mean((y_te - y_te.mean()) * (y_hat_te - y_hat_te.mean())) / (y_te.std() * y_hat_te.std())
        
        cols = ['min', 'max', 'mean', 'std', 'MSE', 'R^2', 'corr_coeff']
        rows = ['Training', 'test']
        p = np.array([
            [E_tr.min(), E_tr.max(), E_tr.mean(), E_tr.std(), E_tr_MSE, R2_tr, c_tr],
            [E_te.min(), E_te.max(), E_te.mean(), E_te.std(), E_te_MSE, R2_te, c_te],
        ])

        print(f"\nResults for {method}:")
        results = pd.DataFrame(p, columns=cols, index=rows)
        print(results)

    # Compares LLS and SD weights visually
    def compare_lls_sd(self):
        if "LLS" not in self.results or "SD" not in self.results:
            return

        w_lls = self.results["LLS"]['w']
        w_sd = self.results["SD"]['w']
        
        nn = np.arange(len(self.regressors))
        plt.figure(figsize=(8, 5))
        plt.plot(nn, w_lls, '-o', label='LLS')
        plt.plot(nn, w_sd, '--x', label='SD')
        plt.xticks(nn, self.regressors, rotation=90)
        plt.ylabel(r'$\hat{w}(n)$')
        plt.title('Comparison: LLS vs SD Weights')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    lab = ParkinsonUPDRS()
    lab.load_and_explore()
    lab.prepare_data(include_motor=False)
    lab.solve_and_evaluate("LLS")
    lab.solve_and_evaluate("SD")
    lab.compare_lls_sd()
