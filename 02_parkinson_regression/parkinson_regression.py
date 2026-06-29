import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import minimization as mymin

matricola_seed = 306457  # Student ID (used as random seed)
csv_path = os.path.join(os.path.dirname(__file__), "data", "parkinsons_updrs.csv")

class ParkinsonUPDRS:
    def __init__(self, seed=matricola_seed):
        self.seed = seed
        self.results = {} #To store w_hat, errors, etc.
        
        # Output settings
        pd.set_option('display.precision', 3)
        plt.close('all')

    # Loads data and plots the covariance matrix 
    def load_and_explore(self):
        # Read the dataset
        self.X = pd.read_csv(csv_path) # read the dataset; x is a Pandas dataframe
        features = list(self.X.columns) # list of features in the dataset
        subj = pd.unique(self.X['subject#']) # existing values of patient ID
        
        print(f"Original dataset shape: {self.X.shape}")
        print(f"Distinct patients: {len(subj)}")
        print(f"Features: {len(features)}")
        print(self.X.describe().T)
        print(self.X.info())

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

    # Suffle, split, normalizes, drop features
    def prepare_data(self, shuffle=True, include_motor=True):

        Np, Nc = self.X.shape
        
        # 1. Shuffle (Flag Controlled) rationale: the original dataset is ordered by patients ID and contain different test for every patients,
        # shuffling it we avoid to learn to weel about a single patients (overfitting)
        if shuffle:
            print("Shuffling: ON")
            self.Xsh = self.X.sample(frac=1, replace=False, random_state=self.seed, axis=0, ignore_index=True)
        else:
            print("Shuffling: OFF")
            self.Xsh = self.X.copy()

        # 2. Split (50/50)
        Ntr = int(Np * 0.5) # number of training points
        Nte = Np - Ntr # number of test points
        
        # 3. evaluate mean and st.dev. for Training Data Only
        X_tr_raw = self.Xsh[0:Ntr]
        self.mm = X_tr_raw.mean()
        self.ss = X_tr_raw.std()
        self.my = self.mm['total_UPDRS']
        self.sy = self.ss['total_UPDRS']

        # 4. Normalize (Scaled training and test datasets)
        Xsh_norm = (self.Xsh - self.mm) / self.ss
        ysh_norm = Xsh_norm['total_UPDRS'] # Regressand

        # 5. Drop Features 
        # Always drop these (total_UPDRS + ID)
        drop_list = ['total_UPDRS', 'subject#']
        
        # Drop Jitter:DDP and Shimmer:DDA
        drop_list.extend(['Jitter:DDP', 'Shimmer:DDA'])
        
        # Handle Motor UPDRS flag
        if not include_motor:
            print("Motor UPDRS: EXCLUDED")
            drop_list.append('motor_UPDRS')
        else:
            print("Motor UPDRS: INCLUDED")

        Xsh_norm = Xsh_norm.drop(drop_list, axis=1) # Regressors only

        self.regressors = list(Xsh_norm.columns)
        print(f"New regressors ({len(self.regressors)}): {self.regressors}")

        # Convert to NumPy
        Xsh_norm = Xsh_norm.values # from datafram to Ndarray
        ysh_norm = ysh_norm.values

        # Final Split
        self.X_tr_norm = Xsh_norm[0:Ntr] # regressors for training phase
        self.X_te_norm = Xsh_norm[Ntr:] # regressors for test phase
        self.y_tr_norm = ysh_norm[0:Ntr] # regressand for training phase
        self.y_te_norm = ysh_norm[Ntr:] # regressand for test phase

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
    lab.prepare_data(shuffle=True, include_motor=False)
    lab.solve_and_evaluate("LLS")
    lab.solve_and_evaluate("SD")
    lab.compare_lls_sd()