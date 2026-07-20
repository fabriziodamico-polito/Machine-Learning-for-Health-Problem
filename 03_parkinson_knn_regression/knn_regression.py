import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import minimization as mymin

matricola_seed = 306457  # Student ID (used as random seed)
csv_path = os.path.join(os.path.dirname(__file__), "data", "parkinsons_updrs.csv")

class UPDRS:
    def __init__(self, seed=matricola_seed):
        self.seed = seed
        #self.results = {} # To store w_hat, errors, etc.
        
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
        #print(self.X.describe().T)
        #print(self.X.info())

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

    #% shuffle, split, normalize, drop features (in this Lab we use the motor_UPDRS)
    def prepare_data(self, shuffle=True):

        Np, Nc = self.X.shape

        # 1. Shuffle:  the original dataset is ordered by patients ID and contain different test for every patients,
        # shuffling it we avoid to learn to weel about a single patients (overfitting)
        Xsh = self.X.sample(frac=1, replace=False, random_state=self.seed, axis=0, ignore_index=True)

        # 2. Split into training, validation and test set
        Ntr = int(Np * 0.4) # number of training points
        Nva = int(Np * 0.2) # number of validation points
        Nte = Np - Ntr - Nva # number of test points

        # 3. evaluate mean and st.dev. for Training Data Only
        X_tr = Xsh[0:Ntr] # dataframe that contains only the training data
        self.mm = X_tr.mean() # mean (series) of each feature
        self.ss = X_tr.std() # standard deviation (series) of each feature
        self.my = self.mm['total_UPDRS'] # mean of regressand/total UPDRS (for later use)
        self.sy = self.ss['total_UPDRS'] # st.dev of regressand/total UPDRS (for later use)

        # 4. Normalize (Scaled training and test datasets)
        Xsh_norm = (Xsh - self.mm) / self.ss # normalized data
        ysh_norm = Xsh_norm['total_UPDRS']
        
        # 5. Drop Features 
        Xsh_norm = Xsh_norm.drop(['total_UPDRS', 'subject#'], axis=1) # regressors only
        Xsh_norm = Xsh_norm.drop(['Jitter:DDP', 'Shimmer:DDA'], axis=1) # drop Jitter and Shimmer to avoid collinearity
        if 'test_time' in Xsh_norm.columns:
            Xsh_norm = Xsh_norm.drop(['test_time'], axis=1)
        
        self.regressors = list(Xsh_norm.columns)
        self.Nf = len(self.regressors) # number of regressors
        print("After dropping, the new regressors are: ", len(self.regressors))
        print(self.regressors)
        
        # DataFrame -> Numpy
        Xsh_norm = Xsh_norm.values # from dataframe to Ndarray
        ysh_norm = ysh_norm.values # from dataframe to Ndarray

        # split numpy
        self.X_tr_norm = Xsh_norm[0:Ntr] # regressors for training phase
        self.X_va_norm = Xsh_norm[Ntr:Ntr + Nva] # regressor for validation phase
        self.X_te_norm = Xsh_norm[Ntr + Nva:] # regressors for test phase
        
        self.y_tr_norm = ysh_norm[0:Ntr] # regressand for training phase
        self.y_va_norm = ysh_norm[Ntr: Ntr + Nva] # regressand for validation phase
        self.y_te_norm = ysh_norm[Ntr + Nva:] # regressand for test phase

        print('The training set shape is {}, The validation set shape is {}, The test set shape is {}'.format(self.X_tr_norm.shape, self.X_va_norm.shape, self.X_te_norm.shape))
    
    #% Find optimal K
    def euclidean_distance(self, x0, X):
        return np.sum((X - x0) ** 2, axis=1)
    

    def fixed_k(self, K, eps):
        self.eps = 1e-8
        n = self.X_va_norm.shape[0]
        y_hat_va_norm = np.zeros(n, dtype=float)
        for i in range(n):
            x = self.X_va_norm[i, :]                       
            d = self.euclidean_distance(x, self.X_tr_norm)
            idx = np.argsort(d)[:K]
            A = self.X_tr_norm[idx, :]                     # (K, F)
            y = self.y_tr_norm[idx].reshape(-1, 1)        # (K, 1)
            F = A.shape[1]
            I = np.eye(F)
            w_hat = np.linalg.inv(A.T @ A + eps * I) @ (A.T @ y)   # (F,1) eps is a term add to be sure that the matrix will be always inverted (ridge regression)
            
            y_hat_va_norm[i] = (x @ w_hat).item()
            
        mse_val = float(np.mean((self.y_va_norm - y_hat_va_norm) ** 2))
        print(f"[K={K}] Validation MSE (normalized): {mse_val:.6f}")
        return mse_val

    def optimized_k(self, k_min, k_max, step):
        K_values = np.arange(int(k_min), int(k_max) + 1, int(step), dtype=int)
        mse_values = np.empty(K_values.shape[0], dtype=float)
        for i, k in enumerate(K_values):
            mse_values[i] = self.fixed_k(K=k, eps=1e-8)
    
        best_idx = int(np.argmin(mse_values))
        self.K_opt = int(K_values[best_idx])
        mse_min = float(mse_values[best_idx])
    
        # plot opzionale
        plt.figure()
        plt.plot(K_values, mse_values, '-o')
        plt.xlabel('K')
        plt.ylabel('Validation MSE (normalized)')
        plt.title('MSE vs K (validation)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('./K_optimization.png')
        plt.draw()
    
        print(f"[optimized_k] Best K = {self.K_opt}  |  MSE_val (norm) = {mse_min:.6f}")

    #% Test phase
    def test(self):
        self.X_tr_norm = np.vstack([self.X_tr_norm, self.X_va_norm]) # to rebuild the training set (true training set + validation set)
        self.y_tr_norm = np.concatenate([self.y_tr_norm, self.y_va_norm]) # to rebuild the regressand
        
        # --- KNN-LLS on Test Set ---
        n = self.X_te_norm.shape[0]
        y_hat_te_norm = np.zeros(n, dtype=float)
        for i in range(n):
            x = self.X_te_norm[i, :]                       
            d = self.euclidean_distance(x, self.X_tr_norm)
            idx = np.argsort(d)[:self.K_opt]
            A = self.X_tr_norm[idx, :]                     # (K, F)
            y = self.y_tr_norm[idx].reshape(-1, 1)        # (K, 1)
            F = A.shape[1]
            I = np.eye(F)
            w_hat = np.linalg.inv(A.T @ A + self.eps * I) @ (A.T @ y)   # (F,1)
            y_hat_te_norm[i] = (x @ w_hat).item()
        
        # De-normalization
        sy = self.sy
        my = self.my
        y_te = self.y_te_norm * sy + my

        y_hat_knn = y_hat_te_norm * sy + my
            
        e_knn = self.calculate_metrics(y_te, y_hat_knn, "KNN-LLS (Test)")
        self.plot_results(y_te, y_hat_knn, e_knn, "KNN-LLS (Test)")
        
        # --- Standard LLS on Test Set ---
        X_train = self.X_tr_norm
        y_train = self.y_tr_norm.reshape(-1, 1)
        
        solver = mymin.SolveLLS(y=y_train, A=X_train)
        solver.run()
        w_lls = solver.what
        y_hat_lls_norm = (self.X_te_norm @ w_lls).flatten()
        
        # De-normalization
        y_hat_lls = y_hat_lls_norm * sy + my
        
        e_lls = self.calculate_metrics(y_te, y_hat_lls, "Standard LLS (Test)")
        self.plot_results(y_te, y_hat_lls, e_lls, "Standard LLS (Test)")
        
        # --- KNN-LLS on Training Set ---
        n_tr = self.X_tr_norm.shape[0]
        y_hat_tr_knn_norm = np.zeros(n_tr, dtype=float)
        for i in range(n_tr):
            x = self.X_tr_norm[i, :]                       
            d = self.euclidean_distance(x, self.X_tr_norm)
            idx = np.argsort(d)[:self.K_opt]
            A = self.X_tr_norm[idx, :]                     # (K, F)
            y = self.y_tr_norm[idx].reshape(-1, 1)        # (K, 1)
            F = A.shape[1]
            I = np.eye(F)
            w_hat = np.linalg.inv(A.T @ A + self.eps * I) @ (A.T @ y)   # (F,1)
            y_hat_tr_knn_norm[i] = (x @ w_hat).item()
            
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
    lab.prepare_data(shuffle=True)
    lab.fixed_k(20, 1e-8)
    lab.optimized_k(17, 100, 3)
    lab.test()