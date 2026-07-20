"""
Minimization Solvers for Linear Regression Problems.

This module provides an OOP-based framework for solving linear least-squares
minimization problems using different optimization strategies:
  - Linear Least Squares (LLS)
  - Gradient Descent (GD)
  - Steepest Descent (SD)

Each solver inherits from the base class `SolveMinProbl` and implements
a `run()` method with the specific optimization logic.
"""

import numpy as np
import matplotlib.pyplot as plt


class SolveMinProbl:
    """Base class for minimization problem solvers.

    Attributes:
        matr (np.ndarray): The regressor matrix A (Np x Nf).
        y (np.ndarray): The target vector y (Np x 1).
        Np (int): Number of data points (rows).
        Nf (int): Number of features (columns).
        what (np.ndarray): The estimated weight vector (Nf x 1).
        min (float): The minimum squared error achieved.
    """

    def __init__(self, y=np.ones((3,)), A=np.eye(3)):
        self.matr = A
        self.y = y
        self.Np = y.shape[0]
        self.Nf = A.shape[1]
        self.what = np.zeros((self.Nf, 1), dtype=float)
        self.min = 0
        self.err = None

    def plot_what(self, title="Solution"):
        """Plot the estimated weight vector."""
        what = self.what
        n = np.arange(self.Nf)
        plt.figure()
        plt.plot(n, what, '-o')
        plt.xlabel("n")
        plt.ylabel("w_hat(n)")
        plt.title(title)
        plt.grid()
        plt.draw()

    def print_result(self, title):
        """Print the optimum weight vector."""
        print(f"{title}:")
        print("The optimum weight vector is:")
        print(self.what.T)


class SolveLLS(SolveMinProbl):
    """Solver using closed-form Linear Least Squares (LLS).

    Computes the least-squares solution with an SVD-based routine.
    """

    def run(self):
        A = self.matr
        y = self.y
        what = np.linalg.lstsq(A, y, rcond=None)[0]
        self.what = what
        self.min = np.linalg.norm(A @ what - y) ** 2


class SolveGrad(SolveMinProbl):
    """Solver using fixed-step Gradient Descent.

    Iteratively updates w using a constant learning rate gamma:
        w_{k+1} = w_k - gamma * gradient
    """

    def run(self, gamma=1e-3, Nit=100):
        self.err = np.zeros((Nit, 2), dtype=float)
        self.gamma = gamma
        self.Nit = Nit
        A = self.matr
        y = self.y
        w = np.zeros((self.Nf, 1), dtype=float)
        sqerr = np.linalg.norm(A @ w - y) ** 2

        for i in range(Nit):
            grad = 2 * A.T @ (A @ w - y)
            w = w - gamma * grad
            sqerr = np.linalg.norm(A @ w - y) ** 2
            self.err[i, :] = [i, sqerr]

        self.what = w
        self.min = sqerr

    def plot_err(self, title="Square error", logy=0, logx=0):
        """Plot the squared error as a function of the iteration number."""
        if self.err is None:
            return
        err = self.err
        err = err[err[:, 0] != 0]

        plt.figure()
        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1])
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 1])
        plt.xlabel("Iteration")
        plt.ylabel("Squared Error")
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.draw()


class SolveSteepDesc(SolveMinProbl):
    """Solver using Steepest Descent with adaptive step size.

    At each iteration, the learning rate gamma is computed optimally:
        gamma = ||grad||^2 / (grad^T H grad)
    where H = 2 A^T A is the Hessian.
    """

    def run(self, Nit=1000, tol=1e-8):
        self.err = np.zeros((Nit, 2), dtype=float)
        self.Nit = Nit
        self.tol = tol
        A = self.matr
        y = self.y
        w = np.zeros((self.Nf, 1), dtype=float)
        H = 2 * A.T @ A
        sqerr = np.linalg.norm(A @ w - y) ** 2

        for i in range(Nit):
            grad = 2 * A.T @ (A @ w - y)
            grad_norm = np.linalg.norm(grad)

            if grad_norm <= tol:
                self.err = self.err[:i, :]
                break

            denom = (grad.T @ H @ grad).item()
            if denom < 1e-12:
                break

            gamma = (grad_norm ** 2) / denom
            w = w - gamma * grad
            sqerr = np.linalg.norm(A @ w - y) ** 2
            self.err[i, :] = [i, sqerr]

        self.what = w
        self.min = sqerr
