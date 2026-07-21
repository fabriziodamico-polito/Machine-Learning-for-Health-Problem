"""Compare three solvers on a reproducible noiseless least-squares problem."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import minimization as mymin


RANDOM_SEED = 42


def run_demo(seed=RANDOM_SEED, plot=True):
    """Run LLS, fixed-step GD and steepest descent on the same system."""
    rng = np.random.default_rng(seed)
    n_samples = 100
    n_features = 4
    matrix = rng.standard_normal((n_samples, n_features))
    true_weights = rng.standard_normal((n_features, 1))
    target = matrix @ true_weights

    lls = mymin.SolveLLS(target, matrix)
    lls.run()

    # The Hessian is 2 A^T A. Choosing gamma below its reciprocal spectral
    # radius gives a stable fixed step for this quadratic objective.
    hessian_lipschitz = 2 * np.linalg.norm(matrix, ord=2) ** 2
    gamma = 0.9 / hessian_lipschitz
    gd = mymin.SolveGrad(target, matrix)
    gd.run(gamma=gamma, Nit=1000)

    steepest = mymin.SolveSteepDesc(target, matrix)
    steepest.run(Nit=1000, tol=1e-10)

    solvers = {
        'LLS': lls,
        'Gradient Descent': gd,
        'Steepest Descent': steepest,
    }
    errors = {
        name: float(np.linalg.norm(solver.what - true_weights))
        for name, solver in solvers.items()
    }

    print(f"Stable GD step size: {gamma:.6g}")
    for name, solver in solvers.items():
        solver.print_result(name)
        print(f"Coefficient error: {errors[name]:.3e}")

    if plot:
        for name, solver in solvers.items():
            solver.plot_what(name)
        gd.plot_err('Gradient Descent: squared error', logy=1)
        plt.show()

    return {
        'matrix': matrix,
        'target': target,
        'true_weights': true_weights,
        'solvers': solvers,
        'coefficient_errors': errors,
        'gamma': gamma,
    }


if __name__ == '__main__':
    run_demo()
