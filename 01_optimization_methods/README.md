# Optimization Methods for Linear Regression

## Objective

Implement and compare three fundamental optimization algorithms for solving the **linear least-squares** minimization problem:

$$\hat{w} = \arg\min_w \|Aw - y\|^2$$

This lab introduces the Object-Oriented Programming (OOP) framework used throughout the course for implementing solvers.

## Techniques

| Method | Type | Description |
|--------|------|-------------|
| **Linear Least Squares (LLS)** | Direct | Stable SVD-based least-squares solution via `numpy.linalg.lstsq` |
| **Gradient Descent (GD)** | Iterative | Fixed learning rate $\gamma$, chosen from the Hessian spectral norm |
| **Steepest Descent (SD)** | Iterative | Adaptive step size via $\gamma_k = \|\nabla J\|^2 / (\nabla J^T H \nabla J)$ |

## Implementation Details

- A seeded random matrix $A \in \mathbb{R}^{100 \times 4}$ and a true weight vector $w$ are generated
- The target vector is computed as $y = Aw$ (noiseless scenario)
- Each solver estimates $\hat{w}$ and compares it against the ground truth
- The shared `SolveMinProbl` base class (in `utils/minimization.py`) provides common functionality via inheritance

## Key Results

- **LLS coefficient error:** `1.07e-15`
- **Gradient Descent coefficient error:** below machine precision after 1,000 iterations
- **Steepest Descent coefficient error:** `3.34e-10`

The test suite requires every solver to recover the known noiseless coefficient vector with error below `1e-6`. This makes convergence an automated numerical result rather than a visual claim.

## How to Run

```bash
cd 01_optimization_methods
python optimization_demo.py
```

## Files

| File | Description |
|------|-------------|
| `optimization_demo.py` | Main script demonstrating LLS, GD, and SD |
