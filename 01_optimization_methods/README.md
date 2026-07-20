# Optimization Methods for Linear Regression

## Objective

Implement and compare three fundamental optimization algorithms for solving the **linear least-squares** minimization problem:

$$\hat{w} = \arg\min_w \|Aw - y\|^2$$

This lab introduces the Object-Oriented Programming (OOP) framework used throughout the course for implementing solvers.

## Techniques

| Method | Type | Description |
|--------|------|-------------|
| **Linear Least Squares (LLS)** | Closed-form | Direct solution via the normal equation: $\hat{w} = (A^T A)^{-1} A^T y$ |
| **Gradient Descent (GD)** | Iterative | Fixed learning rate $\gamma$, iterates $w_{k+1} = w_k - \gamma \nabla J(w_k)$ |
| **Steepest Descent (SD)** | Iterative | Adaptive step size via $\gamma_k = \|\nabla J\|^2 / (\nabla J^T H \nabla J)$ |

## Implementation Details

- A random matrix $A \in \mathbb{R}^{100 \times 4}$ and a true weight vector $w$ are generated
- The target vector is computed as $y = Aw$ (noiseless scenario)
- Each solver estimates $\hat{w}$ and compares it against the ground truth
- The shared `SolveMinProbl` base class (in `utils/minimization.py`) provides common functionality via inheritance

## Key Results

- **LLS** recovers the exact solution in one step (as expected for noiseless data)
- **Gradient Descent** converges to the solution, with convergence rate depending on $\gamma$
- **Steepest Descent** converges faster than GD thanks to the adaptive step size, and is less sensitive to hyperparameter tuning

## How to Run

```bash
cd 01_optimization_methods
python optimization_demo.py
```

## Files

| File | Description |
|------|-------------|
| `optimization_demo.py` | Main script demonstrating LLS, GD, and SD |
