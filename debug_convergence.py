"""Debug script to examine solver convergence behavior"""

import sys
import numpy as np
sys.path.insert(0, 'tests')
from pathsim.solvers.euler import EUB
from pathsim.solvers._referenceproblems import PROBLEMS

# Test the first problem with EUB solver
problem = PROBLEMS[0]
print(f"\nTesting problem: {problem.name}")
print(f"Time span: {problem.t_span}")
print(f"Initial condition: {problem.x0}")

solver = EUB(problem.x0)

# Divisions of integration duration
divisions = np.logspace(2, 3, 30)
timesteps = (problem.t_span[1] - problem.t_span[0]) / divisions

errors = []

for i, dt in enumerate(timesteps):
    solver.reset()
    time, numerical_solution = solver.integrate(
        problem.func,
        problem.jac,
        time_start=problem.t_span[0],
        time_end=problem.t_span[1],
        dt=dt,
        adaptive=False
    )

    analytical_solution = problem.solution(time)
    err = np.mean(abs(numerical_solution - analytical_solution))
    errors.append(err)

    if i < 5 or i > len(timesteps) - 5:
        print(f"Step {i}: dt={dt:.6e}, error={err:.6e}")

errors = np.array(errors)
diffs = np.diff(errors)

print(f"\nTotal steps: {len(errors)}")
print(f"Monotonically decreasing: {np.all(diffs < 0)}")
print(f"Number of non-decreasing steps: {np.sum(diffs >= 0)}")
print(f"\nFirst 10 error differences:")
for i, d in enumerate(diffs[:10]):
    print(f"  Step {i}->{i+1}: {d:.6e} {'✓' if d < 0 else '✗'}")
print(f"\nLast 10 error differences:")
for i, d in enumerate(diffs[-10:], start=len(diffs)-10):
    print(f"  Step {i}->{i+1}: {d:.6e} {'✓' if d < 0 else '✗'}")

# Check if nearly monotonic (within floating point precision)
tolerance = 1e-12
nearly_monotonic = np.all(diffs < tolerance)
print(f"\nNearly monotonic (tol={tolerance}): {nearly_monotonic}")

# Show statistics
print(f"\nError statistics:")
print(f"  Min error: {np.min(errors):.6e}")
print(f"  Max error: {np.max(errors):.6e}")
print(f"  Error range: {np.max(errors) - np.min(errors):.6e}")
print(f"  Max positive diff: {np.max(diffs):.6e}")
