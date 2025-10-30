"""Minimal test to check EUB convergence for exponential decay"""

import numpy as np
from pathsim.solvers.euler import EUB

# Monkey patch to see what's happening
original_integrate = EUB.integrate

def debug_integrate(self, *args, **kwargs):
    result = original_integrate(self, *args, **kwargs)
    times, states = result
    print(f"  Result shapes: times={times.shape}, states={states.shape}, states dtype={states.dtype}")
    print(f"  First 3 states: {states[:3]}")
    return result

EUB.integrate = debug_integrate

# Simple exponential decay problem
def func(x, t):
    return -x

def jac(x, t):
    return -1.0

x0 = 1.0
t_span = (0, 5)

# Test with just a few timesteps
solver = EUB(x0)

divisions = [100, 110, 120, 130, 140, 150]
errors = []

for div in divisions:
    dt = (t_span[1] - t_span[0]) / div

    solver.reset()
    time, numerical_solution = solver.integrate(
        func,
        jac,
        time_start=t_span[0],
        time_end=t_span[1],
        dt=dt,
        adaptive=False
    )

    analytical_solution = np.exp(-time)
    err = np.mean(np.abs(numerical_solution - analytical_solution))
    errors.append(err)

    print(f"div={div:3d}, dt={dt:.6f}, error={err:.10e}")

print(f"\nErrors: {errors}")
print(f"Diffs: {np.diff(errors)}")
print(f"All negative: {np.all(np.diff(errors) < 0)}")
