########################################################################################
##
##                      REFERENCE PROBLEMS FOR SOLVER TESTING
##
##                                Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np


# TEST PROBLEMS ========================================================================

class Problem:
   def __init__(self, name, func, jac, x0, solution, t_span=(0, 10)):
       self.name = name
       self.func = func
       self.jac = jac
       self.x0 = x0
       self.solution = solution
       self.t_span = t_span


# create some reference problems for testing
PROBLEMS = [
    Problem(
        name="exponential_decay",
        func=lambda x, t: -x,
        jac=lambda x, t: -1,
        x0=1.0,
        solution=lambda t: np.exp(-t),
        t_span=(0, 5)
        ),
    Problem(
        name="logistic",
        func=lambda x, t: x*(1-x),
        jac=lambda x, t: 1-2*x,
        x0=0.5,
        solution=lambda t: 1/(1 + np.exp(-t)),
        t_span=(0, 10)
        ),
    Problem(
        name="quadratic",
        func=lambda x, t: x**2,
        jac=lambda x, t: 2*x,
        x0=1.0,
        solution=lambda t: 1/(1-t),
        t_span=(0, 0.6)  # Solution blows up at t=1
        ),
    # Problem(
    #     name="rational_growth",
    #     func=lambda x, t: x/(1+t),
    #     jac=lambda x, t: 1/(1+t),
    #     x0=1.0,
    #     solution=lambda t: (1+t),
    #     t_span=(0, 10)
    #     ),
    Problem(
        name="sin_decay",
        func=lambda x, t: -x*np.sin(t),
        jac=lambda x, t: -np.sin(t),
        x0=1.0,
        solution=lambda t: np.exp(np.cos(t) - 1),
        t_span=(0, 10)
        ),
    # Problem(
    #     name="bounded_growth",
    #     func=lambda x, t: np.sin(t)*x,
    #     jac=lambda x, t: np.sin(t),
    #     x0=2.0,
    #     solution=lambda t: 2.0*np.exp(1-np.cos(t)),
    #     t_span=(0, 10)
    #     ),
    Problem(
        name="polynomial",
        func=lambda x, t: t**2 - x,
        jac=lambda x, t: -1,
        x0=0.0,
        solution=lambda t: t**2 - 2*t + 2 - 2*np.exp(-t),
        t_span=(0, 5)
        )
    ]