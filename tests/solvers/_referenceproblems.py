
########################################################################################
##
##                      REFERENCE PROBLEMS FOR SOLVER TESTING
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np


# TEST PROBLEMS ========================================================================

class Problem:
    def __init__(self, name, func, jac, x0, solution):
        self.name = name
        self.func = func
        self.jac = jac
        self.x0 = x0
        self.solution = solution


#create some reference problems for testing
problems = [
    Problem(name="linear_feedback", 
            func=lambda x, u, t: -x, 
            jac=lambda x, u, t: -1, 
            x0=1.0, 
            solution=lambda t: np.exp(-t)
            ),
    Problem(name="logistic", 
            func=lambda x, u, t: x*(1-x), 
            jac=lambda x, u, t: 1-2*x, 
            x0=0.5, 
            solution=lambda t: 1/(1 + np.exp(-t))
            )
]