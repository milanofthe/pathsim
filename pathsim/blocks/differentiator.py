#########################################################################################
##
##                        DIFFERENTIATOR BLOCK (blocks/differentiator.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.funcs import rel_error


# BLOCKS ================================================================================


class Differentiator(Block):
    """
    Differentiates the input signal (SISO) using a first order transfer function 
    with a pole at the origin which implements a high pass filter. 

        H_diff(s) = s / (1 + s/f_max)

    The approximation holds for signals up to a frequency of approximately f_max.

    NOTE :
        Depending on 'f_max', the resulting system might become stiff or ill conditioned!
        As a practical choice set f_max to 3x the highest expected signal frequency.

    INPUTS :
        f_max : (float) highest expected signal frequency
    """

    def __init__(self, f_max=1e2):
        super().__init__()

        #maximum frequency for differentiator approximation
        self.f_max = f_max

    def __len__(self):
        return 1

    def initialize_solver(self, Solver, tolerance_lte=1e-6):
        #initialize the numerical integration engine with kernel
        def _f(x, u, t): return - self.f_max * (x - u) 
        def _jac(x, u, t): return - self.f_max
        self.engine = Solver(0.0, _f, _jac, tolerance_lte)

    def update(self, t):
        #compute implicit balancing update
        prev_output = self.outputs[0]
        self.outputs[0] = -self.f_max * (self.engine.get() - self.inputs[0])
        return rel_error(prev_output, self.outputs[0])

    def solve(self, t, dt):
        #advance solution of implicit update equation
        return self.engine.solve(self.inputs[0], t, dt)

    def step(self, t, dt):
        #compute update step with integration engine
        return self.engine.step(self.inputs[0], t, dt)