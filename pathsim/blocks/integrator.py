#########################################################################################
##
##                             STANDARD INTEGRATOR BLOCK 
##                              (blocks/integrator.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.funcs import (
    dict_to_array, 
    array_to_dict
    )


# BLOCKS ================================================================================

class Integrator(Block):
    """
    Integrates the input signal using a numerical integration engine. 
    The block is inherently MIMO capable.

    INPUTS : 
        initial_value : (float or array) initial value of integrator
    """

    def __init__(self, initial_value=0.0):
        super().__init__()

        #save initial value
        self.initial_value = initial_value


    def __len__(self):
        return 0


    def set_solver(self, Solver, **solver_args):
        #change solver if already initialized
        if self.engine is not None:
            self.engine = self.engine.change(Solver, **solver_args)
            return #quit early
        #initialize the integration engine
        def _f(x, u, t): return u
        self.engine = Solver(self.initial_value, _f, None, **solver_args)
        

    def update(self, t):
        self.outputs = array_to_dict(self.engine.get())
        return 0.0


    def solve(self, t, dt):
        #advance solution of implicit update equation and update block outputs
        return self.engine.solve(dict_to_array(self.inputs), t, dt)


    def step(self, t, dt):
        #compute update step with integration engine and update block outputs
        return self.engine.step(dict_to_array(self.inputs), t, dt)