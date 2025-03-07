#########################################################################################
##
##                     GENERIC MIXED DYNAMICAL ALGEBRAIC BLOCK 
##                               (blocks/generic.py)
##
##                                Milan Rother 2025
##
#########################################################################################

# IMPORTS ===============================================================================

import warnings


import numpy as np

from ._block import Block

from ..utils.utils import (
    dict_to_array, 
    array_to_dict
    )


# BLOCKS ================================================================================

class Generic(Block):
    """
    This is a generic mixed dynamic-algebraic block where the 
    dynamical component is implemented as an ODE and the algebraic 
    component as a function of the states and the inputs 

        d/dt x = func_dyn(x, u, t)
             y = func_alg(x, u, t)
    
    with inhomogenity (input) 'u' and state vector 'x'. The functions 
    can be nonlinear and the ODE can be of arbitrary order. 
    The block utilizes the integration engine to solve the ODE 
    by integrating the dynamic component 'func_dyn'. 

    Essentially this is a generic nonlinear statespace block.

    INPUTS : 
        func_dyn      : (callable object) dynamic component (rhs function of ODE)
        func_alg      : (callable object) algebraic component 
        initial_value : (array of floats) initial state / initial condition
        jac_dyn       : (callable or None) jacobian of 'func_dyn' or 'None'
    """

    def __init__(self,
                 func_dyn=None,
                 func_alg=lambda x, u, t: 2*u,
                 initial_value=0.0,
                 jac_dyn=None):

        super().__init__()
          
        #dynamic and algebraic components
        self.func_dyn = func_dync
        self.func_alg = func_alg

        #initial condition
        self.initial_value = initial_value

        #jacobian of 'func_dyn'
        self.jac_dyn = jac_dyn

        warnings.warn(
            "Generic block will be deprecated in next release due to naming conflict with core Python", 
            DeprecationWarning
            )


    def __len__(self):
        #assume algebraic passthrough
        return 1


    def set_solver(self, Solver, **solver_args):

        #no dynamic component -> quit
        if self.func_dyn is None: return

        if self.engine is None:
            #initialize the integration engine with right hand side
            self.engine = Solver(self.initial_value, self.func_dyn, self.jac_dyn, **solver_args)
        else:
            #change solver if already initialized
            self.engine = Solver.cast(self.engine, **solver_args)


    def update(self, t):
        prev_outputs = self.outputs.copy()
        u = dict_to_array(self.inputs)
        if self.engine: x = self.engine.get()
        else: x = None
        self.outputs = array_to_dict(self.func_alg(x, u, t))
        return max_error_dicts(prev_outputs, self.outputs)


    def solve(self, t, dt):
        #advance solution of implicit update equation and update block outputs
        if not self.engine: return super().solve(t, dt)
        return self.engine.solve(dict_to_array(self.inputs), t, dt)


    def step(self, t, dt):
        #compute update step with integration engine and update block outputs
        if not self.engine: return super().step(t, dt)
        return self.engine.step(dict_to_array(self.inputs), t, dt)