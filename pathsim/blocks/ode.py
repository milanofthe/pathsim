#########################################################################################
##
##                       ORDINARY DIFFERENTIAL EQUATION BLOCK 
##                                 (blocks/ode.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.funcs import (
    dict_to_array, 
    array_to_dict,
    auto_jacobian
    )


# BLOCKS ================================================================================

class ODE(Block):
    """
    This block implements an ordinary differential equation (ODE) 
    defined by its right hand side

        d/dt x = func(x, u, t)
    
    with inhomogenity (input) u and state vector x. The function 
    can be nonlinear and the ODE can be of arbitrary order. 
    The block utilizes the integration engine to solve the ODE 
    by integrating the 'func' right hand side function.

    INPUTS : 
        func          : (callable object) right hand side function of ODE
        initial_value : (array of floats) initial state / initial condition
        jac           : (callable or None) jacobian of 'func' or 'None'
    """

    def __init__(self,
                 func=lambda x, u, t: -x,
                 initial_value=0.0,
                 jac=None):

        super().__init__()
          
        #right hand side function of ODE
        self.func = func

        #initial condition
        self.initial_value = initial_value

        #jacobian of 'func'
        self.jac = jac
        

    def __len__(self):
        return 0


    def set_solver(self, Solver, **solver_args):
        if self.engine is None:
            #initialize the integration engine with right hand side
            _jac = auto_jacobian(self.func) if self.jac is None else self.jac
            self.engine = Solver(self.initial_value, self.func, _jac, **solver_args)
        else:
            #change solver if already initialized
            self.engine = self.engine.change(Solver, **solver_args)
        

    def update(self, t):
        self.outputs = array_to_dict(self.engine.get())
        return 0


    def solve(self, t, dt):
        #advance solution of implicit update equation and update block outputs
        return self.engine.solve(dict_to_array(self.inputs), t, dt)


    def step(self, t, dt):
        #compute update step with integration engine and update block outputs
        return self.engine.step(dict_to_array(self.inputs), t, dt)