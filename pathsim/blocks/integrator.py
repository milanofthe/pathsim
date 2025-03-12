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

from ..utils.utils import (
    dict_to_array, 
    array_to_dict
    )


# BLOCKS ================================================================================

class Integrator(Block):
    """Integrates the input signal using a numerical integration engine like this:

    .. math::

        y(t) = \\int_0^t u(\\tau) \\ d \\tau
    
    The Integrator block is inherently MIMO capable, so `u` and `y` can be vectors.
    
    Example
    -------
    
    This is how to initialize the integrator: 

    .. code-block:: python
    
        from pathsim.blocks import Integrator
    
        #initial value 0.0
        i1 = Integrator()

        #initial value 2.5
        i2 = Integrator(2.5)
    

    Parameters
    ----------
    initial_value : float, array
        initial value of integrator
    """

    def __init__(self, initial_value=0.0):
        super().__init__()

        #save initial value
        self.initial_value = initial_value


    def __len__(self):
        return 0


    def set_solver(self, Solver, **solver_args):
        """set the internal numerical integrator

        Parameters
        ----------
        Solver : Solver
            numerical integration solver class
        solver_args : dict
            parameters for solver initialization
        """

        #change solver if already initialized
        if self.engine is not None:
            self.engine = Solver.cast(self.engine, **solver_args)
            return #quit early
            
        #initialize the integration engine
        def _f(x, u, t): return u
        self.engine = Solver(self.initial_value, _f, None, **solver_args)
        

    def update(self, t):
        """update system equation fixed point loop

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            deviation to previous iteration for convergence control
        """
        self.outputs = array_to_dict(self.engine.get())
        return 0.0


    def solve(self, t, dt):
        """advance solution of implicit update equation of the solver

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep

        Returns
        ------- 
        error : float
            solver residual norm
        """
        return self.engine.solve(dict_to_array(self.inputs), t, dt)


    def step(self, t, dt):
        """compute timestep update with integration engine
        
        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep
    
        Returns
        ------- 
        success : bool
            step was successful
        error : float
            local truncation error from adaptive integrators
        scale : float
            timestep rescale from adaptive integrators
        """
        return self.engine.step(dict_to_array(self.inputs), t, dt)