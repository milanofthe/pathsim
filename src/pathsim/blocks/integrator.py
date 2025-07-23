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

from ..optim.operator import DynamicOperator


# BLOCKS ================================================================================

class Integrator(Block):
    """Integrates the input signal using a numerical integration engine like this:

    .. math::

        y(t) = \\int_0^t u(\\tau) \\ d \\tau
    
    or in differential form like this:

    .. math::
        \\begin{eqnarray}
            \\dot{x}(t) &= u(t) \\\\
                   y(t) &= x(t) 
        \\end{eqnarray}

    The Integrator block is inherently MIMO capable, so `u` and `y` can be vectors.
    
    Example
    -------
    This is how to initialize the integrator: 

    .. code-block:: python
    
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

        if self.engine is None:
            #initialize the integration engine
            self.engine = Solver(self.initial_value, **solver_args)

        else:
            #change solver if already initialized
            self.engine = Solver.cast(self.engine, **solver_args)


    def update(self, t):
        """update system equation fixed point loop

        Note
        ----
        integrator does not have passthrough, therefore this 
        method is performance optimized for this case

        Parameters
        ----------
        t : float
            evaluation time
        """
        self.outputs.update_from_array(self.engine.get())


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
        f = self.inputs.to_array()
        return self.engine.solve(f, None, dt)


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
        f = self.inputs.to_array()
        return self.engine.step(f, dt)