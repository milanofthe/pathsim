########################################################################################
##
##                        TIME INDEPENDENT STEADY STATE SOLVER
##                              (solvers/steadystate.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ._solver import ImplicitSolver


# SOLVERS ==============================================================================

class SteadyState(ImplicitSolver):
    """Pseudo-solver that finds the time-independent steady-state solution (DC operating point).

    This works by modifying the fixed-point iteration target. Instead of solving
    :math:`x_{n+1} = G(x_{n+1})` for an implicit step, it aims to solve the algebraic equation
    :math:`f(x, u, t_{steady}) = 0` by finding the fixed point of :math:`x = x + f(x, u, t_{steady})`.
    It uses the same internal optimizer (e.g., NewtonAnderson) as other implicit solvers.

    Characteristics
    ---------------
    * Purpose: Find steady-state (:math:`dx/dt = 0`)
    * Implicit (uses optimizer)
    * Not a time-stepping method.
    """
        
    def solve(self, f, J, dt):
        """Solve for steady state by finding x where f(x,u,t) = 0
        using the fixed point equation x = x + f(x,u,t).

        Parameters
        ----------
        f : array_like
            evaluation of function
        J : array_like
            evaluation of jacobian of function
        dt : float 
            integration timestep

        Returns
        -------
        err : float
            residual error of the fixed point update equation
        """

        #fixed point equation g(x) = x + f(x,u,t)
        g = self.x + f
        
        if J is not None:

            #jacobian of g is I + df/dx
            jac_g = np.eye(len(self.x)) + J

            #optimizer step with block local jacobian
            self.x, err = self.opt.step(self.x, g, jac_g)
        
        else:

            #optimizer step without jacobian
            self.x, err = self.opt.step(self.x, g)
            
        return err