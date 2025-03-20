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
    """Solver that finds the time independent steady state solution (DC) 
    by forcing the derivatives to zero, i.e. f(x,u,t) = 0.
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