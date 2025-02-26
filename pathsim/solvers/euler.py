########################################################################################
##
##                      EXPLICIT and IMPLICIT EULER INTEGRATORS
##                                (solvers/euler.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._solver import ExplicitSolver, ImplicitSolver


# SOLVERS ==============================================================================

class EUF(ExplicitSolver):
    """Class that performs explicit (forward) euler integration
    it holds the state and implements the timestep update.

    Use this only if the function to integrate is super smooth 
    or multistep/multistage methods cant be used. 
    """

    def step(self, u, t, dt):
        """performs the explicit forward timestep for (t+dt) 
        based on the state and input at (t)

        Parameters
        ----------
        u : numeric, array[numeric]
            function 'func' input value
        t : float
            evaluation time of function 'func'
        dt : float 
            integration timestep

        Returns
        -------
        success : bool
            timestep was successful
        err : float
            truncation error estimate
        scale : float
            timestep rescale from error controller
        """

        #update state with euler step
        self.x = self.x_0 + dt * self.func(self.x, u, t)

        #no error estimate available
        return True, 0.0, 1.0


class EUB(ImplicitSolver):
    """Class that performs implicit (backward) euler integration
    it holds the state and implements the solution of the 
    implicit update equation at each timestep.

    Its an absolute classic and ok for moderately stiff problems 
    that dont require super high accuracy.
    """

    def solve(self, u, t, dt):
        """Solves the implicit update equation 
        using the internal optimizer.

        Parameters
        ----------
        u : numeric, array[numeric]
            function 'func' input value
        t : float
            evaluation time of function 'func'
        dt : float 
            integration timestep

        Returns
        -------
        err : float
            residual error of the fixed point update equation
        """

        #update the fixed point equation
        g = self.x_0 + dt * self.func(self.x, u, t)

        #use the numerical jacobian
        if self.jac is not None:

            #compute numerical jacobian
            jac_g = dt * self.jac(self.x, u, t)

            #optimizer step with block local jacobian
            self.x, err = self.opt.step(self.x, g, jac_g)

        else:
            #optimizer step (pure)
            self.x, err = self.opt.step(self.x, g, None)

        #return the fixed-point residual
        return err