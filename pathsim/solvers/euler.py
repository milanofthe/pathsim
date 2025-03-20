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

    def step(self, f, dt):
        """performs the explicit forward timestep for (t+dt) 
        based on the state and input at (t)

        Parameters
        ----------
        f : array_like
            evaluation of function
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
        self.x = self.x_0 + dt * f

        #no error estimate available
        return True, 0.0, 1.0


class EUB(ImplicitSolver):
    """Class that performs implicit (backward) euler integration
    it holds the state and implements the solution of the 
    implicit update equation at each timestep.

    Its an absolute classic and ok for moderately stiff problems 
    that dont require super high accuracy.
    """

    def solve(self, f, J, dt):
        """Solves the implicit update equation 
        using the internal optimizer.

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

        #update the fixed point equation
        g = self.x_0 + dt*f

        #use the numerical jacobian
        if J is not None:

            #optimizer step with block local jacobian
            self.x, err = self.opt.step(self.x, g, dt*J)

        else:
            #optimizer step (pure)
            self.x, err = self.opt.step(self.x, g, None)

        #return the fixed-point residual
        return err