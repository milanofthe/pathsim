########################################################################################
##
##                      EXPLICIT and IMPLICIT EULER INTEGRATORS
##                                (solvers/euler.py)
##
########################################################################################

# IMPORTS ==============================================================================

from ._solver import ExplicitSolver, ImplicitSolver


# SOLVERS ==============================================================================

class EUF(ExplicitSolver):
    """Explicit Forward Euler (FE) integration method.

    This is the simplest explicit numerical integration method. It is first-order
    accurate (:math:`O(h)`) and generally not suitable for stiff problems due to its
    limited stability region.

    Method:

    .. math::
        
        x_{n+1} = x_n + dt \\cdot f(x_n, t_n)

    Characteristics:

        * Order: 1
        * Stages: 1
        * Explicit
        * Fixed timestep only
        * Not A-stable
        * Low accuracy and stability, but computationally very cheap.
    
    Note
    ----
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

        #get current state from history
        x_0 = self.history[0]

        #update state with euler step
        self.x = x_0 + dt * f

        #no error estimate available
        return True, 0.0, 1.0


class EUB(ImplicitSolver):
    """Implicit Backward Euler (BE) integration method.

    This is the simplest implicit numerical integration method. It is first-order
    accurate (:math:`O(h)`) and is A-stable and L-stable, making it suitable for very
    stiff problems where stability is paramount, although its low order limits
    accuracy for non-stiff problems or when high precision is required.

    Method:
    
    .. math::

        x_{n+1} = x_n + dt \\cdot f(x_{n+1}, t_{n+1})
    
    This implicit equation is solved iteratively using the internal optimizer.

    Characteristics:

        * Order: 1
        * Stages: 1 (Implicit)
        * Implicit
        * Fixed timestep only
        * A-stable, L-stable
        * Very stable, suitable for stiff problems, but low accuracy.

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

        #get current state from history
        x_0 = self.history[0]

        #update the fixed point equation
        g = x_0 + dt * f

        #use the numerical jacobian
        if J is not None:

            #optimizer step with block local jacobian
            self.x, err = self.opt.step(self.x, g, dt * J)

        else:
            #optimizer step (pure)
            self.x, err = self.opt.step(self.x, g, None)

        #return the fixed-point residual
        return err