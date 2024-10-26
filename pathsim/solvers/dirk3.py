########################################################################################
##
##                       DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/dirk3.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._solver import ImplicitSolver
from ..utils.funcs import numerical_jacobian


# SOLVERS ==============================================================================

class DIRK3(ImplicitSolver):
    """
    Four-stage, 3rd order, L-stable Diagonally Implicit Runge–Kutta (DIRK) method.

    (from Wikipedia)
    """

    def __init__(self, 
                 initial_value=0, 
                 func=lambda x, u, t: u, 
                 jac=None, 
                 tolerance_lte_abs=1e-6, 
                 tolerance_lte_rel=1e-3):
        super().__init__(initial_value, 
                         func, 
                         jac, 
                         tolerance_lte_abs, 
                         tolerance_lte_rel)

        #counter for runge kutta stages
        self.stage = 0

        #slope coefficients for stages
        self.Ks = {}

        #intermediate evaluation times
        self.eval_stages = [1/2, 2/3, 1/2, 1.0]

        #butcher table
        self.BT = {0:[1/2],
                   1:[1/6, 1/2], 
                   2:[-1/2, 1/2, 1/2], 
                   3:[3/2, -3/2, 1/2, 1/2]}


    def solve(self, u, t, dt):
        """
        Solves the implicit update equation via anderson acceleration.
        """

        #update timestep weighted slope 
        self.Ks[self.stage] = self.func(self.x, u, t)

        #compute slope and update fixed-point equation
        slope = sum(k*b for k, b in zip(self.Ks.values(), self.BT[self.stage]))

        #use the jacobian
        if self.jac is not None:

            #most recent butcher coefficient
            b = self.BT[self.stage][self.stage]

            #compute jacobian of fixed-point equation
            jac_g = dt * b * self.jac(self.x, u, t)

            #anderson acceleration step with local newton
            self.x, err = self.acc.step(self.x, dt*slope + self.x_0, jac_g)

        else:
            #anderson acceleration step (pure)
            self.x, err = self.acc.step(self.x, dt*slope + self.x_0, None)

        #return the fixed-point residual
        return err


    def step(self, u, t, dt):
        """
        performs the timestep update
        """

        #restart anderson accelerator 
        self.acc.reset()

        #wrap around stage counter
        self.stage = (self.stage + 1) % 4

        #no error estimate available
        return True, 0.0, 0.0, 1.0