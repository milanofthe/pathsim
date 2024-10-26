########################################################################################
##
##                       DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/dirk2.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._solver import ImplicitSolver
from ..utils.funcs import numerical_jacobian


# SOLVERS ==============================================================================

class DIRK2(ImplicitSolver):
    """
    The 2-stage SSP-optimal Diagonally Implicit Runge–Kutta (DIRK) method 
    of second order, namely the second order RK with the largest radius 
    of absolute monotonicity. 
    It is also symplectic and the optimal 2-stage second order implicit RK.
    
    FROM : 
        L. Ferracina and M.N. Spijker. 
        Strong stability of singlydiagonally-implicit Runge-Kutta methods. 
        Applied Numerical Mathematics, 58:1675–1686, 2008.
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
        self.eval_stages = [1/4, 3/4]

        #butcher table
        self.BT = {0:[1/4],
                   1:[1/2, 1/4]}

        #final evaluation
        self.A = [1/2, 1/2]


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

        #compute final output in second stage
        if self.stage == 1:
            slope = 0.0
            for i, a in enumerate(self.A):
                slope = slope + self.Ks[i] * a 
            self.x = dt*slope + self.x_0

        #restart anderson accelerator 
        self.acc.reset()

        #wrap around stage counter
        self.stage = (self.stage + 1) % 2

        #no error estimate available
        return True, 0.0, 0.0, 1.0

