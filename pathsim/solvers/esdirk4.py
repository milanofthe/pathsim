########################################################################################
##
##                       DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/esdirk4.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._solver import ImplicitSolver
from ..utils.funcs import numerical_jacobian


# SOLVERS ==============================================================================

class ESDIRK4(ImplicitSolver):
    """
    6-stage, 4th order Diagonally Implicit Runge–Kutta (DIRK) method 
    with explicit first stage that is specifically designed to handle 
    differential algebraic equations of indices up to two or three.
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
        self.eval_stages = [0.0, 1/2, 1/6, 37/40, 1/2, 1.0]

        #butcher table
        self.BT = {0:[0.0],
                   1:[1/4, 1/4],
                   2:[-1/36, -1/18, 1/4],
                   3:[-21283/32000, -5143/64000, 90909/64000, 1/4],
                   4:[46010759/749250000, -737693/40500000, 10931269/45500000, -1140071/34090875, 1/4],
                   5:[89/444, 89/804756, -27/364, -20000/171717, 843750/1140071, 1/4]}


    def solve(self, u, t, dt):
        """
        Solves the implicit update equation via anderson acceleration.
        """

        #first stage is explicit
        if self.stage == 0:
            return 0.0
            
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

        #first stage is explicit
        if self.stage == 0:
            self.Ks[self.stage] = self.func(self.x, u, t)

        #restart anderson accelerator 
        self.acc.reset()

        #wrap around stage counter
        self.stage = (self.stage + 1) % 6

        #no error estimate available
        return True, 0.0, 0.0, 1.0

