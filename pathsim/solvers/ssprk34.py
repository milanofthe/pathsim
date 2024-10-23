########################################################################################
##
##               EXPLICIT STRONG STABILITY PRESERVING RUNGE-KUTTA INTEGRATOR
##                                (solvers/ssprk34.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

# import numpy as np

from ._solver import ExplicitSolver


# SOLVERS ==============================================================================

class SSPRK34(ExplicitSolver):
    """
    Strong Stability Preserving (SSP) 3-rd order 4 stage 
    (3,4) Runge-Kutta method
    
    This integrator has one more stage then SSPRK33 but is also
    3-rd order. So in terms or accuracy, they are the same but 
    the 4-th stage gives quite a lot more stability. 
    The stability region includes the point -4 on the real axis 
    and is even more stable then the classical 'RK4' method in 
    this aspect. But again it is 33% more expensive then SSPRK33 
    due to the additional stage. 

    If super high stability is required, this might be a good 
    choice.
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
        self.eval_stages = [0.0, 1/2, 1, 1/2]

        #butcher table
        self.BT = {0:[1/2],
                   1:[1/2, 1/2],
                   2:[1/6, 1/6, 1/6],
                   3:[1/6, 1/6, 1/6, 1/2]}


    def step(self, u, t, dt):
        """
        performs the (explicit) timestep for (t+dt) 
        based on the state and input at (t)
        """

        #buffer intermediate slope
        self.Ks[self.stage] = self.func(self.x, u, t)
        
        #compute slope and update state at stage
        self.x = dt * sum(k*b for k, b in zip(self.Ks.values(), self.BT[self.stage])) + self.x_0

        #wrap around stage counter
        self.stage = (self.stage + 1) % 4

        #no error estimate available
        return True, 0.0, 0.0, 1.0
