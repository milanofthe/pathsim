########################################################################################
##
##               EXPLICIT STRONG STABILITY PRESERVING RUNGE-KUTTA INTEGRATOR
##                                (solvers/ssprk33.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._solver import ExplicitSolver


# SOLVERS ==============================================================================

class SSPRK33(ExplicitSolver):
    """
    Strong Stability Preserving (SSP) 3-rd order 
    three stage (3,3) Runge-Kutta method
    
    This integrator is more accurate and stable then SSPRK22 but 
    also 50% more expensive due to 3 instead of 2 stages. 
    Originally designed for hyperbolic PDEs, this is also a great 
    choice if accuracy and stability and still good speed are important.
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
        self.eval_stages = [0.0, 1.0, 0.5]

        #butcher table
        self.BT = {0:[1.0],
                   1:[1/4, 1/4],
                   2:[1/6, 1/6, 2/3]}


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
        self.stage = (self.stage + 1) % 3

        #no error estimate available
        return True, 0.0, 0.0, 1.0