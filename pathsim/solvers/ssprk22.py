########################################################################################
##
##               EXPLICIT STRONG STABILITY PRESERVING RUNGE-KUTTA INTEGRATOR
##                                (solvers/ssprk22.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._solver import ExplicitSolver


# SOLVERS ==============================================================================

class SSPRK22(ExplicitSolver):
    """
    Strong Stability Preserving (SSP) 2-nd order two stage (2,2) Runge-Kutta method,
    also known as the 'Heun-Method'.

    This integrator has a good trade off between speed, accuracy and stability.
    Especially for non-stiff linear systems, this is probably a great choice.
    """

    def __init__(self, initial_value=0, func=lambda x, u, t: u, jac=None, tolerance_lte=1e-6):
        super().__init__(initial_value, func, jac, tolerance_lte)

        #counter for runge kutta stages
        self.stage = 0

        #slope coefficients for stages
        self.Ks = {}

        #intermediate evaluation times
        self.eval_stages = [0.0, 1.0]

        #butcher table
        self.BT = {0:[1.0],
                   1:[1/2, 1/2]}
                   

    def step(self, u, t, dt):
        """
        performs the (explicit) timestep for (t+dt) 
        based on the state and input at (t)
        """

        #buffer intermediate slope
        self.Ks[self.stage] = self.func(self.x, u, t)
        
        #update state at stage
        slope = 0.0
        for i, b in enumerate(self.BT[self.stage]):
            slope += self.Ks[i] * b
        self.x = dt * slope + self.x_0
        
        #wrap around stage counter
        self.stage = (self.stage + 1) % 2

        #no error estimate available
        return True, 0.0, 1.0