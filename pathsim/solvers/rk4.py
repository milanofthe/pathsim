########################################################################################
##
##                       CLASSICAL EXPLICIT RUNGE-KUTTA INTEGRATOR
##                                 (solvers/rk4.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._solver import ExplicitSolver


# SOLVERS ==============================================================================

class RK4(ExplicitSolver):
    """
    'The' classical 4-th order 4-stage Runge-Kutta method.
    """

    def __init__(self, initial_value=0, func=lambda x, u, t: u, jac=None, tolerance_lte=1e-6):
        super().__init__(initial_value, func, jac, tolerance_lte)

        #counter for runge kutta stages
        self.stage = 0

        #slope coefficients for stages
        self.Ks = {}

        #intermediate evaluation times
        self.eval_stages = [0.0, 0.5, 0.5, 1.0]

        #butcher table
        self.BT = {0:[1/2],
                   1:[0.0, 1/2],
                   2:[0.0, 0.0, 1.0], 
                   3:[1/6, 2/6, 2/6, 1/6]}
                   

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
        self.stage = (self.stage + 1) % 4

        #no error estimate available
        return True, 0.0, 1.0

