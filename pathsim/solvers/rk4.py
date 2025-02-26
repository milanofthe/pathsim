########################################################################################
##
##                       CLASSICAL EXPLICIT RUNGE-KUTTA INTEGRATOR
##                                 (solvers/rk4.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RK4(ExplicitRungeKutta):
    """'The' classical 4-th order 4-stage Runge-Kutta method.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 4

        #order of scheme
        self.n = 4

        #intermediate evaluation times
        self.eval_stages = [0.0, 0.5, 0.5, 1.0]

        #butcher table
        self.BT = {0:[1/2],
                   1:[0.0, 1/2],
                   2:[0.0, 0.0, 1.0], 
                   3:[1/6, 2/6, 2/6, 1/6]}


    def interpolate(self, r, dt):
        k1, k2, k3, k4 = self.K[0], self.K[1], self.K[2], self.K[3]
        b1, b2, b3, b4 = r*(1-r)**2/6, r**2*(2-3*r)/2, r**2*(3*r-4)/2, r**3/6
        return self.x_0 + dt*(b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4)