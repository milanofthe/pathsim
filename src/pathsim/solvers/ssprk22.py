########################################################################################
##
##               EXPLICIT STRONG STABILITY PRESERVING RUNGE-KUTTA INTEGRATOR
##                                (solvers/ssprk22.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class SSPRK22(ExplicitRungeKutta):
    """Strong Stability Preserving (SSP) 2-nd order two stage (2,2) Runge-Kutta method,
    also known as the 'Heun-Method'.

    This integrator has a good trade off between speed, accuracy and stability.
    Especially for non-stiff linear systems, this is probably a great choice.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 2

        #order of scheme
        self.n = 2

        #intermediate evaluation times
        self.eval_stages = [0.0, 1.0]

        #butcher table
        self.BT = {0:[1.0],
                   1:[1/2, 1/2]}


    def interpolate(self, r, dt):
        k1, k2 = self.K[0], self.K[1]
        b1, b2 = r*(2-r)/2, r**2/2
        return self.x_0 + dt*(b1 * k1 + b2 * k2)