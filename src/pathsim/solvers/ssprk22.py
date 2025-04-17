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
    """Two-stage, 2nd order, Strong Stability Preserving (SSP) explicit Runge-Kutta method.

    Also known as the explicit midpoint method or Heun's method. SSP methods are designed
    to preserve stability properties (like total variation diminishing - TVD) when solving
    hyperbolic PDEs, but are also effective general-purpose low-order explicit methods.

    Characteristics:
        * Order: 2
        * Stages: 2
        * Explicit (SSP)
        * Fixed timestep only
        * Good balance of simplicity, cost, and stability (for an explicit method).
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