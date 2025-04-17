########################################################################################
##
##               EXPLICIT STRONG STABILITY PRESERVING RUNGE-KUTTA INTEGRATOR
##                                (solvers/ssprk34.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class SSPRK34(ExplicitRungeKutta):
    """Four-stage, 3rd order, Strong Stability Preserving (SSP) explicit Runge-Kutta method.

    Provides a larger stability region compared to SSPRK33, particularly along the negative
    real axis, at the cost of an additional stage. Useful when stability is more critical
    than computational cost for a 3rd order explicit method.

    Characteristics:
        * Order: 3
        * Stages: 4
        * Explicit (SSP)
        * Fixed timestep only
        * Enhanced stability compared to SSPRK33.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 4

        #order of scheme
        self.n = 3

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/2, 1, 1/2]

        #butcher table
        self.BT = {0:[1/2],
                   1:[1/2, 1/2],
                   2:[1/6, 1/6, 1/6],
                   3:[1/6, 1/6, 1/6, 1/2]}