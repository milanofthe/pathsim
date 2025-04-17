########################################################################################
##
##                   EMBEDDED DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/esdirk32.py)
##
##                                  Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import DiagonallyImplicitRungeKutta


# SOLVERS ==============================================================================

class ESDIRK32(DiagonallyImplicitRungeKutta):
    """Four-stage, 3rd order Embedded Singly Diagonally Implicit Runge-Kutta (ESDIRK) method.

    Features an embedded 2nd order method for adaptive step size control. The first stage
    is explicit (FSAL property not present in this specific implementation structure).
    Designed to be applicable to index-2 Differential Algebraic Equations (DAEs).

    FROM: 
        Williams et al.

    Characteristics:
        * Order: 3
        * Embedded Order: 2
        * Stages: 4 (1 Explicit, 3 Implicit)
        * Implicit (ESDIRK)
        * Adaptive timestep
        * A-stable 
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 4

        #order of scheme and embedded method
        self.n = 3
        self.m = 2

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1.0, 3/2, 1.0]

        #butcher table
        self.BT = {0:None, #explicit first stage
                   1:[1/2, 1/2],
                   2:[5/8, 3/8, 1/2],
                   3:[7/18, 1/3, -2/9, 1/2]}

        #coefficients for truncation error estimate
        self.TR = [-1/9, -1/6, -2/9, 1/2]