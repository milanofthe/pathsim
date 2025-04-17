########################################################################################
##
##                       DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/dirk3.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import DiagonallyImplicitRungeKutta


# SOLVERS ==============================================================================

class DIRK3(DiagonallyImplicitRungeKutta):
    """Four-stage, 3rd order, L-stable Diagonally Implicit Runge-Kutta (DIRK) method.

    L-stability (A-stability and stiffly accurate, i.e., :math:`|R(\\infty)| = 0`) makes
    this method suitable for stiff problems where damping of high-frequency components
    is desired.

    (Butcher tableau often attributed to Crouzeix, or Nørsett-Thomsen based on Wikipedia/Hairer).

    Characteristics:
        * Order: 3
        * Stages: 4 (Implicit)
        * Implicit (DIRK)
        * Fixed timestep only
        * L-stable (and thus A-stable)
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 4

        #order of scheme
        self.n = 3

        #intermediate evaluation times
        self.eval_stages = [1/2, 2/3, 1/2, 1.0]

        #butcher table
        self.BT = {0:[1/2],
                   1:[1/6, 1/2], 
                   2:[-1/2, 1/2, 1/2], 
                   3:[3/2, -3/2, 1/2, 1/2]}