########################################################################################
##
##                       DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/dirk2.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import DiagonallyImplicitRungeKutta


# SOLVERS ==============================================================================

class DIRK2(DiagonallyImplicitRungeKutta):
    """Two-stage, 2nd order, Diagonally Implicit Runge-Kutta (DIRK) method.

    This specific method is SSP-optimal (largest radius of absolute monotonicity
    for a 2-stage, 2nd order DIRK), symplectic, and A-stable. It's a robust choice
    for moderately stiff problems where second-order accuracy is sufficient.

    FROM:
        L. Ferracina and M.N. Spijker.
        Strong stability of singly-diagonally-implicit Runge-Kutta methods.
        Applied Numerical Mathematics, 58:1675–1686, 2008.

    Characteristics:
        * Order: 2
        * Stages: 2 (Implicit)
        * Implicit (DIRK)
        * Fixed timestep only
        * A-stable, SSP-optimal, Symplectic
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 2

        #order of scheme
        self.n = 2

        #intermediate evaluation times
        self.eval_stages = [1/4, 3/4]

        #butcher table
        self.BT = {0:[1/4],
                   1:[1/2, 1/4]}

        #final evaluation
        self.A = [1/2, 1/2]