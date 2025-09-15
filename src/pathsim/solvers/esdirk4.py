########################################################################################
##
##                       DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/esdirk4.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import DiagonallyImplicitRungeKutta


# SOLVERS ==============================================================================

class ESDIRK4(DiagonallyImplicitRungeKutta):
    """Six-stage, 4th order Singly Diagonally Implicit Runge-Kutta (ESDIRK) method.

    Features an explicit first stage (making it ESDIRK). This specific tableau is often
    designed for handling stiff problems and potentially Differential Algebraic Equations (DAEs)
    of index up to two or three. Does not have an embedded method for error estimation in this
    implementation (fixed step only).

    Characteristics:
       * Order: 4
       * Stages: 6 (1 Explicit, 5 Implicit)
       * Implicit (ESDIRK)
       * Fixed timestep only
       * A-stable
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 6

        #order of scheme
        self.n = 4

        #intermediate evaluation times
        self.eval_stages = [
            0.0, 1/2, 1/6, 37/40, 1/2, 1.0
            ]

        #butcher table
        self.BT = {
            0: None, #explicit first stage
            1: [1/4, 1/4],
            2: [-1/36, -1/18, 1/4],
            3: [-21283/32000, -5143/64000, 90909/64000, 1/4],
            4: [46010759/749250000, -737693/40500000, 10931269/45500000, -1140071/34090875, 1/4],
            5: [89/444, 89/804756, -27/364, -20000/171717, 843750/1140071, 1/4]
            }