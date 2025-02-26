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
    """The 2-stage SSP-optimal Diagonally Implicit Runge–Kutta (DIRK) method 
    of second order, namely the second order RK with the largest radius 
    of absolute monotonicity. 
    It is also symplectic and the optimal 2-stage second order implicit RK.
    
    FROM : 
        L. Ferracina and M.N. Spijker. 
        Strong stability of singlydiagonally-implicit Runge-Kutta methods. 
        Applied Numerical Mathematics, 58:1675–1686, 2008.
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