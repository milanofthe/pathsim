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
    """Four-stage, 3rd order L-stable Diagonally Implicit Runge-Kutta (DIRK) method.

    L-stability (A-stability and stiffly accurate, i.e., :math:`|R(\\infty)| = 0`) makes
    this method suitable for stiff problems where damping of high-frequency components
    is desired. The stiffly accurate property ensures good behavior for problems with
    singular perturbations and differential-algebraic equations.

    Characteristics
    ---------------
    * Order: 3
    * Stages: 4 (Implicit)
    * Implicit (DIRK)
    * Fixed timestep only
    * L-stable (and thus A-stable)
    * Stiffly accurate

    When to Use
    -----------
    * **Stiff problems**: Excellent stability for very stiff ODEs
    * **Damping required**: L-stability damps high-frequency oscillations
    * **Differential-algebraic equations**: Stiffly accurate property helps with DAEs
    * **3rd order implicit**: Moderate accuracy with strong stability

    **Recommended** for stiff problems requiring 3rd order accuracy. For higher order,
    consider ESDIRK54. For variable timestep, use adaptive ESDIRK methods.

    References
    ----------
    .. [1] Crouzeix, M. (1975). "Sur l'approximation des équations différentielles
           opérationnelles linéaires par des méthodes de Runge-Kutta". PhD thesis,
           Université Paris VI.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems". Springer Series in Computational
           Mathematics, Vol. 14.
    .. [3] Alexander, R. (1977). "Diagonally implicit Runge-Kutta methods for stiff O.D.E.'s".
           SIAM Journal on Numerical Analysis, 14(6), 1006-1021.

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
        self.BT = {
            0: [1/2],
            1: [1/6, 1/2], 
            2: [-1/2, 1/2, 1/2], 
            3: [3/2, -3/2, 1/2, 1/2]
            }