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
    """Two-stage, 2nd order Diagonally Implicit Runge-Kutta (DIRK) method.

    This specific method is SSP-optimal (largest radius of absolute monotonicity
    for a 2-stage, 2nd order DIRK), symplectic, and A-stable. It's a robust choice
    for moderately stiff problems where second-order accuracy is sufficient.

    Characteristics
    ---------------
    * Order: 2
    * Stages: 2 (Implicit)
    * Implicit (DIRK)
    * Fixed timestep only
    * A-stable, SSP-optimal, Symplectic

    When to Use
    -----------
    * **Moderately stiff problems**: Good entry-level implicit method for stiff ODEs
    * **SSP requirements with stiffness**: Combines strong stability preservation with A-stability
    * **Symplectic integration**: Preserves geometric structure in Hamiltonian systems
    * **Low-order implicit needs**: When 2nd order implicit accuracy is sufficient

    **Trade-off**: Lower order than ESDIRK or BDF methods but has SSP and symplectic
    properties. For higher accuracy on stiff problems, consider ESDIRK43 or BDF methods.

    References
    ----------
    .. [1] Ferracina, L., & Spijker, M. N. (2008). "Strong stability of singly-diagonally-
           implicit Runge-Kutta methods". Applied Numerical Mathematics, 58(11), 1675-1686.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems". Springer Series in Computational
           Mathematics, Vol. 14.

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
        self.BT = {
            0: [1/4],
            1: [1/2, 1/4]
            }

        #final evaluation
        self.A = [1/2, 1/2]