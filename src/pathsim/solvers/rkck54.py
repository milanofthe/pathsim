########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                (solvers/rkck54.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RKCK54(ExplicitRungeKutta):
    """Six-stage, 5th order explicit Runge-Kutta method by Cash and Karp.

    Features an embedded 4th order method. The difference between the 5th and 4th order
    results provides a 5th order error estimate, typically used to control the step size
    while propagating the 5th order solution (local extrapolation). Known for better
    stability properties compared to RKF45.

    Characteristics
    ---------------
    * Order: 5 (Propagating solution)
    * Embedded Order: 4
    * Stages: 6
    * Explicit
    * Adaptive timestep
    * Good stability, suitable for moderate-to-high accuracy requirements

    When to Use
    -----------
    * **Improved stability over RKF45**: When RKF45 exhibits stability issues
    * **Moderate-to-high accuracy needs**: 5th order for better accuracy than 3rd order methods
    * **Non-stiff problems**: Excellent for smooth, non-stiff ODEs
    * **Alternative to RKDP54**: Similar performance, sometimes better for specific problems

    Note
    ----
    RKDP54 is generally more efficient when first-same-as-last (FSAL) property is implemented 
    (which it currently is not!), but RKCK54 can have better stability for certain problems. 
    Both are excellent 5th order adaptive methods.

    References
    ----------
    .. [1] Cash, J. R., & Karp, A. H. (1990). "A variable order Runge-Kutta method for
           initial value problems with rapidly varying right-hand sides". ACM Transactions
           on Mathematical Software, 16(3), 201-222.
    .. [2] Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007).
           "Numerical Recipes: The Art of Scientific Computing" (3rd ed.). Cambridge
           University Press.
    .. [3] Hairer, E., Nørsett, S. P., & Wanner, G. (1993). "Solving Ordinary
           Differential Equations I: Nonstiff Problems". Springer Series in Computational
           Mathematics, Vol. 8.

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 6

        #order of scheme and embedded method
        self.n = 5
        self.m = 4

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/5, 3/10, 3/5, 1, 7/8]

        #extended butcher table 
        self.BT = {
            0: [       1/5],
            1: [      3/40,    9/40],
            2: [      3/10,   -9/10,       6/5],
            3: [    -11/54,     5/2,    -70/27,        35/27],
            4: [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096],
            5: [    37/378,       0,   250/621,      125/594,        0, 512/1771]
            }

        #coefficients for local truncation error estimate
        self.TR = [-277/64512, 0, 6925/370944, -6925/202752, -277/14336, 277/7084]