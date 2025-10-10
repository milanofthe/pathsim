########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                (solvers/rkdp54.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RKDP54(ExplicitRungeKutta):
    """Seven-stage, 5th order explicit Runge-Kutta method by Dormand and Prince (DOPRI5).

    Features an embedded 4th order method. Widely considered one of the most efficient
    general-purpose adaptive step size solvers for non-stiff problems requiring moderate
    to high accuracy. The 5th order result is used for propagation. Used as the basis for
    MATLAB's ode45. FSAL property (not available in this implementation).

    Characteristics
    ---------------
    * Order: 5 (Propagating solution)
    * Embedded Order: 4
    * Stages: 7 (6 effective due to FSAL, not here though)
    * Explicit
    * Adaptive timestep
    * Industry-standard adaptive solver

    When to Use
    -----------
    * **Default adaptive solver**: Excellent first choice for most non-stiff problems
    * **Moderate-to-high accuracy**: 5th order provides good accuracy efficiently
    * **General-purpose integration**: Reliable for a wide variety of ODE systems
    * **Industry standard**: Well-tested and widely used in production software
    
    Note
    ----
    Recommended as the primary adaptive solver for non-stiff problems. For stiff
    problems, use BDF or ESDIRK methods. For very high accuracy, consider RKV65 or RKDP87.

    References
    ----------
    .. [1] Dormand, J. R., & Prince, P. J. (1980). "A family of embedded Runge-Kutta
           formulae". Journal of Computational and Applied Mathematics, 6(1), 19-26.
    .. [2] Shampine, L. F., & Reichelt, M. W. (1997). "The MATLAB ODE Suite".
           SIAM Journal on Scientific Computing, 18(1), 1-22.
    .. [3] Hairer, E., Nørsett, S. P., & Wanner, G. (1993). "Solving Ordinary
           Differential Equations I: Nonstiff Problems". Springer Series in Computational
           Mathematics, Vol. 8.

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 7

        #order of scheme and embedded method
        self.n = 5
        self.m = 4

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
        
        #extended butcher table
        self.BT = {
            0: [       1/5],
            1: [      3/40,        9/40],
            2: [     44/45,      -56/15,       32/9], 
            3: [19372/6561, -25360/2187, 64448/6561, -212/729],
            4: [ 9017/3168,     -355/33, 46732/5247,   49/176, -5103/18656],
            5: [    35/384,           0,   500/1113,  125/192,  -2187/6784, 11/84],
            6: [    35/384,           0,   500/1113,  125/192,  -2187/6784, 11/84]
            }

        #coefficients for local truncation error estimate
        self.TR = [71/57600, 0, - 71/16695, 71/1920, -17253/339200, 22/525, -1/40]