########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                 (solvers/rkf21.py)
##
##                                 Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RKF21(ExplicitRungeKutta):
    """Three-stage, 2nd order embedded Runge-Kutta-Fehlberg method.

    Features an embedded 1st order method for adaptive step size control. This is a
    classic low-order adaptive method. The three stages make it computationally cheap,
    but the low order limits accuracy. The error estimate is also less accurate than
    higher-order methods.

    Characteristics
    ---------------
    * Order: 2 (Propagating solution)
    * Embedded Order: 1 (Error estimation)
    * Stages: 3
    * Explicit
    * Adaptive timestep
    * Efficient but low accuracy

    When to Use
    -----------
    * **Computationally cheap adaptive stepping**: When you need some adaptive control but minimal cost
    * **Coarse integration**: Problems where high accuracy is not required
    * **Event detection**: When timestep is limited by events rather than truncation error
    * **Initial exploration**: Quick preliminary runs before using higher-order methods

    Note
    ----
    Low accuracy. For most applications requiring adaptive stepping, RKBS32 or RKDP54 are 
    better choices.

    References
    ----------
    .. [1] Fehlberg, E. (1969). "Low-order classical Runge-Kutta formulas with stepsize
           control and their application to some heat transfer problems". NASA Technical
           Report TR R-315.
    .. [2] Hairer, E., Nørsett, S. P., & Wanner, G. (1993). "Solving Ordinary
           Differential Equations I: Nonstiff Problems". Springer Series in Computational
           Mathematics, Vol. 8.

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 3

        #order of scheme and embedded method
        self.n = 2
        self.m = 1

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/2, 1]

        #extended butcher table 
        self.BT = {
            0: [  1/2],
            1: [1/256, 255/256],
            2: [1/512, 255/256, 1/512]
            }

        #coefficients for local truncation error estimate
        self.TR = [1/512, 0, -1/512]