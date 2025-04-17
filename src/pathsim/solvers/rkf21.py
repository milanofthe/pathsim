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
    """3-stage 2-nd order embedded Runge-Kutta-Fehlberg method 
    with 2-nd order truncation error estimate that can be used to 
    adaptively control the timestep.

    This is an absolute classic, the three stages make it relatively 
    cheap, but its only second order and the error estimate is not that 
    accurate. However, if you need some kind of adaptive integrator, and 
    the timestep is not limited by the local truncation error, this solver 
    might be a good choice.

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
        self.BT = {0:[  1/2],
                   1:[1/256, 255/256],
                   2:[1/512, 255/256, 1/512]}

        #coefficients for local truncation error estimate
        self.TR = [1/512, 0, -1/512]