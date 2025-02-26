########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                 (solvers/rkf45.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RKF45(ExplicitRungeKutta):
    """6-stage 4-th order embedded Runge-Kutta-Fehlberg method 
    with 5-th order truncation error estimate that can be used to 
    adaptively control the timestep. 

    Absolute classic but relatively slow.
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
        self.eval_stages = [0.0, 1/4, 3/8, 12/13, 1, 1/2]

        #extended butcher table 
        self.BT = {0:[      1/4],
                   1:[     3/32,       9/32],
                   2:[1932/2197, -7200/2197,  7296/2197],
                   3:[  439/216,         -8,   3680/513, -845/4104],
                   4:[    -8/27,          2, -3554/2565, 1859/4104, -11/40],
                   5:[   25/216,          0,  1408/2565, 2197/4104,   -1/5, 0]}

        #coefficients for local truncation error estimate
        self.TR = [1/360, 0, -128/4275, -2197/75240, 1/50, 2/55]