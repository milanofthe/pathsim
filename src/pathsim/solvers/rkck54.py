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
    while propagating the 5th order solution (local extrapolation). Known for good stability
    properties compared to RKF45.

    Characteristics:
        * Order: 5 (Propagating solution)
        * Embedded Order: 4
        * Stages: 6
        * Explicit
        * Adaptive timestep
        * Good stability, suitable for moderate accuracy requirements.
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
        self.BT = {0:[       1/5],
                   1:[      3/40,    9/40],
                   2:[      3/10,   -9/10,       6/5],
                   3:[    -11/54,     5/2,    -70/27,        35/27],
                   4:[1631/55296, 175/512, 575/13824, 44275/110592, 253/4096],
                   5:[    37/378,       0,   250/621,      125/594,        0, 512/1771]}

        #coefficients for local truncation error estimate
        self.TR = [-277/64512, 0, 6925/370944, -6925/202752, -277/14336, 277/7084]