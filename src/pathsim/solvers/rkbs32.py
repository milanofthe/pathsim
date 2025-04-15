########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                (solvers/rkbs32.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RKBS32(ExplicitRungeKutta):
    """The Bogacki–Shampine method is a Runge–Kutta method of order three with four stages.
    It has an embedded second-order method which can be used to implement adaptive 
    step size. The Bogacki–Shampine method is implemented in the 'ode3' for fixed 
    step solver and 'ode23' for a variable step solver function in MATLAB.

    This is the adaptive variant. It is a good choice of low accuracy is acceptable.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 4

        #order of scheme and embedded method
        self.n = 3
        self.m = 2

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/2, 3/4, 1.0]
        
        #extended butcher table
        self.BT = {0:[1/2],
                   1:[0.0 , 3/4],
                   2:[2/9 , 1/3, 4/9],
                   3:[2/9 , 1/3, 4/9]}

        #coefficients for truncation error estimate
        self.TR = [-5/72, 1/12, 1/9, -1/8]