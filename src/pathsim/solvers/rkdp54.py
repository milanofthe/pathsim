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
    """Dormand–Prince method with seven Runge-Kutta stages is 5-th order 
    accurate with an embedded 4-th order method. 

    The 5-th order method is used for timestepping (local extrapolation) 
    and the difference to the 5-th order solution is used as an estimate 
    for the local truncation error of the 4-th order solaution.
    
    Wikipedia:
        As of 2023, Dormand–Prince is the default method 
        in the 'ode45' solver for MATLAB

    Great choice for all kinds of problems that require high accuracy 
    and where the adaptive timestepping doesnt cause problems.
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
        self.BT = {0:[       1/5],
                   1:[      3/40,        9/40],
                   2:[     44/45,      -56/15,       32/9], 
                   3:[19372/6561, -25360/2187, 64448/6561, -212/729],
                   4:[ 9017/3168,     -355/33, 46732/5247,   49/176, -5103/18656],
                   5:[    35/384,           0,   500/1113,  125/192,  -2187/6784, 11/84],
                   6:[    35/384,           0,   500/1113,  125/192,  -2187/6784, 11/84]}

        #coefficients for local truncation error estimate
        self.TR = [71/57600, 0, - 71/16695, 71/1920, -17253/339200, 22/525, -1/40]