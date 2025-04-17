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
    """Four-stage, 3rd order explicit Runge-Kutta method by Bogacki and Shampine.

    Features an embedded 2nd order method for adaptive step size control (FSAL property -
    First Same As Last). The 3rd order result is used for propagation. Commonly used in
    software packages (e.g., MATLAB's ode23). Good for problems requiring low to moderate
    accuracy with efficiency.

    Characteristics:
        * Order: 3 (Propagating solution)
        * Embedded Order: 2 (Error estimation)
        * Stages: 4 (3 effective due to FSAL)
        * Explicit
        * Adaptive timestep
        * Efficient low-to-moderate accuracy solver.
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