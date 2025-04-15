########################################################################################
##
##               EXPLICIT STRONG STABILITY PRESERVING RUNGE-KUTTA INTEGRATOR
##                                (solvers/ssprk34.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class SSPRK34(ExplicitRungeKutta):
    """Strong Stability Preserving (SSP) 3-rd order 4 stage 
    (3,4) Runge-Kutta method
    
    This integrator has one more stage then SSPRK33 but is also
    3-rd order. So in terms or accuracy, they are the same but 
    the 4-th stage gives quite a lot more stability. 
    The stability region includes the point -4 on the real axis 
    and is even more stable then the classical 'RK4' method in 
    this aspect. But again it is 33% more expensive then SSPRK33 
    due to the additional stage. 

    If super high stability is required, this might be a good 
    choice.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 4

        #order of scheme
        self.n = 3

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/2, 1, 1/2]

        #butcher table
        self.BT = {0:[1/2],
                   1:[1/2, 1/2],
                   2:[1/6, 1/6, 1/6],
                   3:[1/6, 1/6, 1/6, 1/2]}