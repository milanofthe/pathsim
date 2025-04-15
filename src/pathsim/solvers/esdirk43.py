########################################################################################
##
##                   EMBEDDED DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/esdirk32.py)
##
##                                  Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ._rungekutta import DiagonallyImplicitRungeKutta


# SOLVERS ==============================================================================

class ESDIRK43(DiagonallyImplicitRungeKutta):
    """6 stage 4-th order ESDIRK method with embedded 3-rd order method for stepsize control. 
    The first stage is explicit, followed by 5 implicit stages.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 6

        #order of scheme and embedded method
        self.n = 4
        self.m = 3

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/2, (2-np.sqrt(2))/4, 2012122486997/3467029789466, 1.0, 1.0]

        #butcher table
        self.BT = {0:None, # explicit first stage
                   1:[1/4, 1/4],
                   2:[-1356991263433/26208533697614, -1356991263433/26208533697614, 1/4],
                   3:[-1778551891173/14697912885533, -1778551891173/14697912885533, 
                      7325038566068/12797657924939, 1/4],
                   4:[-24076725932807/39344244018142, -24076725932807/39344244018142, 
                      9344023789330/6876721947151, 11302510524611/18374767399840, 1/4],
                   5:[657241292721/9909463049845, 657241292721/9909463049845, 
                      1290772910128/5804808736437, 1103522341516/2197678446715, -3/28, 1/4]}

        #coefficients for truncation error estimate
        _A1 = [657241292721/9909463049845, 657241292721/9909463049845, 
               1290772910128/5804808736437, 1103522341516/2197678446715, -3/28, 1/4]
        _A2 = [-71925161075/3900939759889, -71925161075/3900939759889, 
               2973346383745/8160025745289, 3972464885073/7694851252693, 
               -263368882881/4213126269514, 3295468053953/15064441987965]
        self.TR = [a1 - a2 for a1, a2 in zip(_A1, _A2)]