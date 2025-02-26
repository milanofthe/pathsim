########################################################################################
##
##                   EMBEDDED DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/esdirk54.py)
##
##                                  Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import DiagonallyImplicitRungeKutta


# SOLVERS ==============================================================================

class ESDIRK54(DiagonallyImplicitRungeKutta):
    """7 stage 5-th order L-stable and stiffly accurate ESDIRK method with 
    embedded 4-th order method for stepsize control. This integrator is 
    suited for moderately stiff problems that require high accuracy.
    The first stage is explicit, followed by 6 implicit stages.

    FROM : 
        Diagonally implicit Runge–Kutta methods for stiff ODEs
        Christopher A.Kennedy, Mark H.Carpenter
        Applied Numerical Mathematics, 2019
        ESDIRK5(4)7L[2]SA2
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
        self.eval_stages = [0.0, 46/125, 7121331996143/11335814405378, 49/353, 
                            3706679970760/5295570149437, 347/382, 1.0]

        #butcher table
        self.BT = {0:None, #explicit first stage
                   1:[23/125, 23/125], 
                   2:[791020047304/3561426431547, 791020047304/3561426431547, 23/125], 
                   3:[-158159076358/11257294102345, -158159076358/11257294102345, 
                      -85517644447/5003708988389, 23/125], 
                   4:[-1653327111580/4048416487981, -1653327111580/4048416487981, 
                      1514767744496/9099671765375, 14283835447591/12247432691556, 23/125],
                   5:[-4540011970825/8418487046959, -4540011970825/8418487046959, 
                      -1790937573418/7393406387169, 10819093665085/7266595846747, 
                      4109463131231/7386972500302, 23/125],
                   6:[-188593204321/4778616380481, -188593204321/4778616380481, 
                      2809310203510/10304234040467, 1021729336898/2364210264653, 
                      870612361811/2470410392208, -1307970675534/8059683598661, 23/125]}

        #coefficients for truncation error estimate
        _A1 = [-188593204321/4778616380481, -188593204321/4778616380481, 
               2809310203510/10304234040467, 1021729336898/2364210264653, 
               870612361811/2470410392208, -1307970675534/8059683598661, 23/125]
        _A2 = [-582099335757/7214068459310, -582099335757/7214068459310, 
               615023338567/3362626566945, 3192122436311/6174152374399, 
               6156034052041/14430468657929, -1011318518279/9693750372484, 
               1914490192573/13754262428401]
        self.TR = [_a1 - _a2 for _a1, _a2 in zip(_A1, _A2)]