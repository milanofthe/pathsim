########################################################################################
##
##               EXPLICIT STRONG STABILITY PRESERVING RUNGE-KUTTA INTEGRATOR
##                                (solvers/ssprk33.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class SSPRK33(ExplicitRungeKutta):
    """Strong Stability Preserving (SSP) 3-rd order 
    three stage (3,3) Runge-Kutta method
    
    This integrator is more accurate and stable then SSPRK22 but 
    also 50% more expensive due to 3 instead of 2 stages. 
    Originally designed for hyperbolic PDEs, this is also a great 
    choice if accuracy and stability and still good speed are important.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 3

        #order of scheme
        self.n = 3

        #intermediate evaluation times
        self.eval_stages = [0.0, 1.0, 0.5]

        #butcher table
        self.BT = {0:[1.0],
                   1:[1/4, 1/4],
                   2:[1/6, 1/6, 2/3]}

    def interpolate(self, r, dt):
        k1, k2, k3 = self.K[0], self.K[1], self.K[2]
        b1, b2, b3 = r*(2-r)**2/2, r**2*(3-2*r)/2, r**3
        return self.x_0 + dt*(b1 * k1 + b2 * k2 + b3 * k3)