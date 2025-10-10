########################################################################################
##
##               EXPLICIT STRONG STABILITY PRESERVING RUNGE-KUTTA INTEGRATOR
##                                (solvers/ssprk22.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class SSPRK22(ExplicitRungeKutta):
    """Two-stage, 2nd order Strong Stability Preserving (SSP) explicit Runge-Kutta method.

    Also known as Heun's method or the explicit midpoint method. SSP methods are designed
    to preserve stability properties (like total variation diminishing - TVD) when solving
    hyperbolic PDEs with spatial discretizations that have strong stability properties.
    Also effective as a general-purpose low-order explicit method.

    Characteristics
    ---------------
    * Order: 2
    * Stages: 2
    * Explicit (SSP)
    * Fixed timestep only
    * SSP coefficient: :math:`C = 1`
    * Good balance of simplicity, cost, and stability

    When to Use
    -----------
    * **Hyperbolic PDEs**: Ideal for shock-capturing schemes and conservation laws
    * **TVD/SSP requirements**: When preserving monotonicity or boundedness is critical
    * **Discontinuous solutions**: Shocks, contact discontinuities, rarefactions
    * **Method of lines**: Time integration of spatially discretized PDEs
    
    Note
    ----
    Computational fluid dynamics, shallow water equations, traffic flow,
    Burgers' equation, Euler equations.

    References
    ----------
    .. [1] Shu, C. W., & Osher, S. (1988). "Efficient implementation of essentially
           non-oscillatory shock-capturing schemes". Journal of Computational Physics,
           77(2), 439-471.
    .. [2] Gottlieb, S., Shu, C. W., & Tadmor, E. (2001). "Strong stability-preserving
           high-order time discretization methods". SIAM Review, 43(1), 89-112.
    .. [3] Ketcheson, D. I. (2008). "Highly efficient strong stability-preserving
           Runge-Kutta methods with low-storage implementations". SIAM Journal on
           Scientific Computing, 30(4), 2113-2136.

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 2

        #order of scheme
        self.n = 2

        #intermediate evaluation times
        self.eval_stages = [0.0, 1.0]

        #butcher table
        self.BT = {
            0: [1.0],
            1: [1/2, 1/2]
            }


    def interpolate(self, r, dt):
        k1, k2 = self.K[0], self.K[1]
        b1, b2 = r*(2-r)/2, r**2/2
        return self.x_0 + dt*(b1 * k1 + b2 * k2)