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
    """Three-stage, 3rd order Strong Stability Preserving (SSP) explicit Runge-Kutta method.

    Offers higher accuracy than SSPRK22 while maintaining the SSP property. This is the
    optimal 3-stage 3rd order SSP method. A popular choice for problems where TVD
    properties are important or when a simple, stable 3rd order explicit method is needed.

    Characteristics:
        * Order: 3
        * Stages: 3
        * Explicit (SSP)
        * Fixed timestep only
        * SSP coefficient: :math:`C = 1`
        * Optimal 3-stage SSP method
        * Good stability properties for an explicit 3rd order method

    When to Use
    -----------
    * **Hyperbolic conservation laws**: Standard choice for higher-order TVD schemes
    * **Higher accuracy than SSPRK22**: When 3rd order accuracy is needed with SSP
    * **WENO schemes**: Common pairing with weighted essentially non-oscillatory methods
    * **Compressible flow**: Euler and Navier-Stokes equations with shocks

    **Recommended** as the standard SSP method for most applications requiring 3rd order
    accuracy. For enhanced stability, consider SSPRK34 (4 stages).

    References
    ----------
    .. [1] Shu, C. W., & Osher, S. (1988). "Efficient implementation of essentially
           non-oscillatory shock-capturing schemes". Journal of Computational Physics,
           77(2), 439-471.
    .. [2] Gottlieb, S., Shu, C. W., & Tadmor, E. (2001). "Strong stability-preserving
           high-order time discretization methods". SIAM Review, 43(1), 89-112.
    .. [3] Gottlieb, S., Ketcheson, D. I., & Shu, C. W. (2011). "Strong Stability
           Preserving Runge-Kutta and Multistep Time Discretizations". World Scientific.

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
        self.BT = {
            0: [1.0],
            1: [1/4, 1/4],
            2: [1/6, 1/6, 2/3]
            }

    def interpolate(self, r, dt):
        k1, k2, k3 = self.K[0], self.K[1], self.K[2]
        b1, b2, b3 = r*(2-r)**2/2, r**2*(3-2*r)/2, r**3
        return self.x_0 + dt*(b1 * k1 + b2 * k2 + b3 * k3)