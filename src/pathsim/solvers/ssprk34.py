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
    """Four-stage, 3rd order Strong Stability Preserving (SSP) explicit Runge-Kutta method.

    Provides a larger stability region and higher SSP coefficient compared to SSPRK33,
    particularly along the negative real axis, at the cost of an additional stage. Useful
    when stability is more critical than computational cost for a 3rd order explicit method.

    Characteristics
    ---------------
    * Order: 3
    * Stages: 4
    * Explicit (SSP)
    * Fixed timestep only
    * SSP coefficient: :math:`C = 2`
    * Enhanced stability compared to SSPRK33

    When to Use
    -----------
    * **Larger timesteps**: SSP coefficient of 2 allows larger stable timesteps
    * **Difficult hyperbolic problems**: More robust than SSPRK33 for challenging cases
    * **Extra stability needed**: When SSPRK33 exhibits instabilities
    * **Worth extra stage**: When the improved stability justifies 4 stages vs 3

    **Trade-off**: More expensive than SSPRK33 but allows larger timesteps and better
    stability. Use when stability is critical.

    References
    ----------
    .. [1] Spiteri, R. J., & Ruuth, S. J. (2002). "A new class of optimal high-order
           strong-stability-preserving time discretization methods". SIAM Journal on
           Numerical Analysis, 40(2), 469-491.
    .. [2] Gottlieb, S., Shu, C. W., & Tadmor, E. (2001). "Strong stability-preserving
           high-order time discretization methods". SIAM Review, 43(1), 89-112.
    .. [3] Gottlieb, S., Ketcheson, D. I., & Shu, C. W. (2011). "Strong Stability
           Preserving Runge-Kutta and Multistep Time Discretizations". World Scientific.

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
        self.BT = {
            0: [1/2],
            1: [1/2, 1/2],
            2: [1/6, 1/6, 1/6],
            3: [1/6, 1/6, 1/6, 1/2]
            }