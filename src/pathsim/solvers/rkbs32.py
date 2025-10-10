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

    Features an embedded 2nd order method for adaptive step size control with FSAL
    (First Same As Last) property. The 3rd order result is used for propagation.
    Commonly used in software packages (e.g., MATLAB's ode23). Good for problems
    requiring low to moderate accuracy with efficiency.

    Characteristics
    ---------------
    * Order: 3 (Propagating solution)
    * Embedded Order: 2 (Error estimation)
    * Stages: 4 (3 effective due to FSAL)
    * Explicit
    * Adaptive timestep
    * Efficient low-to-moderate accuracy solver

    When to Use
    -----------
    * **Low-to-moderate accuracy needs**: When stringent accuracy is not required
    * **Efficiency-focused applications**: Cheaper than 5th order methods
    * **Smooth non-stiff problems**: Well-suited for mildly nonlinear problems
    * **Default low-order adaptive solver**: Good general-purpose choice for less demanding problems
    
    Note
    ----
    More efficient than 5th order methods but less accurate. For higher
    accuracy requirements, use RKDP54 or RKCK54. Nonetheless a good default 
    explicit adaptive timestep solver.

    References
    ----------
    .. [1] Bogacki, P., & Shampine, L. F. (1989). "A 3(2) pair of Runge-Kutta formulas".
           Applied Mathematics Letters, 2(4), 321-325.
    .. [2] Shampine, L. F., & Reichelt, M. W. (1997). "The MATLAB ODE Suite".
           SIAM Journal on Scientific Computing, 18(1), 1-22.
    .. [3] Hairer, E., Nørsett, S. P., & Wanner, G. (1993). "Solving Ordinary
           Differential Equations I: Nonstiff Problems". Springer Series in Computational
           Mathematics, Vol. 8.

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
        self.BT = {
            0: [1/2],
            1: [0.0 , 3/4],
            2: [2/9 , 1/3, 4/9],
            3: [2/9 , 1/3, 4/9]
            }

        #coefficients for truncation error estimate
        self.TR = [-5/72, 1/12, 1/9, -1/8]