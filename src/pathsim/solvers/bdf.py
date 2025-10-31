########################################################################################
##
##                         BACKWARD DIFFERENTIATION FORMULAS
##                                 (solvers/bdf.py)
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from collections import deque

from ._solver import ImplicitSolver
from .dirk3 import DIRK3


# BASE BDF SOLVER ======================================================================

class BDF(ImplicitSolver):
    """Base class for the backward differentiation formula (BDF) integrators.
    
    Notes
    ----- 
    This solver class is not intended to be used directly

    Attributes
    ----------
    x : numeric, array[numeric]
        internal 'working' state
    n : int
        order of integration scheme
    s : int
        number of internal intermediate stages
    stage : int
        counter for current intermediate stage
    eval_stages : list[float]
        rations for evaluation times of intermediate stages
    opt : NewtonAnderson, Anderson, etc.
        optimizer instance to solve the implicit update equation
    K : dict[int: list[float]]
        bdf coefficients for the state buffer for each order
    F : dict[int: float]
        bdf coefficients for the function 'func' for each order
    history : deque[numeric]
        internal history of past results
    startup : Solver
        internal solver instance for startup (building history) 
        of multistep methods (using 'DIRK3' for 'BDF' methods)
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order
        self.n = None

        #bdf coefficients for orders 1 to 6
        self.K = {
            1:[1.0], 
            2:[4/3, -1/3], 
            3:[18/11, -9/11, 2/11], 
            4:[48/25, -36/25, 16/25, -3/25],
            5:[300/137, -300/137, 200/137, -75/137, 12/137],
            6:[360/147, -450/147, 400/147, -225/147, 72/147, -10/147]
            }
        self.F = {1:1.0, 2:2/3, 3:6/11, 4:12/25, 5:60/137, 6:60/147}

        #initialize startup solver from 'self' and flag
        self._needs_startup = True
        self.startup = DIRK3.cast(self, self.parent)


    @classmethod
    def cast(cls, other, parent, **solver_kwargs):
        """cast to this solver needs special handling of startup method

        Parameters
        ----------
        other : Solver
            solver instance to cast new instance of this class from
        parent : None | Solver
            solver instance to use as parent
        solver_kwargs : dict
            other args for the solver

        Returns
        -------
        engine : BDF
            instance of `BDF` solver with params and state from `other`
        """
        engine = super().cast(other, parent, **solver_kwargs)
        engine.startup = DIRK3.cast(engine, parent)

        return engine


    def stages(self, t, dt):
        """Generator that yields the intermediate evaluation 
        time during the timestep 't + ratio * dt'.

        Parameters
        ----------
        t : float 
            evaluation time
        dt : float
            integration timestep
        """

        #not enough history for full order -> stages of startup method
        if self._needs_startup:
            for self.stage, _t in enumerate(self.startup.stages(t, dt)):
                yield _t
        else:
            for _t in super().stages(t, dt):
                yield _t


    def reset(self):
        """"Resets integration engine to initial state."""

        #clear history (BDF solution buffer)
        self.history.clear()

        #overwrite state with initial value (ensure array format)
        self.x = np.atleast_1d(self.initial_value).copy()

        #reset startup solver
        self.startup.reset()


    def buffer(self, dt):
        """buffer the state for the multistep method
        
        Parameters
        ----------
        dt : float
            integration timestep
        """
            
        #reset optimizer
        self.opt.reset()

        #add current solution to history
        self.history.appendleft(self.x)

        #flag for startup method, not enough history
        self._needs_startup = len(self.history) < self.n

        #buffer with startup method
        if self._needs_startup:
            self.startup.buffer(dt)


    def solve(self, f, J, dt):
        """Solves the implicit update equation using the optimizer of the engine.

        Parameters
        ----------
        f : array_like
            evaluation of function
        J : array_like
            evaluation of jacobian of function
        dt : float 
            integration timestep

        Returns
        -------
        err : float
            residual error of the fixed point update equation
        """

        #not enough history for full order -> solve with startup method
        if self._needs_startup:
            err = self.startup.solve(f, J, dt)
            self.x = self.startup.get()
            return err

        #fixed-point function update
        g = self.F[self.n] * dt * f
        for b, k in zip(self.history, self.K[self.n]):
            g = g + b * k

        #use the jacobian
        if J is not None:

            #optimizer step with block local jacobian
            self.x, err = self.opt.step(self.x, g, self.F[self.n] * dt * J)

        else:
            #optimizer step (pure)
            self.x, err = self.opt.step(self.x, g, None)

        #return the fixed-point residual
        return err


    def step(self, f, dt):
        """Performs the explicit timestep for (t+dt) based 
        on the state and input at (t).

        Note
        ----
        This is only required for the startup solver.

        Parameters
        ----------
        f : numeric, array[numeric]
            evaluation of rhs function
        dt : float 
            integration timestep

        Returns 
        -------
        success : bool
            True if the timestep was successful
        error : float
            estimated error of the internal error controller
        scale : float
            estimated timestep rescale factor for error control
        """

        #not enough histors -> step the startup solver
        if self._needs_startup:
            self.startup.step(f, dt)
            self.x = self.startup.get()
            
        return True, 0.0, 1.0


# SOLVERS ==============================================================================

class BDF2(BDF):
    """Fixed-step 2nd order Backward Differentiation Formula (BDF).

    Implicit linear multistep method using the previous two solution points. A-stable,
    making it excellent for stiff problems. Uses DIRK3 startup method for the first steps.

    Characteristics
    ---------------
    * Order: 2
    * Implicit Multistep
    * Fixed timestep only
    * A-stable

    When to Use
    -----------
    * **Stiff problems with fixed timestep**: Classic choice for stiff ODEs
    * **Long-time integration**: Very stable for extended simulations
    * **Known timestep**: When timestep is predetermined
    * **Efficient stiff solver**: Lower overhead than higher-order BDFs

    **Recommended** for fixed-timestep stiff problems. For adaptive stepping, use GEAR21
    or ESDIRK methods.

    References
    ----------
    .. [1] Gear, C. W. (1971). "Numerical Initial Value Problems in Ordinary
           Differential Equations". Prentice-Hall.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems". Springer Series in Computational
           Mathematics, Vol. 14.

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order (local)
        self.n = 2

        #longer history for BDF
        self.history = deque([], maxlen=2)


class BDF3(BDF):
    """Fixed-step 3rd order Backward Differentiation Formula (BDF).

    Implicit linear multistep method using the previous three solution points. A(alpha)-stable
    with :math:`\\alpha \\approx 86^\\circ`, providing excellent stability for stiff problems.
    Uses DIRK3 startup method for initial steps.

    Characteristics
    ---------------
    * Order: 3
    * Implicit Multistep
    * Fixed timestep only
    * A(alpha)-stable (:math:`\\alpha \\approx 86^\\circ`)

    When to Use
    -----------
    * **Stiff problems with higher accuracy**: 3rd order for better accuracy than BDF2
    * **Fixed-timestep applications**: When timestep is predetermined
    * **Good stability/accuracy balance**: Better accuracy with still-excellent stability
    * **Chemical kinetics**: Common in reaction-diffusion problems

    **Trade-off**: Slightly less stable than BDF2, but more accurate. For adaptive stepping,
    use GEAR32 or ESDIRK43.

    References
    ----------
    .. [1] Gear, C. W. (1971). "Numerical Initial Value Problems in Ordinary
           Differential Equations". Prentice-Hall.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems". Springer Series in Computational
           Mathematics, Vol. 14.

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order (local)
        self.n = 3

        #longer history for BDF
        self.history = deque([], maxlen=3)


class BDF4(BDF):
    """Fixed-step 4th order Backward Differentiation Formula (BDF).

    Implicit linear multistep method using the previous four solution points. A(alpha)-stable
    with :math:`\\alpha \\approx 73^\\circ`. Good for stiff problems requiring moderate-to-high
    accuracy. Uses DIRK3 startup method for initial steps.

    Characteristics
    ---------------
    * Order: 4
    * Implicit Multistep
    * Fixed timestep only
    * A(alpha)-stable (:math:`\\alpha \\approx 73^\\circ`)

    When to Use
    -----------
    * **Moderate-to-high accuracy on stiff problems**: 4th order with good stability
    * **Fixed timestep**: When timestep is predetermined
    * **Accurate stiff solver**: Higher accuracy than BDF3
    * **Scientific computing**: Common in engineering simulations

    **Note**: Stability angle is smaller than BDF3. For very stiff problems, BDF2 or BDF3
    may be more robust. For adaptive stepping, use GEAR43 or ESDIRK43.

    References
    ----------
    .. [1] Gear, C. W. (1971). "Numerical Initial Value Problems in Ordinary
           Differential Equations". Prentice-Hall.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems". Springer Series in Computational
           Mathematics, Vol. 14.

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order (local)
        self.n = 4

        #longer history for BDF
        self.history = deque([], maxlen=4)


class BDF5(BDF):
    """Fixed-step 5th order Backward Differentiation Formula (BDF).

    Implicit linear multistep method using the previous five solution points. A(alpha)-stable
    with :math:`\\alpha \\approx 51^\\circ`. Suitable for stiff problems requiring high accuracy,
    but with reduced stability angle. Uses DIRK3 startup method for initial steps.

    Characteristics
    ---------------
    * Order: 5
    * Implicit Multistep
    * Fixed timestep only
    * A(alpha)-stable (:math:`\\alpha \\approx 51^\\circ`)

    When to Use
    -----------
    * **High accuracy on mildly stiff problems**: 5th order when stability angle is sufficient
    * **Fixed timestep applications**: When timestep is predetermined
    * **Smooth stiff problems**: Problems without extreme stiffness
    * **High-precision requirements**: Better accuracy than BDF4

    **Warning**: Reduced stability compared to lower-order BDFs. For very stiff problems,
    use BDF2 or BDF3. For adaptive stepping, use GEAR54 or ESDIRK54.

    References
    ----------
    .. [1] Gear, C. W. (1971). "Numerical Initial Value Problems in Ordinary
           Differential Equations". Prentice-Hall.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems". Springer Series in Computational
           Mathematics, Vol. 14.

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order (local)
        self.n = 5

        #longer history for BDF
        self.history = deque([], maxlen=5)


class BDF6(BDF):
    """Fixed-step 6th order Backward Differentiation Formula (BDF).

    Implicit linear multistep method using the previous six solution points. Not A-stable;
    stability region does not contain the entire left half-plane (stability angle only
    :math:`\\approx 18^\\circ`), severely limiting its use for stiff problems. Uses DIRK3
    startup method for initial steps.

    Characteristics
    ---------------
    * Order: 6
    * Implicit Multistep
    * Fixed timestep only
    * Not A-stable (stability angle approx :math:`18^\\circ`)

    When to Use
    -----------
    * **Very smooth, mildly stiff problems**: Only when stiffness is minimal
    * **High accuracy priority**: When 6th order accuracy justifies poor stability
    * **Specialized applications**: Rarely used in practice

    **Warning**: Very limited stability. Generally not recommended for stiff problems.
    For most applications requiring 6th order accuracy, use explicit methods like RKV65
    on non-stiff problems, or lower-order BDFs with smaller timesteps on stiff problems.

    References
    ----------
    .. [1] Gear, C. W. (1971). "Numerical Initial Value Problems in Ordinary
           Differential Equations". Prentice-Hall.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems". Springer Series in Computational
           Mathematics, Vol. 14.
    .. [3] Curtiss, C. F., & Hirschfelder, J. O. (1952). "Integration of stiff equations".
           Proceedings of the National Academy of Sciences, 38(3), 235-243.

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order (local)
        self.n = 6

        #longer history for BDF
        self.history = deque([], maxlen=6)