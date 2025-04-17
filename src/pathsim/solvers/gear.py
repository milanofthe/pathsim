########################################################################################
##
##                            GEAR-type INTEGRATION METHODS 
##                                 (solvers/gear.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ._solver import ImplicitSolver

from .._constants import (
    TOLERANCE, 
    SOL_BETA, 
    SOL_SCALE_MIN,
    SOL_SCALE_MAX
    )


# HELPERS ==============================================================================

def compute_bdf_coefficients(order, timesteps):
    """Computes the coefficients for backward differentiation formulas for a given order.
    The timesteps can be specified for variable timestep BDF methods. 

    For m-th order BDF we have for the n-th timestep:
        sum(alpha_i * x_i; i=n-m,...,n) = h_n * f_n(x_n, t_n)
    or 
        x_n = beta * h_n * f_n(x_n, t_n) - sum(alpha_j * x_{n-1-j}; j=0,...,order-1)

    Parameters
    ----------
    order : int
        order of the integration scheme
    timesteps : list[float]
        timestep buffer (h_{n-j}; j=0,...,order-1)
    
    Returns
    ------- 
    beta : float
        weight for function
    alpha : array[float]
        weights for previous solutions
    """

    #check if valid order
    if order < 1:
        raise RuntimeError(f"BDF coefficients of order '{order}' not possible!")

    #quit early for no buffer (euler backward)
    if len(timesteps) < 2:
        return 1.0, [1.0]

    # Compute timestep ratios rho_j = h_{n-j} / h_n
    rho = np.array(timesteps[1:])/timesteps[0]

    # Compute normalized time differences theta_j
    theta = -np.ones(order+1)
    theta[0] = 0
    for j in range(2, order+1):
        theta[j] -= sum(rho[:j-1])

    # Set up the linear system (p + 1 equations)
    A = np.zeros((order+1, order+1))
    b = np.zeros(order+1)
    b[1] = 1 
    for m in range(order+1):
        A[m, :] = theta ** m 

    # Solve the linear system A * alpha = b
    alphas = np.linalg.solve(A, b)

    #return function and buffer weights
    return 1/alphas[0], -alphas[1:]/alphas[0]


# BASE GEAR SOLVER =====================================================================

class GEAR(ImplicitSolver):
    """Base class for GEAR-type integrators that defines the universal methods.

    Numerical integration method based on BDFs (linear multistep methods). 
    Uses n-th order BDF for timestepping and (n-1)-th order BDF coefficients 
    to estimate a lower ordersolutuin for error control. 

    The adaptive timestep BDF coefficients are dynamically computed at the 
    beginning of each timestep from the buffered previous timsteps.

    Notes
    -----
    Not to be used directly!!!
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = None
        self.m = None

        #safety factor for error controller (if available)
        self.beta = SOL_BETA

        #bdf solution buffer
        self.B = []

        #gear timestep buffer
        self.T = []

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation 
        self.eval_stages = [1.0]


    def reset(self):
        """"Resets integration engine to initial state."""

        #clear buffers 
        self.B = []
        self.T = []

        #overwrite state with initial value
        self.x = self.x_0 = self.initial_value


    def buffer(self, dt):
        """Buffer the state and timestep. Dynamically precompute 
        the variable timestep BDF coefficients on the fly for the 
        current timestep.
        
        Parameters
        ----------
        dt : float
            integration timestep
        """

        #reset optimizer
        self.opt.reset()
            
        #buffer state directly
        self.x_0 = self.x

        #add to buffers
        self.B.insert(0, self.x)
        self.T.insert(0, dt)

        #truncate buffers if too long
        if len(self.B) > self.n:
            self.B.pop()
            self.T.pop()

        #precompute coefficients here, where buffers are available
        self.F, self.K = {}, {}
        for n in range(1, len(self.T)+1):
            self.F[n], self.K[n] = compute_bdf_coefficients(n, self.T)


    # methods for adaptive timestep solvers --------------------------------------------

    def revert(self):
        """Revert integration engine to previous timestep, this is only 
        relevant for adaptive methods where the simulation timestep 'dt' 
        is rescaled and the engine step is recomputed with the smaller 
        timestep.
        """
        
        #reset internal state to previous state
        self.x = self.x_0

        #remove most recent buffer entry
        self.B.pop(0)
        self.T.pop(0)


    def error_controller(self, tr):
        """Compute scaling factor for adaptive timestep based on absolute and 
        relative tolerances for local truncation error. 

        Checks if the error tolerance is achieved and returns a success metric.
        
        Parameters
        ----------
        tr : array[float]
            truncation error estimate 

        Returns 
        -------
        success : bool
            True if the timestep was successful
        error : float
            estimated error of the internal error controller
        scale : float
            estimated timestep rescale factor for error control
        """

        #compute scaling factors (avoid division by zero)
        scale = self.tolerance_lte_abs + self.tolerance_lte_rel * np.abs(self.x)

        #compute scaled truncation error (element-wise)
        scaled_error = np.abs(tr) / scale

        #compute the error norm and clip it
        error_norm = np.clip(float(np.max(scaled_error)), TOLERANCE, None)

        #determine if the error is acceptable
        success = error_norm <= 1.0

        #compute timestep scale factor using accuracy order of truncation error
        timestep_rescale = self.beta / error_norm ** (1/self.n)

        #clip the rescale factor to a reasonable range
        timestep_rescale = np.clip(timestep_rescale, SOL_SCALE_MIN, SOL_SCALE_MAX)

        return success, error_norm, timestep_rescale


    # methods for timestepping ---------------------------------------------------------

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

        #order of scheme for current step
        n = min(self.n, len(self.B))
        
        #fixed-point function update (faster then sum comprehension)
        g = self.F[n]*dt*f
        for b, k in zip(self.B, self.K[n]):
            g = g + b*k

        #use the jacobian
        if J is not None:

            #optimizer step with block local jacobian
            self.x, err = self.opt.step(self.x, g, self.F[n]*dt*J)

        else:
            #optimizer step (pure)
            self.x, err = self.opt.step(self.x, g, None)

        #return the fixed-point residual
        return err


    def step(self, f, dt):
        """Finalizes the timestep by resetting the solver for the implicit 
        update equation and computing the lower order estimate of the 
        solution for error control.

        Parameters
        ----------
        f : array_like
            evaluation of function
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

        #early exit if buffer not long enough for two solutions
        if len(self.B) < self.n:
            return True, 0.0, 1.0

        #estimate truncation error from lower order solution
        tr = self.x - self.F[self.m]*dt*f
        for b, k in zip(self.B, self.K[self.m]):
            tr = tr - b*k

        #error control
        return self.error_controller(tr)


# SOLVERS ==============================================================================

class GEAR21(GEAR):
    """Adaptive-step GEAR integrator using 2nd order BDF for timestepping
    and 1st order BDF (Backward Euler) for truncation error estimation.

    Suitable for moderately stiff problems where variable timestepping is beneficial.

    Characteristics:
        * Stepping Order: 2 (max)
        * Error Estimation Order: 1
        * Implicit Variable-Step Multistep
        * Adaptive timestep
        * A-stable (based on BDF2)
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = 2
        self.m = 1


class GEAR32(GEAR):
    """Adaptive-step GEAR integrator using 3rd order BDF for timestepping
    and 2nd order BDF for truncation error estimation.

    Suitable for stiff problems requiring higher accuracy than GEAR21.

    Characteristics:
        * Stepping Order: 3 (max)
        * Error Estimation Order: 2
        * Implicit Variable-Step Multistep
        * Adaptive timestep
        * A(alpha)-stable (based on BDF3)
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = 3
        self.m = 2


class GEAR43(GEAR):
    """Adaptive-step GEAR integrator using 4th order BDF for timestepping
    and 3rd order BDF for truncation error estimation.

    Suitable for stiff problems requiring good accuracy.

    Characteristics:
        * Stepping Order: 4 (max)
        * Error Estimation Order: 3
        * Implicit Variable-Step Multistep
        * Adaptive timestep
        * A(alpha)-stable (based on BDF4)
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = 4
        self.m = 3


class GEAR54(GEAR):
    """Adaptive-step GEAR integrator using 5th order BDF for timestepping
    and 4th order BDF for truncation error estimation.

    Suitable for stiff problems requiring high accuracy, but stability region
    is smaller than lower-order GEAR methods.

    Characteristics:
        * Stepping Order: 5 (max)
        * Error Estimation Order: 4
        * Implicit Variable-Step Multistep
        * Adaptive timestep
        * A(alpha)-stable (based on BDF5)
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = 5
        self.m = 4


class GEAR52A(GEAR):
    """Adaptive-order, adaptive-stepsize GEAR integrator (Variable-Step Variable-Order BDF).

    This method dynamically adjusts the BDF order used for timestepping (between 2 and 5)
    based on error estimates from lower and higher order predictors. It aims to optimize
    step size by using higher orders for smooth regions and lower, more stable orders
    for stiff or rapidly changing regions.

    Error estimation compares the current order solution with predictions from
    order n-1 and n+1 formulas.

    Characteristics:
        * Stepping Order: Variable (2 to 5)
        * Error Estimation Orders: n-1 and n+1 (relative to current n)
        * Implicit Variable-Step, Variable-Order Multistep
        * Adaptive timestep and order
        * Stability varies with the currently selected order (A-stable or A(alpha)-stable)
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #initial integration order
        self.n = 2

        #minimum and maximum BDF order to select
        self.n_min, self.n_max = 2, 5


    def buffer(self, dt):
        """
        Buffer the state and timestep. Dynamically precompute the variable 
        timestep BDF coefficients on the fly for the current timestep.

        Parameters
        ----------
        dt : float
            evaluation time

        """
            
        #reset optimizer
        self.opt.reset()

        #buffer state directly
        self.x_0 = self.x

        #add to buffers
        self.B.insert(0, self.x)
        self.T.insert(0, dt)

        #truncate buffers if too long
        if len(self.B) > self.n_max + 1:
            self.B.pop()
            self.T.pop()

        #precompute coefficients here, where buffers are available
        self.F, self.K = {}, {}
        for n in range(1, len(self.T)+1):
            self.F[n], self.K[n] = compute_bdf_coefficients(n, self.T)


    def error_controller(self, tr_m, tr_p):
        """Compute scaling factor for adaptive timestep based on absolute and 
        relative tolerances of the local truncation error estimate obtained from 
        esimated lower and higher order solution. 

        Checks if the error tolerance is achieved and returns a success metric.

        Adapts the stepping order such that the normalized error is minimized and 
        larger steps can be taken by the integrator.

        Parameters
        ----------
        tr_m : array[float]
            lower order truncation error estimate
        tr_p : array[float]
            higher order truncation error estimate

        Returns 
        -------
        success : bool
            True if the timestep was successful
        error : float
            estimated error of the internal error controller
        scale : float
            estimated timestep rescale factor for error control
        """

        #compute scaling factors (avoid division by zero)
        scale = self.tolerance_lte_abs + self.tolerance_lte_rel * np.abs(self.x)

        #compute scaled truncation error (element-wise)
        scaled_error_m = np.abs(tr_m) / scale
        scaled_error_p = np.abs(tr_p) / scale

        #compute the error norm and clip it
        error_norm_m = np.clip(float(np.max(scaled_error_m)), TOLERANCE, None)
        error_norm_p = np.clip(float(np.max(scaled_error_p)), TOLERANCE, None)      

        #success metric (use lower order estimate)
        success = error_norm_m <= 1.0

        #compute timestep scale factor using accuracy order of truncation error
        timestep_rescale = self.beta / error_norm_m ** (1/self.n)  

        #clip the rescale factor to a reasonable range
        timestep_rescale = np.clip(timestep_rescale, SOL_SCALE_MIN, SOL_SCALE_MAX)

        #decrease the order if smaller order is more accurate (stability)
        if error_norm_m < error_norm_p:
            self.n = max(self.n-1, self.n_min)
        
        #increase the order if larger order is more accurate (accuracy -> larger steps)
        else:
            self.n = min(self.n+1, self.n_max)

        return success, error_norm_p, timestep_rescale


    # methods for timestepping ---------------------------------------------------------

    def step(self, f, dt):
        """Finalizes the timestep by resetting the solver for the implicit 
        update equation and computing the lower and higher order estimate 
        of the solution. 

        Then calls the error controller.

        Parameters
        ----------
        f : array_like
            evaluation of function
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

        #early exit if buffer not long enough for two solutions
        if len(self.B) < self.n + 1:
            return True, 0.0, 1.0

        #lower and higher order
        n_m, n_p = self.n-1, self.n+1 

        #estimate truncation error from lower order solution
        tr_m = self.x - self.F[n_m]*dt*f
        for b, k in zip(self.B, self.K[n_m]):
            tr_m = tr_m - b*k

        #estimate truncation error from higher order solution
        tr_p = self.x - self.F[n_p]*dt*f
        for b, k in zip(self.B, self.K[n_p]):
            tr_p = tr_p - b*k

        return self.error_controller(tr_m, tr_p)
