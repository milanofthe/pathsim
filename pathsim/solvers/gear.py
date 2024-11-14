########################################################################################
##
##                             GEARS INTEGRATION METHODS 
##                                 (solvers/gear.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._solver import ImplicitSolver

import numpy as np


# HELPERS ==============================================================================

def compute_bdf_coefficients(order, timesteps):
    """
    Computes the BDF coefficients for a given order and list of timesteps.
    
    For m-th order BDF we have for the n-th timestep:

        sum(alpha_i * x_i; i=n-m,...,n) = h_n * f_n(x_n, t_n)

    INPUTS : 
        order     : (int) order of the integration scheme
        timesteps : (list[float]) timestep buffer (h_{n-j}; j=0,...,order-1)
    """

    #check if valid order
    if order < 1:
        raise RuntimeError("BDF coefficients of order 0 not possible!")

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

class GearBase(ImplicitSolver):
    """
    Base class for GEAR-type integrators that defines the specific methods.

    Numerical integration method based on BDFs (linear multistep methods). 
    Uses n-th and m-th (n-1) order BDFs in two separate stages for adaptive 
    timestep control. The timestep adaption is handled by a buffer for 
    the past timesteps. The adaptive timestep BDF weights are dynamically 
    computed at each timestep.

    NOTE:
        Not to be used directly!!!
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = None
        self.m = None

        #safety factor for error controller (if available)
        self.beta = 0.9

        #bdf solution buffer
        self.B = []

        #gear timestep buffer
        self.T = []

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation 
        self.eval_stages = [1.0]


    def reset(self):
        """"
        Resets integration engine to initial state.
        """

        #clear buffers 
        self.B = []
        self.T = []

        #overwrite state with initial value
        self.x = self.x_0 = self.initial_value

    
    def set(self, x):
        """
        Sets the internal state of the integration engine and buffer 
        for the multistep method. 

        This method is required for event based simulations, and to 
        handle discontinuities in state variables.
        """

        #overwrite internal state with value
        self.x = self.x_0 = x

        #reset stage counter
        self.stage = 0


    def buffer(self, dt):
        """
        Buffer the state and timestep. Dynamically precompute the 
        variable timestep BDF coefficients on the fly for 
        the current timestep.
        """
            
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
        """
        Revert integration engine to previous timestep, this is only relevant 
        for adaptive methods where the simulation timestep 'dt' is rescaled and 
        the engine step is recomputed with the smaller timestep.
        """
        
        #reset internal state to previous state
        self.x = self.x_0

        #remove most recent buffer entry
        self.B.pop(0)
        self.T.pop(0)

        #reset stage counter
        self.stage = 0   


    def error_controller(self, x_m):
        """
        compute scaling factor for adaptive timestep based on absolute and 
        relative local truncation error estimate, also checks if the error 
        tolerance is achieved and returns a success metric.

        INPUTS:
            x_m : (array[float]) lower order solution 
        """

        #compute scaling factors (avoid division by zero)
        scale = self.tolerance_lte_abs + self.tolerance_lte_rel * np.abs(self.x)

        #compute scaled truncation error (element-wise)
        scaled_error = np.abs(self.x - x_m) / scale

        #compute the error norm and clip it
        error_norm = np.clip(float(np.max(scaled_error)), 1e-18, None)

        #determine if the error is acceptable
        success = error_norm <= 1.0

        #compute timestep scale factor using accuracy order of truncation error
        timestep_rescale = self.beta / error_norm ** (1/self.n)

        #clip the rescale factor to a reasonable range
        timestep_rescale = np.clip(timestep_rescale, 0.1, 10.0)

        return success, error_norm, timestep_rescale


    # methods for timestepping ---------------------------------------------------------

    def solve(self, u, t, dt):
        """
        Solves the implicit update equation via anderson acceleration.
        """

        #order of scheme for current step
        n = min(self.n, len(self.B))
        
        #fixed-point function update (faster then sum comprehension)
        g = self.F[n] * dt * self.func(self.x, u, t) 
        for b, k in zip(self.B, self.K[n]):
            g = g + b*k

        #use the jacobian
        if self.jac is not None:

            #compute jacobian
            jac_g = self.F[n] * dt * self.jac(self.x, u, t)

            #anderson acceleration step with local newton
            self.x, err = self.acc.step(self.x, g, jac_g)

        else:
            #anderson acceleration step (pure)
            self.x, err = self.acc.step(self.x, g, None)

        #return the fixed-point residual
        return err


    def step(self, u, t, dt):
        """
        Performs the timestep by buffering the previous state.
        """

        #reset anderson accelerator
        self.acc.reset()

        #early exit if buffer not long enough for two solutions
        if len(self.B) < self.n:
            return True, 0.0, 1.0

        #estimate lower order solution
        x_m = self.F[self.m] * dt * self.func(self.x, u, t) 
        for b, k in zip(self.B, self.K[self.m]):
            x_m = x_m + b*k

        #error control
        return self.error_controller(x_m)





# SOLVERS ==============================================================================

class GEAR32(GearBase):
    """
    Adaptive GEAR integrator with 3-rd order BDF for timestepping 
    and internal 2-nd order BDF for truncation error estimation.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = 3
        self.m = 2


class GEAR43(GearBase):
    """
    Adaptive GEAR integrator with 4-th order BDF for timestepping 
    and internal 3-rd order BDF for truncation error estimation.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = 4
        self.m = 3


class GEAR54(GearBase):
    """
    Adaptive GEAR integrator with 5-th order BDF for timestepping 
    and internal 4-th order BDF for truncation error estimation.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = 5
        self.m = 4





class GEAR52A(GearBase):
    """
    Adaptive order adaptive stepsize GEAR integrator.
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #initial integration order
        self.n = 2

        #minimum and maximum BDF order to select
        self.n_min, self.n_max = 2, 5


    def buffer(self, dt):
        """
        Buffer the state and timestep. Dynamically precompute the 
        variable timestep BDF coefficients on the fly for 
        the current timestep.
        """
            
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


    def error_controller(self, x_m, x_p):
        """
        compute scaling factor for adaptive timestep based on absolute and 
        relative local truncation error estimate, also checks if the error 
        tolerance is achieved and returns a success metric.

        INPUTS:
            x_m : (array[float]) lower order solution estimate
            x_p : (array[float]) higher order solution estimate
        """

        #compute scaling factors (avoid division by zero)
        scale = self.tolerance_lte_abs + self.tolerance_lte_rel * np.abs(self.x)

        #compute scaled truncation error (element-wise)
        scaled_error_m = np.abs(self.x - x_m) / scale
        scaled_error_p = np.abs(self.x - x_p) / scale

        #compute the error norm and clip it
        error_norm_m = np.clip(float(np.max(scaled_error_m)), 1e-18, None)
        error_norm_p = np.clip(float(np.max(scaled_error_p)), 1e-18, None)        

        #decrease the order if smaller order is more accurate
        if error_norm_m < error_norm_p:

            #success metric
            success = error_norm_m <= 1.0

            #compute timestep scale factor using accuracy order of truncation error
            timestep_rescale = self.beta / error_norm_m ** (1/self.n)
            timestep_rescale = np.clip(timestep_rescale, 0.1, 10.0)

            #decrease method order by one
            self.n = max(self.n-1, self.n_min)
    
            return success, error_norm_m, timestep_rescale

        else:
            
            #success metric
            success = error_norm_p <= 1.0

            #compute timestep scale factor using accuracy order of truncation error
            timestep_rescale = self.beta / error_norm_p ** (1/(self.n + 1))
            timestep_rescale = np.clip(timestep_rescale, 0.1, 10.0)

            #increase method order by one
            self.n = min(self.n+1, self.n_max)

            return success, error_norm_p, timestep_rescale


    # methods for timestepping ---------------------------------------------------------

    def step(self, u, t, dt):
        """
        Performs the timestep by buffering the previous state.
        """

        #reset anderson accelerator
        self.acc.reset()

        #early exit if buffer not long enough for two solutions
        if len(self.B) < self.n + 1:
            return True, 0.0, 1.0

        #estimate lower order solution
        x_m = self.F[self.n-1] * dt * self.func(self.x, u, t) 
        for b, k in zip(self.B, self.K[self.n-1]):
            x_m = x_m + b*k

        #estimate higher order solution
        x_p = self.F[self.n+1] * dt * self.func(self.x, u, t) 
        for b, k in zip(self.B, self.K[self.n+1]):
            x_p = x_p + b*k

        #error estimate after last stage
        return self.error_controller(x_m, x_p)