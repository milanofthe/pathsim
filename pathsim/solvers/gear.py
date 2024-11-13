########################################################################################
##
##                             GEARS INTEGRATION METHODS 
##                                (solvers/gear32.py)
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

    #aux value
    order_p1 = order + 1

    #quit early for no buffer (euler backward)
    if len(timesteps) < 2:
        return 1.0, [1.0]

    # Compute timestep ratios rho_j = h_{n-j} / h_n
    rho = np.array(timesteps[1:]) / timesteps[0]

    # Compute normalized time differences theta_j
    theta = -np.ones(order_p1)
    theta[0] = 0
    for j in range(2, order_p1):
        theta[j] -= sum(rho[:j - 1])

    # Set up the linear system (p + 1 equations)
    A = np.zeros((order_p1, order_p1))
    b = np.zeros(order_p1)
    b[1] = 1 
    for m in range(order_p1):
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

        #intermediate evaluation times (two multistep methods)
        self.eval_stages = [1.0, 1.0]


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


    # def error_controller(self, dt):
    #     """
    #     compute scaling factor for adaptive timestep based on absolute and 
    #     relative local truncation error estimate, also checks if the error 
    #     tolerance is achieved and returns a success metric.

    #     INPUTS:
    #         dt : (float) integration timestep
    #     """

    #     #early exit if buffer not long enough for two solutions
    #     if len(self.B) < self.n:
    #         return True, 0.0, 0.0, 1.0

    #     #compute local truncation error as difference of two internal schemes
    #     tr = self.x - self.x_m

    #     #compute and clip truncation error, error ratio abs
    #     truncation_error_abs = float(np.max(abs(tr)))
    #     error_ratio_abs = self.tolerance_lte_abs / np.clip(truncation_error_abs, 1e-18, None)

    #     #compute and clip truncation error, error ratio rel
    #     if np.any(self.x == 0.0): 
    #         truncation_error_rel = 1.0
    #         error_ratio_rel = 0.0
    #     else:
    #         truncation_error_rel = float(np.max(abs(tr/self.x)))
    #         error_ratio_rel = self.tolerance_lte_rel / np.clip(truncation_error_rel, 1e-18, None)
        
    #     #compute error ratio and success check
    #     error_ratio = max(error_ratio_abs, error_ratio_rel)
    #     success = error_ratio >= 1.0

    #     #compute timestep scale factor using accuracy order of truncation error
    #     timestep_rescale = self.beta * error_ratio**(1/(self.m + 1))

    #     #clip the rescale factor to a reasonable range
    #     timestep_rescale = np.clip(timestep_rescale, 0.1, 10.0)

    #     return success, truncation_error_abs, truncation_error_rel, timestep_rescale




    def error_controller(self, dt):
        """
        compute scaling factor for adaptive timestep based on absolute and 
        relative local truncation error estimate, also checks if the error 
        tolerance is achieved and returns a success metric.

        INPUTS:
            dt : (float) integration timestep
        """

        #early exit if buffer not long enough for two solutions
        if len(self.B) < self.n:
            return True, 0.0, 1.0

        #compute scaling factors (avoid division by zero)
        scale = self.tolerance_lte_abs + self.tolerance_lte_rel * np.abs(self.x)

        #compute scaled truncation error (element-wise)
        scaled_error = np.abs(self.x - self.x_m) / scale

        #compute the error norm and clip it
        error_norm = np.clip(float(np.max(scaled_error)), 1e-18, None)

        #determine if the error is acceptable
        success = error_norm <= 1.0

        #compute timestep scale factor using accuracy order of truncation error
        timestep_rescale = self.beta / error_norm ** (1/(self.m + 1))

        #clip the rescale factor to a reasonable range
        timestep_rescale = np.clip(timestep_rescale, 0.1, 10.0)

        return success, error_norm, timestep_rescale


    # methods for timestepping ---------------------------------------------------------

    def solve(self, u, t, dt):
        """
        Solves the implicit update equation via anderson acceleration.
        """

        #order of scheme for current step
        n = min(self.n if self.stage == 1 else self.m, len(self.B))
        
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

        #add to buffer
        if self.stage == 1:

            #reset stage counter
            self.stage = 0

            #error estimate after last stage
            return self.error_controller(dt)

        else:

            #save lower order solution
            self.x_m = self.x

            #increment stage counter
            self.stage = 1

            #no error estimate after first stage
            return True, 0.0, 1.0


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

        #safety factor for error controller (if available)
        self.beta = 0.9

        #bdf solution buffer
        self.B = []

        #gear timestep buffer
        self.T = []

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times (two multistep methods)
        self.eval_stages = [1.0, 1.0]


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

        #safety factor for error controller (if available)
        self.beta = 0.9

        #bdf solution buffer
        self.B = []

        #gear timestep buffer
        self.T = []

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times (two multistep methods)
        self.eval_stages = [1.0, 1.0]


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

        #safety factor for error controller (if available)
        self.beta = 0.9

        #bdf solution buffer
        self.B = []

        #gear timestep buffer
        self.T = []

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times (two multistep methods)
        self.eval_stages = [1.0, 1.0]

