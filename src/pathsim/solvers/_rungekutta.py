########################################################################################
##
##                       BASE CLASS FOR RUNGE-KUTTA INTEGRATORS
##                              (solvers/_rungekutta.py)
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from .._constants import (
    TOLERANCE, 
    SOL_BETA, 
    SOL_SCALE_MIN,
    SOL_SCALE_MAX
    )

from ._solver import ExplicitSolver, ImplicitSolver


# SOLVERS ==============================================================================

class ExplicitRungeKutta(ExplicitSolver):
    """Base class for explicit Runge-Kutta integrators which implements 
    the timestepping at intermediate stages and the error control if 
    the coefficients for the local truncation error estimate are defined.        
    
    Notes
    -----
    This class is not intended to be used directly!!!

    Attributes
    ----------
    n : int 
        order of stepping integration scheme
    m : int
        order of embedded integration scheme for error control
    s : int
        numer of RK stages
    history : deque[numeric]
        internal history of past results
    beta : float
        safety factor for error control
    Ks : dict
        slopes at RK stages
    BT : dict[int: None, list[float]], None
        butcher table
    TR : list[float]
        coefficients for truncation error estimate
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #order of the integration scheme and embedded method (if available)
        self.n = 0
        self.m = 0

        #number of stages in RK scheme
        self.s = 0

        #safety factor for error controller (if available)
        self.beta = SOL_BETA

        #slope coefficients for stages
        self.Ks = {}

        #extended butcher tableau
        self.BT = None

        #coefficients for local truncation error estimate
        self.TR = None


    def error_controller(self, dt):
        """Compute scaling factor for adaptive timestep based on 
        absolute and relative local truncation error estimate, 
        also checks if the error tolerance is achieved and returns 
        a success metric.

        Parameters
        ----------
        dt : float 
            integration timestep

        Returns
        -------
        success : bool
            timestep was successful
        err : float
            truncation error estimate
        scale : float
            timestep rescale from error controller
        """

        #no error estimate or not last stage -> early exit
        if self.TR is None or self.stage < self.s: 
            return True, 0.0, 1.0

        #local truncation error slope (this is faster then 'sum' comprehension)
        slope = 0.0
        for i, b in enumerate(self.TR):
            slope = slope + self.Ks[i] * b

        #compute scaling factors (avoid division by zero)
        scale = self.tolerance_lte_abs + self.tolerance_lte_rel * np.abs(self.x)

        #compute scaled truncation error (element-wise)
        scaled_error = np.abs(dt * slope) / scale

        #compute the error norm and clip it
        error_norm = np.clip(float(np.max(scaled_error)), TOLERANCE, None)

        #determine if the error is acceptable
        success = error_norm <= 1.0

        #compute timestep scale factor using accuracy order of truncation error
        timestep_rescale = self.beta / error_norm ** (1/(min(self.m, self.n) + 1)) 

        #clip the rescale factor to a reasonable range
        timestep_rescale = np.clip(timestep_rescale, SOL_SCALE_MIN, SOL_SCALE_MAX)

        return success, error_norm, timestep_rescale


    def step(self, f, dt):
        """Performs the (explicit) timestep at the intermediate RK stages 
        for (t+dt) based on the state and input at (t)

        Parameters
        ----------
        f : numeric, array[numeric]
            evaluation of function
        dt : float 
            integration timestep

        Returns
        -------
        success : bool
            timestep was successful
        err : float
            truncation error estimate
        scale : float
            timestep rescale from error controller        
        """

        #buffer intermediate slope
        self.Ks[self.stage] = f

        #get current state from history
        x_0 = self.history[0]

        #compute slope at stage, faster then 'sum' comprehension
        slope = 0.0
        for i, b in enumerate(self.BT[self.stage]):
            slope = slope + self.Ks[i] * b
        self.x = x_0 + dt * slope

        #increment stage counter
        self.stage += 1

        #compute truncation error estimate
        return self.error_controller(dt)


class DiagonallyImplicitRungeKutta(ImplicitSolver):
    """Base class for diagonally implicit Runge-Kutta (DIRK) integrators 
    which implements the timestepping at intermediate stages, involving
    the numerical solution of the implicit update equation and the 
    error control if the coefficients for the local truncation error 
    estimate are defined.

    Extensions and checks to also handle explicit first stages (ESDIRK) 
    and additional final evaluation coefficients (not stiffly accurate)
    
    Notes
    -----
    This class is not intended to be used directly!!!

    Attributes
    ----------
    n : int 
        order of stepping integration scheme
    m : int
        order of embedded integration scheme for error control
    s : int
        numer of RK stages
    beta : float
        safety factor for error control
    Ks : dict
        slopes at RK stages
    BT : dict[int: None, list[float]], None
        butcher table
    A : list[float], None
        coefficients for final solution evaluation
    TR : list[float]
        coefficients for truncation error estimate

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #order of the integration scheme and embedded method (if available)
        self.n = 0
        self.m = 0

        #number of stages in RK scheme
        self.s = 0

        #safety factor for error controller (if available)
        self.beta = SOL_BETA

        #slope coefficients for stages
        self.Ks = {}

        #extended butcher tableau
        self.BT = None

        #final evaluation (if not stiffly accurate)
        self.A = None

        #coefficients for local truncation error estimate
        self.TR = None


    def error_controller(self, dt):
        """Compute scaling factor for adaptive timestep based on 
        absolute and relative local truncation error estimate, 
        also checks if the error tolerance is achieved and returns 
        a success metric.

        Parameters
        ----------
        dt : float 
            integration timestep

        Returns
        -------
        success : bool
            timestep was successful
        err : float
            truncation error estimate
        scale : float
            timestep rescale from error controller
        """

        #no error estimate or not last stage -> early exit
        if self.TR is None or self.stage < self.s: 
            return True, 0.0, 1.0

        #local truncation error slope (this is faster then 'sum' comprehension)
        slope = 0.0
        for i, b in enumerate(self.TR):
            slope = slope + self.Ks[i] * b

        #compute scaling factors (avoid division by zero)
        scale = self.tolerance_lte_abs + self.tolerance_lte_rel * np.abs(self.x)

        #compute scaled truncation error (element-wise)
        scaled_error = np.abs(dt * slope) / scale

        #compute the error norm and clip it#compute the error norm and clip it
        error_norm = np.clip(float(np.max(scaled_error)), TOLERANCE, None)

        #determine if the error is acceptable
        success = error_norm <= 1.0

        #compute timestep scale factor using accuracy order of truncation error
        timestep_rescale = self.beta / error_norm ** (1/(min(self.m, self.n) + 1)) 

        #clip the rescale factor to a reasonable range
        timestep_rescale = np.clip(timestep_rescale, SOL_SCALE_MIN, SOL_SCALE_MAX)

        return success, error_norm, timestep_rescale


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

        #first stage is explicit -> ESDIRK -> early exit
        if self.stage == 0 and self.BT[self.stage] is None:
            return 0.0
            
        #update timestep weighted slope 
        self.Ks[self.stage] = f

        #get past state from history
        x_0 = self.history[0]

        #compute slope (this is faster then 'sum' comprehension)
        slope = 0.0
        for i, a in enumerate(self.BT[self.stage]):
            slope = slope + self.Ks[i] * a

        #use the jacobian
        if J is not None:

            #most recent butcher coefficient
            b = self.BT[self.stage][self.stage]

            #optimizer step with block local jacobian
            self.x, err = self.opt.step(self.x, x_0 + dt * slope, dt * b * J)

        else:
            #optimizer step (pure)
            self.x, err = self.opt.step(self.x, x_0 + dt * slope, None)

        #return the fixed-point residual
        return err


    def step(self, f, dt):
        """performs the (explicit) timestep at the intermediate RK stages 
        for (t+dt) based on the state and input at (t)

        Parameters
        ----------
        f : array_like
            evaluation of function
        dt : float 
            integration timestep

        Returns
        -------
        success : bool
            timestep was successful
        err : float
            truncation error estimate
        scale : float
            timestep rescale from error controller
        """

        #first stage is explicit -> ESDIRK
        if self.stage == 0 and self.BT[self.stage] is None:
            self.Ks[self.stage] = f

        #increment stage counter
        self.stage += 1

        #compute final output if not stiffly accurate
        if self.A is not None and self.stage == self.s:

            #get past state from history
            x_0 = self.history[0]

            #compute slope (this is faster then 'sum' comprehension)
            slope = 0.0
            for i, a in enumerate(self.A):
                slope = slope + self.Ks[i] * a
            self.x = x_0 + dt * slope    

        #compute truncation error estimate in final stage
        return self.error_controller(dt)