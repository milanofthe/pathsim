########################################################################################
##
##                       BASE CLASS FOR RUNGE-KUTTA INTEGRATORS
##                              (solvers/_rungekutta.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ._solver import ExplicitSolver, ImplicitSolver


# SOLVERS ==============================================================================

class ExplicitRungeKutta(ExplicitSolver):
    """
    Base class for explicit Runge-Kutta integrators which implements 
    the timestepping at intermediate stages and the error control if 
    the coefficients for the local truncation error estimate are defined.        
    
    NOTE:
        This class is not intended to be used directly!!!
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = None

        #slope coefficients for stages
        self.Ks = {}

        #extended butcher tableau
        self.BT = None

        #coefficients for local truncation error estimate
        self.TR = None


    def error_controller(self, dt):
        """
        compute scaling factor for adaptive timestep based on 
        absolute and relative local truncation error estimate, 
        also checks if the error tolerance is achieved and returns 
        a success metric.

        INPUTS:
            dt : (float) integration timestep
        """

        #early exit of not enough slopes or no error estimate at all
        if self.TR is None or len(self.Ks) < len(self.TR): 
            return True, 0.0, 0.0, 1.0

        #compute local truncation error
        tr = dt * sum(k*b for k, b in zip(self.Ks.values(), self.TR))

        #compute and clip truncation error, error ratio abs
        truncation_error_abs = float(np.max(np.clip(abs(tr), 1e-18, None)))
        error_ratio_abs = self.tolerance_lte_abs / truncation_error_abs

        #compute and clip truncation error, error ratio rel
        if np.any(self.x == 0.0): 
            truncation_error_rel = 1.0
            error_ratio_rel = 0.0
        else:
            truncation_error_rel = float(np.max(np.clip(abs(tr/self.x), 1e-18, None)))
            error_ratio_rel = self.tolerance_lte_rel / truncation_error_rel
        
        #compute error ratio and success check
        error_ratio = max(error_ratio_abs, error_ratio_rel)
        success = error_ratio >= 1.0

        #compute timestep scale
        timestep_rescale = 0.9 * (error_ratio)**(1/self.n)        

        return success, truncation_error_abs, truncation_error_rel, timestep_rescale


    def step(self, u, t, dt):
        """
        performs the (explicit) timestep at the intermediate RK stages 
        for (t+dt) based on the state and input at (t)

        INPUTS:
            u  : (float) non-autonomous external component for integration
            t  : (float) evaluation time for right-hand-side function
            dt : (float) integration timestep
        """

        #buffer intermediate slope
        self.Ks[self.stage] = self.func(self.x, u, t)
        
        #compute slope and update state at stage
        self.x = dt * sum(k*b for k, b in zip(self.Ks.values(), self.BT[self.stage])) + self.x_0

        #error and step size control
        if self.stage < self.s - 1:

            #increment stage counter
            self.stage += 1

            #no error control for intermediate stages
            return True, 0.0, 0.0, 1.0
        
        else: 

            #reset stage counter
            self.stage = 0

            #compute truncation error estimate in final stage
            return self.error_controller(dt)



class DiagonallyImplicitRungeKutta(ImplicitSolver):
    """
    Base class for diagonally implicit Runge-Kutta (DIRK) integrators 
    which implements the timestepping at intermediate stages and the 
    error control if the coefficients for the local truncation error 
    estimate are defined.

    Extensions and checks to also handle explicit first stages (ESDIRK) 
    and additional final evaluation coefficients (not stiffly accurate)
    
    NOTE:
        This class is not intended to be used directly!!!
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = None

        #slope coefficients for stages
        self.Ks = {}

        #extended butcher tableau
        self.BT = None

        #final evaluation (if not stiffly accurate)
        self.A = None

        #coefficients for local truncation error estimate
        self.TR = None


    def error_controller(self, dt):
        """
        compute scaling factor for adaptive timestep based on 
        absolute and relative local truncation error estimate, 
        also checks if the error tolerance is achieved and returns 
        a success metric.

        INPUTS:
            dt : (float) integration timestep
        """

        if self.TR is None or len(self.Ks) < len(self.TR): 
            return True, 0.0, 0.0, 1.0

        #compute local truncation error
        tr = dt * sum(k*b for k, b in zip(self.Ks.values(), self.TR))

        #compute and clip truncation error, error ratio abs
        truncation_error_abs = float(np.max(np.clip(abs(tr), 1e-18, None)))
        error_ratio_abs = self.tolerance_lte_abs / truncation_error_abs

        #compute and clip truncation error, error ratio rel
        if np.any(self.x == 0.0): 
            truncation_error_rel = 1.0
            error_ratio_rel = 0.0
        else:
            truncation_error_rel = float(np.max(np.clip(abs(tr/self.x), 1e-18, None)))
            error_ratio_rel = self.tolerance_lte_rel / truncation_error_rel
        
        #compute error ratio and success check
        error_ratio = max(error_ratio_abs, error_ratio_rel)
        success = error_ratio >= 1.0

        #compute timestep scale
        timestep_rescale = 0.9 * (error_ratio)**(1/self.n)

        return success, truncation_error_abs, truncation_error_rel, timestep_rescale


    def solve(self, u, t, dt):
        """
        Solves the implicit update equation via anderson acceleration.

        INPUTS:
            u  : (float) non-autonomous external component for integration
            t  : (float) evaluation time for right-hand-side function
            dt : (float) integration timestep
        """

        #first stage is explicit -> ESDIRK -> early exit
        if self.stage == 0 and self.BT[self.stage] is None:
            return 0.0
            
        #update timestep weighted slope 
        self.Ks[self.stage] = self.func(self.x, u, t)

        #compute slope and update fixed-point equation
        slope = sum(k*b for k, b in zip(self.Ks.values(), self.BT[self.stage]))

        #use the jacobian
        if self.jac is not None:

            #most recent butcher coefficient
            b = self.BT[self.stage][self.stage]

            #compute jacobian of fixed-point equation
            jac_g = dt * b * self.jac(self.x, u, t)

            #anderson acceleration step with local newton
            self.x, err = self.acc.step(self.x, dt*slope + self.x_0, jac_g)

        else:
            #anderson acceleration step (pure)
            self.x, err = self.acc.step(self.x, dt*slope + self.x_0, None)

        #return the fixed-point residual
        return err


    def step(self, u, t, dt):
        """
        performs the (explicit) timestep at the intermediate RK stages 
        for (t+dt) based on the state and input at (t)

        INPUTS:
            u  : (float) non-autonomous external component for integration
            t  : (float) evaluation time for right-hand-side function
            dt : (float) integration timestep
        """

        #first stage is explicit -> ESDIRK
        if self.stage == 0 and self.BT[self.stage] is None:
            self.Ks[self.stage] = self.func(self.x, u, t)

        #restart anderson accelerator 
        self.acc.reset()

        #error and step size control
        if self.stage < self.s - 1:

            #increment stage counter
            self.stage += 1

            #no error estimate for intermediate stages
            return True, 0.0, 0.0, 1.0

        else: 

            #compute final output if not stiffly accurate
            if self.A is not None:
                self.x = dt * sum(k*a for k, a in zip(self.Ks.values(), self.A)) + self.x_0
            
            #reset stage counter
            self.stage = 0

            #compute truncation error estimate in final stage
            return self.error_controller(dt)