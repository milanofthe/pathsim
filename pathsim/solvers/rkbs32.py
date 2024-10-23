########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                (solvers/rkbs32.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ._solver import ExplicitSolver


# SOLVERS ==============================================================================

class RKBS32(ExplicitSolver):
    """
    The Bogacki–Shampine method is a Runge–Kutta method of order three with four stages.
    It has an embedded second-order method which can be used to implement adaptive 
    step size. The Bogacki–Shampine method is implemented in the 'ode3' for fixed 
    step solver and 'ode23' for a variable step solver function in MATLAB.

    This is the adaptive variant. It is a good choice of low accuracy is acceptable.
    """

    def __init__(self, 
                 initial_value=0, 
                 func=lambda x, u, t: u, 
                 jac=None, 
                 tolerance_lte_abs=1e-6, 
                 tolerance_lte_rel=1e-3):
        super().__init__(initial_value, 
                         func, 
                         jac, 
                         tolerance_lte_abs, 
                         tolerance_lte_rel)

        #counter for runge kutta stages
        self.stage = 0

        #flag adaptive timestep solver
        self.is_adaptive = True

        #slope coefficients for stages
        self.Ks = {}

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/2, 3/4, 1.0]
        
        #extended butcher table
        self.BT = {0:[1/2],
                   1:[0.0 , 3/4],
                   2:[2/9 , 1/3, 4/9]}

        #coefficients for truncation error estimate
        self.TR = [-5/72, 1/12, 1/9, -1/8]


    def error_controller(self, dt):
        """
        compute scaling factor for adaptive timestep 
        based on local truncation error estimate and returns both
        """
        if len(self.Ks)<len(self.TR): 
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
        timestep_rescale = 0.9 * (error_ratio)**(1/3)        

        return success, truncation_error_abs, truncation_error_rel, timestep_rescale


    def step(self, u, t, dt):
        """
        performs the (explicit) timestep for (t+dt) 
        based on the state and input at (t)
        """
        
        #buffer intermediate slope
        self.Ks[self.stage] = self.func(self.x, u, t)
        
        #error and step size control
        if self.stage < 3:

            #compute slope and update state at stage
            self.x = dt * sum(k*b for k, b in zip(self.Ks.values(), self.BT[self.stage])) + self.x_0

            self.stage += 1
            return True, 0.0, 0.0, 1.0
        else: 
            self.stage = 0
            return self.error_controller(dt)