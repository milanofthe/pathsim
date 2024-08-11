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

    def __init__(self, initial_value=0, func=lambda x, u, t: u, jac=None, tolerance_lte=1e-6):
        super().__init__(initial_value, func, jac, tolerance_lte)

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
            return True, 0.0, 1.0

        #compute local truncation error slope
        slope = 0.0
        for i, b in enumerate(self.TR):
            slope += self.Ks[i] * b

        #compute and clip truncation error
        truncation_error = np.max(np.clip(abs(dt*slope), 1e-18, None))
        
        #compute error ratio
        error_ratio = self.tolerance_lte / truncation_error
        success = error_ratio >= 1.0

        #compute timestep scale
        timestep_rescale = 0.9 * (error_ratio)**(1/3)        

        return success, truncation_error, timestep_rescale


    def step(self, u, t, dt):
        """
        performs the (explicit) timestep for (t+dt) 
        based on the state and input at (t)
        """
        
        #buffer intermediate slope
        self.Ks[self.stage] = self.func(self.x, u, t)
        
        #error and step size control
        if self.stage < 3:

            #update state at stage
            slope = 0.0
            for i, b in enumerate(self.BT[self.stage]):
                slope += self.Ks[i] * b
            self.x = dt*slope + self.x_0

            self.stage += 1
            return True, 0.0, 1.0
        else: 
            self.stage = 0
            return self.error_controller(dt)