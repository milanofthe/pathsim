########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                (solvers/rkdp54.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ._solver import ExplicitSolver


# SOLVERS ==============================================================================

class RKDP54(ExplicitSolver):
    """
    Dormand–Prince method with seven Runge-Kutta stages is 5-th order 
    accurate with an embedded 4-th order method. The 5-th order method 
    is used for timestepping (local extrapolation) and the difference 
    to the 5-th order solution is used as an estimate for the local 
    truncation error of the 4-th order solaution.
    
    Wikipedia:
        As of 2023, Dormand–Prince is the default method 
        in the 'ode45' solver for MATLAB

    Great choice for all kinds of problems that require high accuracy 
    and where the adaptive timestepping doesnt cause problems.
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
        self.eval_stages = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
        
        #extended butcher table
        self.BT = {0:[       1/5],
                   1:[      3/40,        9/40],
                   2:[     44/45,      -56/15,       32/9], 
                   3:[19372/6561, -25360/2187, 64448/6561, -212/729],
                   4:[ 9017/3168,     -355/33, 46732/5247,   49/176, -5103/18656],
                   5:[    35/384,           0,   500/1113,  125/192,  -2187/6784, 11/84]}

        #coefficients for local truncation error estimate
        self.TR = [71/57600, 0, - 71/16695, 71/1920, -17253/339200, 22/525, -1/40]


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
        timestep_rescale = 0.9 * (error_ratio)**(1/5)        

        return success, truncation_error, timestep_rescale


    def step(self, u, t, dt):
        """
        performs the (explicit) timestep for (t+dt) 
        based on the state and input at (t)
        """

        #buffer intermediate slope
        self.Ks[self.stage] = self.func(self.x, u, t)

        #error and step size control
        if self.stage < 6:
            
            #update state at stage
            slope = 0.0
            for i, b in enumerate(self.BT[self.stage]):
                slope += self.Ks[i] * b
            self.x = dt * slope + self.x_0

            #increment stage counter
            self.stage += 1
            return True, 0.0, 1.0
        else: 
            self.stage = 0
            return self.error_controller(dt)