########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                (solvers/rkck54.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ._solver import ExplicitSolver


# SOLVERS ==============================================================================

class RKCK54(ExplicitSolver):
    """
    6-stage 5-th order with embedded 4-th order Runge-Kutta method from Cash and Karp 
    with 5-th order truncation error estimate for the 4-th order solution that can be 
    used to adaptively control the timestep. The 5-th order method is used for 
    timestepping (local extrapolation) and the difference to the 5-th order solution 
    is used as an estimate for the local truncation error of the 4-th order solution.
    
    This is the fixed order Cash-Karp scheme without early quitting.

    The method balances the accuracy of the 5-th and 4-th order solution and 
    has enhanced stability properties compared to Fehlberg or Dormand-Prince 
    methods. This makes it suitable for slightly stiff problems.
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
        self.eval_stages = [0.0, 1/5, 3/10, 3/5, 1, 7/8]

        #extended butcher table 
        self.BT = {0:[       1/5],
                   1:[      3/40,    9/40],
                   2:[      3/10,   -9/10,       6/5],
                   3:[    -11/54,     5/2,    -70/27,        35/27],
                   4:[1631/55296, 175/512, 575/13824, 44275/110592, 253/4096],
                   5:[    37/378,       0,   250/621,      125/594,        0, 512/1771]}

        #coefficients for local truncation error estimate
        self.TR = [-277/64512, 0, 6925/370944, -6925/202752, -277/14336, 277/7084]


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
        
        #update state at stage
        slope = 0.0
        for i, b in enumerate(self.BT[self.stage]):
            slope += self.Ks[i] * b
        self.x = dt * slope + self.x_0

        #error and step size control
        if self.stage < 5:
            self.stage += 1
            return True, 0.0, 1.0
        else: 
            self.stage = 0
            return self.error_controller(dt)