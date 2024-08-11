########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                 (solvers/rkf78.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ._solver import ExplicitSolver


# SOLVERS ==============================================================================

class RKF78(ExplicitSolver):
    """
    13-stage 7-th order embedded Runge-Kutta-Fehlberg method 
    with 8-th order truncation error estimate that can be used to 
    adaptively control the timestep. 

    This solver is a great choice if extremely high accuracy is required. 
    It is also almost symplectic and therefore quite suitable for 
    conservation systems such as celestial dynamics, etc.
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
        self.eval_stages = [0, 2/27, 1/9, 1/6, 5/12, 1/2, 5/6, 1/6, 2/3, 1/3, 1, 0, 1]

        #extended butcher table 
        self.BT = {0: [      2/27],
                   1: [      1/36, 1/12],
                   2: [      1/24,    0,    1/8],
                   3: [      5/12,    0, -25/16,    25/16],
                   4: [      1/20,    0,      0,      1/4,       1/5],
                   5: [   -25/108,    0,      0,  125/108,    -65/27,  125/54],
                   6: [    31/300,    0,      0,        0,    61/225,    -2/9,    13/900],
                   7: [         2,    0,      0,    -53/6,    704/45,  -107/9,     67/90,     3],
                   8: [   -91/108,    0,      0,   23/108,  -976/135,  311/54,    -19/60,  17/6,  -1/12],
                   9: [ 2383/4100,    0,      0, -341/164, 4496/1025, -301/82, 2133/4100, 45/82, 45/164, 18/41],
                   10:[     3/205,    0,      0,        0,         0,   -6/41,    -3/205, -3/41,   3/41,  6/41],
                   11:[-1777/4100,    0,      0, -341/164, 4496/1025, -289/82, 2193/4100, 51/82, 33/164, 12/41,   0, 1],
                   12:[    41/840,    0,      0,        0,         0,  34/105,      9/35,  9/35,  9/280, 9/280, 41/840]}

        #coefficients for local truncation error estimate
        self.TR = [41/840, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41/840, -41/840, -41/840]


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
        timestep_rescale = 0.9 * (error_ratio)**(1/8)        

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
        if self.stage < 12:
            self.stage += 1
            return True, 0.0, 1.0
        else: 
            self.stage = 0
            return self.error_controller(dt)