########################################################################################
##
##                               ANDERSON ACCELERATION 
##                                (utils/anderson.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np
import matplotlib.pyplot as plt


# CLASS ================================================================================

class AndersonAcceleration:
    """
    Class for accelerated fixed-point iteration through anderson acceleration. 
    Solves a nonlinear set of equations given in the fixed-point form:

        x = g(x)

    Anderson Accelerstion tracks the evolution of the solution from the previous 
    iterations. The next step in the iteration is computed as a linear combination 
    of the previous iterates. The coefficients are computed to minimize the least 
    squares error of the fixed-point problem.

    INPUTS : 
        m       : (int) buffer length
        restart : (bool) clear buffer when full
    """

    def __init__(self, m=1, restart=True):

        #length of buffer for next estimate
        self.m = m

        #restart after buffer length is reached?
        self.restart = restart

        #rolling buffers
        self.x_buffer = []
        self.f_buffer = []

        #iteration counter for debugging
        self.counter = 0


    def reset(self):
        """
        reset the anderson accelerator
        """

        #clear buffers
        self.x_buffer = []
        self.f_buffer = []

        #reset iteration counter
        self.counter = 0


    def step(self, x, g):
        """
        Perform one iteration on the fixed-point solution.

        INPUTS : 
            x : (float or array) current solution
            g : (float or array) current evaluation of g(x)
        """

        #increment counter
        self.counter += 1

        #residual (this gets minimized)
        f = g - x
        
        #fallback to regular fpi if 'm == 0'
        if self.m == 0:
            return g, np.linalg.norm(f)

        #make x vectorial if g is vector
        if np.isscalar(x) and not np.isscalar(g):
            x *= np.ones_like(g)
    
        #append to buffer
        self.x_buffer.append(x)
        self.f_buffer.append(f)

        #total buffer length
        k = len(self.f_buffer)

        #if no buffer, regular fixed-point update
        if k == 1:
            return g, np.linalg.norm(f)

        #if buffer size 'm' reached, restart or truncate
        elif self.m is not None and k > self.m + 1:
            if self.restart:
                self.x_buffer = []
                self.f_buffer = []
                return g, np.linalg.norm(f)
            else:
                self.x_buffer.pop(0)
                self.f_buffer.pop(0)

        #get deltas 
        dX = np.diff(self.x_buffer, axis=0)
        dF = np.diff(self.f_buffer, axis=0)

        #exit for scalar values
        if np.isscalar(f):

            #delta squared norm
            dF2 = np.linalg.norm(dF)**2

            #catch division by zero
            if dF2 <= 1e-14:
                return g, abs(f)

            #new solution and residual
            return x - f * sum(dF * dX) / dF2, abs(f)

        #compute coefficients from least squares problem
        C, *_ = np.linalg.lstsq(dF.T, f, rcond=None)

        #new solution and residual
        return x - C @ dX, np.linalg.norm(f)



class NewtonAndersonAcceleration(AndersonAcceleration):
    """
    Modified class for hybrid anderson acceleration that can use a jacobian 'jac' of 
    the function 'g' for a newton step before the fixed point step for the initial 
    estimate before applying the anderson acceleration.

    If a jacobian 'jac' is available, this significantly improves the convergence 
    (speed and robustness) of the solution.
    """

    def _newton(self, x, g, jac):
        """
        Newton step on solution, where 'f=g-x' is the 
        residual and 'jac' is the jacobian of 'g'.
        """

        #compute residual
        f = g - x

        #early exit for scalar or purely vectorial values
        if np.isscalar(f) or np.ndim(jac) == 1:
            return x - f / (jac - 1.0), abs(f)

        #vectorial values (newton raphson)
        jac_f = jac - np.eye(len(f))
        return x - np.linalg.solve(jac_f, f), np.linalg.norm(f)


    def step(self, x, g, jac=None):
        """
        Perform one iteration on the fixed-point solution. 
        If the jacobian of g 'jac' is provided, a newton step 
        is performed previous to anderson acceleration.
        """

        #newton step if jacobian available
        if jac is None: 

            #regular anderson step with residual
            return super().step(x, g)
        else: 
            #newton step with residual
            _x, res = self._newton(x, g, jac)

            #anderson step with no residual
            y, _ = super().step(_x, g)

            return y, res
