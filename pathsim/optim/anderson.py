########################################################################################
##
##                               ANDERSON ACCELERATION 
##                                (optim/anderson.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from collections import deque

from .value import Value, der, jac


# CLASS ================================================================================

class Anderson:
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

    def __init__(self, m=5, restart=False):

        #length of buffer for next estimate
        self.m = m

        #restart after buffer length is reached?
        self.restart = restart

        #rolling difference buffers
        self.dx_buffer = deque(maxlen=self.m)
        self.dr_buffer = deque(maxlen=self.m)

        #prvious values
        self.x_prev = None
        self.r_prev = None


    def reset(self):
        """
        reset the anderson accelerator
        """

        #clear difference buffers
        self.dx_buffer.clear()
        self.dr_buffer.clear()

        #clear previous values
        self.x_prev = None
        self.r_prev = None


    def step(self, x, g):
        """
        Perform one iteration on the fixed-point solution.

        INPUTS : 
            x : (float or array) current solution
            g : (float or array) current evaluation of g(x)
        """

        #make numeric if value
        _x = Value.numeric(x)
        _g = Value.numeric(g)

        #residual (this gets minimized)
        _res = _g - _x
        
        #fallback to regular fpi if 'm == 0'
        if self.m == 0:
            return g, np.linalg.norm(_res)
    
        #if no buffer, regular fixed-point update
        if self.x_prev is None:

            #save values for next iteration
            self.x_prev = _x
            self.r_prev = _res

            return g, np.linalg.norm(_res)

        #append to difference buffer
        self.dx_buffer.append(_x - self.x_prev)
        self.dr_buffer.append(_res - self.r_prev)
        
        #save values for next iteration
        self.x_prev = _x
        self.r_prev = _res

        #if buffer size 'm' reached, restart
        if self.restart and len(self.dx_buffer) >= self.m:
            self.reset()
            return g, np.linalg.norm(_res)

        #get difference matrices 
        dX = np.array(self.dx_buffer)
        dR = np.array(self.dr_buffer)

        #exit for scalar values
        if np.isscalar(_res):

            #delta squared norm
            dR2 = np.dot(dR, dR)

            #catch division by zero
            if dR2 <= 1e-14:
                return g, abs(_res)

            #new solution and residual
            return x - _res * np.dot(dR, dX) / dR2, abs(_res)

        #compute coefficients from least squares problem
        C, *_ = np.linalg.lstsq(dR.T, _res, rcond=None)

        #new solution and residual norm
        return x - C @ dX, np.linalg.norm(_res)



class NewtonAnderson(Anderson):
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

        #make numeric if value
        _x   = Value.numeric(x)
        _g   = Value.numeric(g)
        _jac = Value.numeric(jac)
        
        #compute residual
        _res = _g - _x

        #early exit for scalar or purely vectorial values
        if np.isscalar(_res) or np.ndim(_jac) == 1:
            
            return x - _res / (_jac - 1.0), np.linalg.norm(_res)

        #vectorial values (newton raphson)
        return x - np.linalg.solve(_jac - np.eye(len(_res)), _res), np.linalg.norm(_res)


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
            _x, res_norm = self._newton(x, g, jac)

            #anderson step with no residual
            y, _ = super().step(_x, g)

            return y, res_norm