########################################################################################
##
##                           NEWTON-TYPE OPTIMIZERS WITH AD
##                                 (optim/newton.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ..diff import Value, der, jac


# CLASS ================================================================================

class NewtonRaphsonAD:
    """
    This class implements the newton raphson method using pathsims 
    automatic differentiation framework to compute anlytical jacobians.
    """

    def __init__(self):
        self.x = None
        self.alpha = 1e-8


    def reset(self):
        self.x = None


    def step(self, x, g, *args, **kwargs):

        #compute residual -> make sure its an array
        res = np.atleast_1d(g - x)

        #cast residual to numeric
        _res = Value.numeric(res)

        #initial fixed-point step
        if self.x is None:

            #cast 'g' to value array
            self.x = Value.array(g)

            return self.x, np.linalg.norm(_res)

        #jacobian with damping
        J = jac(res, self.x)
            
        #check conditioning of jacobian
        if np.isfinite(np.linalg.cond(J)):
            
            #update values in place with newton steps
            for i, dx in enumerate(np.linalg.solve(J, _res)): 
                self.x[i] -= dx
        else:
            #fallback to fixed-point step if singular 
            self.x = Value.array(g)

        return self.x, np.linalg.norm(_res)


class GaussNewtonAD:
    """
    This class implements the gauss newton method using pathsims 
    automatic differentiation framework to compute anlytical jacobians.
    """

    def __init__(self):
        self.x = None


    def reset(self):
        self.x = None


    def step(self, x, g, *args, **kwargs):

        #compute residual -> make sure its an array
        res = np.atleast_1d(g - x)

        #cast residual to numeric
        _res = Value.numeric(res)

        #initial fixed-point step
        if self.x is None:

            #cast 'g' to value array
            self.x = Value.array(g)

            return self.x, np.linalg.norm(_res)

        #jacobian and damping
        J = jac(res, self.x)

        #gauss newton matrix
        JTJ = J.T @ J

        #check conditioning of gnm
        if np.isfinite(np.linalg.cond(JTJ)):
            
            #update values in place gauss newton step
            for i, dx in enumerate(np.linalg.solve(JTJ, J.T @ _res)): 
                self.x[i] -= dx

        else:
            #fallback to fixed-point step if singular 
            self.x = Value.array(g)

        return self.x, np.linalg.norm(_res)


class LevenbergMarquardtAD:
    """
    This class implements the levenberg marquardt algorithm using pathsims 
    automatic differentiation framework to compute anlytical jacobians.
    """

    def __init__(self):
        self.x = None
        self.cost = None
        self.alpha = 1e-6


    def reset(self):
        self.x = None
        self.cost = None
        self.alpha = 1e-6


    def _adjust_params(self, cost):

        #first iteration -> set prev_cost and quit
        if self.cost is None:
            self.cost = cost
            return

        #cost decreased / increased -> adjust parameter
        scale = 0.8 if cost - self.cost < 0 else 2.0 

        #ensure alpha and beta stay within reasonable bounds
        self.alpha = np.clip(self.alpha * scale, 1e-9, 10.0)

        #update prev_cost for next iteration
        self.cost = cost


    def step(self, x, g, *args, **kwargs):

        #compute residual -> make sure its an array
        res = np.atleast_1d(g - x)

        #cast residual to numeric
        _res = Value.numeric(res)

        _res_norm = np.linalg.norm(_res)

        #initial fixed-point step
        if self.x is None:

            #cast 'g' to value array
            self.x = Value.array(g)

            return self.x, _res_norm

        #adjust the lm parameters
        self._adjust_params(_res_norm**2)

        #jacobian with AD and dampig matrix
        J = jac(res, self.x)

        #lm matrix
        LM = J.T @ J + self.alpha * np.eye(len(_res))

        #check conditioning of gnm
        if np.isfinite(np.linalg.cond(LM)):
            
            #update values in place with lm step
            for i, dx in enumerate(np.linalg.solve(LM, J.T @ _res)): 
                self.x[i] -= dx

        else:
            #fallback to fixed-point step if singular 
            self.x = Value.array(g)

        return self.x, _res_norm
