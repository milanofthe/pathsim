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

from .value import Value, der, jac


# CLASS ================================================================================

class NewtonRaphsonAD:
    """
    This class implements the newton raphson method using pathsims 
    automatic differentiation framework to compute anlytical jacobians.
    """

    def __init__(self):
        self.x = None


    def solve(self, func, x0, iterations_max=100, tolerance=1e-6):
        """Solve the function 'func' with initial 
        value 'x0' up to a certain tolerance.

        Parameters
        ----------
        func : callable
            function to solve
        x0 : numeric
            starting value for solution
        iterations_max : int
            maximum number of solver iterations
        tolerance : float
            convergence condition

        Returns
        -------
        x : numeric
            solution
        res : float
            residual
        i : int
            iteration count
        """

        _x = x0.copy()
        for i in range(iterations_max):
            _x, _res = self.step(_x, func(_x)+_x)
            if _res < tolerance:
                return _x, _res, i

        raise RuntimeError(f"did not converge in {iterations_max} steps")


    def reset(self):
        self.x = None


    def step(self, x, g, *args, **kwargs):
        """Perform one newton-raphson step

        Parameters
        ----------
        x : float, array
            current solution
        g : float, array
            current evaluation of g(x)
        
        Returns
        -------
        x : float, array
            new solution
        res : float
            residual norm, fixed point error
        """

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


class GaussNewtonAD(NewtonRaphsonAD):
    """
    This class implements the gauss newton method using pathsims 
    automatic differentiation framework to compute anlytical jacobians.
    """

    def step(self, x, g, *args, **kwargs):
        """Perform one gauss-newton step

        Parameters
        ----------
        x : float, array
            current solution
        g : float, array
            current evaluation of g(x)
        
        Returns
        -------
        x : float, array
            new solution
        res : float
            residual norm, fixed point error
        """

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


class LevenbergMarquardtAD(NewtonRaphsonAD):
    """This class implements the levenberg marquardt algorithm using pathsims 
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
        """Adjust the LM parameters based on some cost.

        Parameters
        ----------
        cost : float
            cost for LM parameter adjustment
        """

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
        """Perform one LM step

        Parameters
        ----------
        x : float, array
            current solution
        g : float, array
            current evaluation of g(x)
        
        Returns
        -------
        x : float, array
            new solution
        res : float
            residual norm, fixed point error
        """

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
