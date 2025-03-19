#########################################################################################
##
##                              ALGEBRAIC OPERATOR DEFINITION
##                                   (optim.operator.py)
##
##                                    Milan Rother 2025
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .value import Value
from .numerical import num_jac


# OPERATOR CLASS ========================================================================

class Operator(object):
    """Operator class for function evaluation and linearization.
    
    This class wraps a function to provide both direct evaluation and linear approximation
    capabilities. When linearized around a point x0, subsequent calls use the first-order
    Taylor approximation 
    
    .. math::

        f(x) \\approx f(x_0) + \\mathbf{J}(x_0) (x - x_0)


    instead of evaluating the function.
    
    The class supports multiple methods for Jacobian computation: user-provided analytical
    Jacobians, automatic differentiation via the Value class, and numerical differentiation
    as a fallback.
    
    Example
    -------
    Basic usage with automatic differentiation:
    
    .. code-block:: python
    
        def f(x):
            return x**2 + np.sin(x)
            
        op = Operator(f)
        
        # Direct function evaluation
        y1 = op(2.0)
        
        # Linearize at current point
        op.linearize(2.0)
        
        # Use linear approximation
        y2 = op(2.1)  # Returns f(2.0) + J(2.0) (2.1-2.0)
        
    With user-provided Jacobian:
    
    .. code-block:: python
    
        def f(x):
            return x**2 + np.sin(x)
            
        def df_dx(x):
            return 2*x + np.cos(x)
            
        op = Operator(f, jac=df_dx)
        
        op.linearize(2.0)  # Uses df_dx for Jacobian
    
    Parameters
    ----------
    func : callable
        The function to wrap
    jac : callable, optional
        Optional analytical Jacobian of func. If None, automatic or numerical
        differentiation will be used.

    Attributes
    ----------
    x0 : array_like
        operating point
    f0 : array_like
        function evaluation at operating point
    J : array_like
        jacobian matrix at operating point

    """

    def __init__(self, func, jac=None):
        self._func = func
        self._jac = jac
        self.x0 = None
        self.f0 = None
        self.J = None
        
        
    def __call__(self, x):
        """Evaluate the function or its linear approximation.
        
        If the operator has been linearized (x0 is not None), returns the linear
        approximation 
    
        .. math::

            f(x_0) + \\mathbf{J}(x_0) (x - x_0)
    

        otherwise, returns f(x) directly.
        
        Parameters
        ----------
        x : array_like
            Point at which to evaluate
            
        Returns
        -------
        value : array_like
            Function value or linear approximation
        """
        if self.x0 is None: return self._func(x)
        return self.f0 + np.dot(self.J, x - self.x0)
        

    def jac(self, x):
        """Compute the Jacobian matrix at point x.
        
        Uses the following methods in order of preference:
        1. User-provided analytical Jacobian if available
        2. Automatic differentiation via Value class
        3. Numerical differentiation as fallback
        
        Parameters
        ----------
        x : array_like
            Point at which to evaluate the Jacobian
            
        Returns
        -------
        jacobian : ndarray
            Jacobian matrix at x
        """
        if self._jac is None:
            try:
                # Try automatic differentiation
                _x = Value.array(x)
                return Value.jac(self._func(_x), _x)
            except:
                # Fallback to numerical differentiation
                return num_jac(self._func, x)
        else:
            # Use analytical jacobian
            return self._jac(x)
            

    def linearize(self, x):
        """Linearize the function at point x.
        
        Computes and stores both the function value and its Jacobian at x.
        After linearization, calls to the operator will use the linear
        approximation until reset() is called.
        
        Parameters
        ----------
        x : array_like
            Point at which to linearize the function
        """
        self.x0, self.f0, self.J = x, self._func(x), self.jac(x)
        

    def reset(self):
        """Reset the linearization.
        
        Clears the stored linearization point and Jacobian, causing the
        operator to evaluate the function directly on subsequent calls.
        """
        self.x0, self.f0, self.J = None, None, None