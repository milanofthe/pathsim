#########################################################################################
##
##                         MATH FUNCTIONS FOR VALUE AND OPTIMIZER
##                                     (functions.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
from .value import Value


# FUNCTION DEFINITIONS ==================================================================

def sin(x):
    """
    Sine function for Value class.
    """
    if isinstance(x, Value):
        return Value(val=np.sin(x.val), 
                     grad={k: v * np.cos(x.val) 
                           for k, v in x.grad.items()})
    else:
        return np.sin(x)


def cos(x):
    """
    Cosine function for Value class.
    """
    if isinstance(x, Value):
        return Value(val=np.cos(x.val), 
                     grad={k: -v * np.sin(x.val) 
                           for k, v in x.grad.items()})
    else:
        return np.cos(x)


def exp(x):
    """
    Exponential function for Value class.
    """
    if isinstance(x, Value):
        return Value(val=np.exp(x.val), 
                     grad={k: v * np.exp(x.val) 
                           for k, v in x.grad.items()})
    else:
        return np.exp(x)


def log(x, base=np.e):
    """
    Logarithm function for Value class.
    """
    if isinstance(x, Value):
        return Value(val=np.log(x.val) / np.log(base), 
                     grad={k: v / (x.val * np.log(base)) 
                           for k, v in x.grad.items()})
    else:
        return np.log(x) / np.log(base)


def log10(x):
    """
    Base-10 logarithm function for Value class.
    """
    return log(x, base=10)


def log2(x):
    """
    Base-2 logarithm function for Value class.
    """
    return log(x, base=2)


def tan(x):
    """
    Tangent function for Value class.
    """
    if isinstance(x, Value):
        return Value(val=np.tan(x.val), 
                     grad={k: v / (np.cos(x.val) ** 2) 
                           for k, v in x.grad.items()})
    else:
        return np.tan(x)


def sinh(x):
    """
    Hyperbolic sine function for Value class.
    """
    if isinstance(x, Value):
        return Value(val=np.sinh(x.val), 
                     grad={k: v * np.cosh(x.val) 
                           for k, v in x.grad.items()})
    else:
        return np.sinh(x)


def cosh(x):
    """
    Hyperbolic cosine function for Value class.
    """
    if isinstance(x, Value):
        return Value(val=np.cosh(x.val), 
                     grad={k: v * np.sinh(x.val) 
                           for k, v in x.grad.items()})
    else:
        return np.cosh(x)


def tanh(x):
    """
    Hyperbolic tangent function for Value class.
    """
    if isinstance(x, Value):
        return Value(val=np.tanh(x.val), 
                     grad={k: v / (np.cosh(x.val) ** 2) 
                           for k, v in x.grad.items()})
    else:
        return np.tanh(x)


def sqrt(x):
    """
    Square root function for Value class.
    """
    if isinstance(x, Value):
        return Value(val=np.sqrt(x.val), 
                     grad={k: v / (2 * np.sqrt(x.val)) 
                           for k, v in x.grad.items()})
    else:
        return np.sqrt(x)