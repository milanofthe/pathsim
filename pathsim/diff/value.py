#########################################################################################
##
##                                 MATH VALUE DEFINITION  
##                                       (value.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import functools


# HELPER FUNCTIONS ======================================================================

def der(array, val):
    """
    Compute the derivative of an array of 'Value' objects with respect 
    to 'val', fallback to scalar case and non-'Value' objects
    
    INPUTS : 
        array : (array[Value]) array or list of Values
        val   : (Value) dual number value for AD
    """
    if np.isscalar(array): return array.d(val) if isinstance(array, Value) else 0.0
    else: return np.array([a.d(val) if isinstance(a, Value) else 0.0 for a in array])


# FUNCTION GRADIENTS ====================================================================

FUNC_GRAD = {
    # Trigonometric functions
    np.sin     : lambda x: np.cos(x),
    np.cos     : lambda x: -np.sin(x),
    np.tan     : lambda x: 1/np.cos(x)**2,
    np.arcsin  : lambda x: 1/np.sqrt(1 - x**2),
    np.arccos  : lambda x: -1/np.sqrt(1 - x**2),
    np.arctan  : lambda x: 1/(1 + x**2),
    np.sinh    : lambda x: np.cosh(x),
    np.cosh    : lambda x: np.sinh(x),
    np.tanh    : lambda x: 1 - np.tanh(x)**2,
    np.arcsinh : lambda x: 1/np.sqrt(x**2 + 1),
    np.arccosh : lambda x: 1/(np.sqrt(x - 1)*np.sqrt(x + 1)),
    np.arctanh : lambda x: 1/(1 - x**2),

    # Exponential and logarithmic functions
    np.exp     : lambda x: np.exp(x),
    np.exp2    : lambda x: np.exp2(x)*np.log(2),
    np.log     : lambda x: 1/x,
    np.log2    : lambda x: 1/(x*np.log(2)),
    np.log10   : lambda x: 1/(x*np.log(10)),
    np.log1p   : lambda x: 1/(1 + x),
    np.expm1   : lambda x: np.exp(x),

    # Power functions
    np.sqrt    : lambda x: 0.5/np.sqrt(x),
    np.square  : lambda x: 2*x,
    np.power   : lambda x, p: p*np.power(x, p-1),
    np.cbrt    : lambda x: 1/(3*np.cbrt(x)**2),

    # Complex functions
    np.conj    : lambda x: np.conj(x),
    np.abs     : lambda x: x/np.abs(x),  
    np.angle   : lambda x: -1j/x,

    # Statistical functions
    np.sign    : lambda x: 0.0,
}


# WRAPPER THAT ADDS ADDITIONAL METHODS ==================================================

def add_funcs(cls):
    """
    Decorator that adds numpy functions as methods to the Value class while overloading 
    them with their analytical derivatives to propagate the gradient with the chain rule.
    """
    
    def create_method(fnc, grd):
        @functools.wraps(fnc)
        def method(self, *args, **kwargs):
            val = fnc(self.val, *args, **kwargs)
            grad_val = grd(self.val, *args, **kwargs)
            return cls(val=val, grad={k: grad_val * v for k, v in self.grad.items()})
        return method

    # Add methods to class
    for fnc, grd in FUNC_GRAD.items():
        if not hasattr(cls, fnc.__name__):
            setattr(cls, fnc.__name__, create_method(fnc, grd))
    
    return cls


# CLASS DEFINITION ======================================================================

@add_funcs
class Value:
    """
    Dual number 'Value' definition for small autograd framework. 

    The dunder methods of the 'Value' class are overloaded to simultaneously compute 
    the partial derivatives with respect to the instance itself and other instances 
    of the value class. 

    This is realized by a dictionary that handles the reference tracking via the id 
    of the 'Value' instances.
    
    INPUTS : 
        val  : (float, int, complex) The numerical value.
        grad : (dict, optional) The gradient dictionary. If None, initializes with self derivative.
    """

    #restrict attributes, makes access faster
    __slots__ = ["val", "grad", "_id"] 
    _id_counter = 0


    def __init__(self, val=0.0, grad=None):
        self.val = val
        if grad is None:
            self._id = Value._id_counter
            self.grad = {self._id:1.0} 
            Value._id_counter += 1
        else:
            self.grad = grad
            self._id = None


    def d(self, other):
        """
        Get the partial derivative with respect to 'other'.

        INPUTS :    
            other : (Value) variable with respect to which to take the derivative

        RETURNS :
            float : The partial derivative value
        """
        return self.grad.get(other._id, 0.0)


    @property
    def real(self):
        return Value(
            val=np.real(self.val), 
            grad={k: np.real(v) for k, v in self.grad.items()}
            ) 


    @property
    def imag(self):
        return Value(
            val=np.imag(self.val), 
            grad={k: np.imag(v) for k, v in self.grad.items()}
            ) 


    def __hash__(self):
        return id(self)


    def __iter__(self):
        yield self


    def __repr__(self):
        return f"Value(val={self.val}, grad={self.grad})"    


    # comparison operators ----------------------------------------------------------------------

    def __eq__(self, other):
        if isinstance(other, Value):
            return self.val == other.val
        else:
            return self.val == other


    def __ne__(self, other):
        if isinstance(other, Value):
            return self.val != other.val
        else:
            return self.val != other


    def __lt__(self, other):
        if isinstance(other, Value):
            return self.val < other.val
        else:
            return self.val < other


    def __gt__(self, other):
        if isinstance(other, Value):
            return self.val > other.val
        else:
            return self.val > other


    def __le__(self, other):
        if isinstance(other, Value):
            return self.val <= other.val
        else:
            return self.val <= other


    def __ge__(self, other):
        if isinstance(other, Value):
            return self.val >= other.val
        else:
            return self.val >= other


    # type casting operators --------------------------------------------------------------------

    def __bool__(self):
        return bool(self.val)


    def __int__(self):
        return int(self.val)


    def __float__(self):
        return float(self.val)


    # unary operators ---------------------------------------------------------------------------

    def __pos__(self):
        return self


    def __neg__(self):
        return Value(
            val=-self.val, 
            grad={k: -v for k, v in self.grad.items()}
            )


    def __abs__(self):
        return Value(
            val=abs(self.val), 
            grad={k: v * np.sign(self.val) for k, v in self.grad.items()}
            )


    # arithmetic operators ----------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, Value):
            return Value(
                val=self.val + other.val, 
                grad={k: self.grad.get(k, 0) + other.grad.get(k, 0) 
                      for k in range(Value._id_counter)}
                )
        elif isinstance(other, np.ndarray):
            return np.array([self + x for x in other])
        else:
            return Value(
                val=self.val + other, 
                grad=self.grad
                )


    def __radd__(self, other):
        return self + other


    def __iadd__(self, other):
        return self + other


    def __mul__(self, other):
        if isinstance(other, Value):
            return Value(
                val=self.val * other.val, 
                grad={k: self.grad.get(k, 0) * other.val + self.val * other.grad.get(k, 0) 
                      for k in range(Value._id_counter)}
                )
        elif isinstance(other, np.ndarray):
            return np.array([self * x for x in other])
        else:
            return Value(
                val=self.val * other, 
                grad={k: v * other for k, v in self.grad.items()}
                )


    def __rmul__(self, other):
        return self * other


    def __imul__(self, other):
        return self * other        


    def __sub__(self, other):
        if isinstance(other, Value):
            return Value(
                val=self.val - other.val, 
                grad={k: self.grad.get(k, 0) - other.grad.get(k, 0) 
                      for k in range(Value._id_counter)}
                )
        elif isinstance(other, np.ndarray):
            return np.array([self - x for x in other])
        else:
            return Value(
                val=self.val - other, 
                grad=self.grad
                )


    def __rsub__(self, other):
        return -(self - other)


    def __isub__(self, other):
        return self - other


    def __truediv__(self, other):
        if isinstance(other, Value):
            return Value(
                val=self.val / other.val, 
                grad={k: (self.grad.get(k, 0) * other.val - self.val * other.grad.get(k, 0)) / (other.val ** 2) 
                      if other.val != 0.0 else 0.0 for k in range(Value._id_counter)}
                )
        if isinstance(other, np.ndarray):
            return np.array([self / x for x in other])
        else:
            return Value(
                val=self.val / other, 
                grad={k: v / other for k, v in self.grad.items()}
                )


    def __rtruediv__(self, other):
        if isinstance(other, Value):
            return other / self
        else:
            return Value(
                val=other / self.val, 
                grad={k: -other * v / (self.val ** 2) if self.val != 0.0 else 0.0
                      for k, v in self.grad.items()}
                )


    def __pow__(self, power):
        if isinstance(power, Value):
            new_val = self.val ** power.val
            return Value(
                val=new_val, 
                grad={k: new_val * (power.val * self.grad.get(k, 0) / self.val + np.log(self.val) * power.grad.get(k, 0))
                      for k in range(Value._id_counter)}
                )
        else:
            return Value(
                val=self.val ** power, 
                grad={k: power * (self.val ** (power - 1)) * v for k, v in self.grad.items()}
                )


    def __rpow__(self, base):
        new_val = base ** self.val
        return Value(
            val=new_val, 
            grad={k: new_val * np.log(base) * v for k, v in self.grad.items()}
            )