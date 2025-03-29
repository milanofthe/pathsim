#########################################################################################
##
##                   DUAL NUMBER DEFINITION FOR AUTOMATIC DIFFERENTIATION  
##                                   (optim.value.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import functools

from collections import defaultdict


# HELPER FUNCTIONS ======================================================================

def der(arr, val):
    """Compute the derivative of an array of 'Value' objects with respect 
    to 'val', fallback to scalar case and non-'Value' objects
    
    Parameters
    ----------
    arr : array[Value]
        array or list of Values
    val : Value
        dual number value for AD
    
    Returns
    -------
    der : array[float]
        partial derivatives w.r.t. the 'val'
    """
    return np.array([a(val) if isinstance(a, Value) else 0.0 for a in np.atleast_1d(arr)])


def jac(arr, vals):
    """Compute the derivative of an array of 'Value' objects with respect 
    to each 'Value' object in 'vals', fallback to scalars in both cases.

    This effectively constructs the jacobian. 

    Parameters
    ----------
    arr : array[Value]
        array or list of Values
    vals : array[Value]
        array or list of Values
    
    Returns
    -------
    jac : array[float]
        partial derivatives w.r.t. all values in 'vals', effectively the jacobian
    """
    return np.array([der(arr, val) for val in np.atleast_1d(vals)]).T


def var(arr, vals):
    """Compute the variance of an array of 'Value' objects with respect 
    to all the 'Value' objects in 'vals' using the standard deviations 
    of the values and a first order taylor approximation using the 
    partial derivatives.

    Parameters
    ----------
    arr : array[Value]
        array or list of Values
    vals : array[Value]
        array or list of Values
    
    Returns
    -------
    var : array[float]
        variance from taylor expansion w.r.t. all values in 'vals'
    """
    return sum((val.sig*der(arr, val))**2 for val in np.atleast_1d(vals))


def autojac(fnc):
    """Decorator that wraps a function such that it computes its jacobian 
    alongside its evaluaiton.
    
    Parameters
    ----------
    fnc : callable
        function to wrap and compute jacobian of
    """
    @functools.wraps(fnc)
    def wrap(*args):
        vals = Value.array(args)
        out = fnc(*vals)
        return Value.numeric(out), jac(out, vals)
    return wrap


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
    """Decorator that adds numpy functions as methods to the Value class while overloading 
    them with their analytical derivatives to propagate the gradient with the chain rule.
    """
    
    def create_method(fnc, grd):
        @functools.wraps(fnc)
        def method(self, *args, **kwargs):
            grad_val = grd(self.val, *args, **kwargs)
            return cls(
                val=fnc(self.val, *args, **kwargs), 
                grad=defaultdict(
                    float, 
                    {k: grad_val * v for k, v in self.grad.items()}
                )
            )
        return method

    # Add methods to class
    for fnc, grd in FUNC_GRAD.items():
        if not hasattr(cls, fnc.__name__):
            setattr(cls, fnc.__name__, create_method(fnc, grd))
    
    return cls


# CLASS DEFINITION ======================================================================

@add_funcs
class Value:
    """Dual number 'Value' definition for small autograd framework. 

    The dunder methods of the 'Value' class are overloaded to simultaneously compute 
    the partial derivatives with respect to the instance itself and other instances 
    of the value class. 

    This is realized by a dictionary that handles the reference tracking via the id 
    of the 'Value' instances.

    Example
    -------
    Automatic differentiation of numpy functions:

    .. code-block:: python
    
        import numpy as np

        v = Value(3)
    
        #AD
        w = np.sin(v)
    
        #reference
        dw_dv = np.cos(v)

        print(dw_dv == w(v))
        #True    
    
    Automatic differentiation of concatenations:

    .. code-block:: python
    
        import numpy as np

        v = Value(2)
        
        #AD
        w = np.exp(np.sin(v) + v**3)
        
        #reference
        dw_dv = np.exp(np.sin(v) + v**3) * (np.cos(v) + 3*v**2)

        print(dw_dv == w(v))
        #True

    Automatic differentiation with multiple values:

    .. code-block:: python
    
        import numpy as np
    
        u = Value(0.5)
        v = Value(2)
        
        #AD
        w = u * v + np.sin(u + v)
        
        #references
        dw_du = v + np.cos(u + v)
        dw_dv = u + np.cos(u + v)

        print(dw_du == w(u), dw_dv == w(v))
        #True True

    Parameters
    ----------
    val  : float, int, complex
        The numerical value.
    grad : defaultdict, None
        The gradient dictionary. If None, initializes with self derivative.
    sig : float, int, complex
        Standard deviation of value for uncertainty analysis (optional)

    Attributes
    ----------
    _id : int 
        id for reference tracking in gradient dict

    """

    #restrict attributes, makes access faster
    __slots__ = ["val", "grad", "sig", "_id"] 


    def __init__(self, val=0.0, grad=None, sig=0.0):

        #catch instantiating with Value
        if isinstance(val, Value):
            self.val = val.val
        else:
            self.val = val

        #initialize fresh gradients
        if grad is None:
            self._id = id(self)
            self.grad = defaultdict(float, {self._id: 1.0})
        else:
            self._id = None
            self.grad = grad

        #standard deviation
        self.sig = sig


    # dynamic properties ------------------------------------------------------------------------

    @property
    def real(self):
        return Value(
            val=np.real(self.val), 
            grad=defaultdict(float, {k: np.real(v) for k, v in self.grad.items()})
            ) 


    @property
    def imag(self):
        return Value(
            val=np.imag(self.val), 
            grad=defaultdict(float, {k: np.imag(v) for k, v in self.grad.items()})
            ) 


    # array conversions -------------------------------------------------------------------------

    @classmethod
    def numeric(cls, arr):
        """Cast an array with value objects to an array of numeric values. 
        
        Numeric entries are just passed through.
    
        Parameters
        ----------  
        arr : array[obj]
            array of mixed value, numeric objects
    
        Returns
        -------
        numeric : array[numeric] 
            array of numeric values
        """
        _arr = np.atleast_1d(arr)
        return np.array([a.val if isinstance(a, cls) else a  
            for a in _arr.ravel()]).reshape(_arr.shape).squeeze()


    @classmethod
    def array(cls, arr):
        """Cast an array or list to an array of value objects. 
        
        For Value entries, their numeric values are used to 
        create a new Value instance.

        Example
        -------
        Instantiate multiple values from 1d array:

        .. code-block:: python

            a, b, c = Value.array([1, 2, 3])

        Instantiate numpy array filled with values:

        .. code-block:: python
            
            import numpy as np

            A_num = np.random.rand(3, 5)

            A = Value.array(A_num)

        Parameters
        ----------  
        arr : array[obj]
            array of mixed value, numeric objects
    
        Returns
        -------
        array : array[Value] 
            array of Value objects
        """
        _arr = np.atleast_1d(arr)
        return np.array([cls(a) for a in _arr.ravel()]).reshape(_arr.shape).squeeze()


    # convenience static methods ----------------------------------------------------------------

    @staticmethod
    def der(arr, val):
        return der(arr, val)


    @staticmethod
    def jac(arr, vals):
        return jac(arr, vals)


    @staticmethod
    def var(arr, vals):
        return var(arr, vals)


    # overload builtins -------------------------------------------------------------------------

    def __call__(self, other):
        """
        Get the partial derivative with respect to 'other'.

        Parameters
        ----------     
        other : Value
            variable with respect to which to take the derivative

        Returns
        -------
        out : float
            The partial derivative value
        """
        if isinstance(other, Value): return self.grad[other._id]
        else: return 0.0


    def __hash__(self):
        return id(self)


    def __iter__(self):
        yield self


    def __len__(self):
        return len(self.grad)


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


    def __complex__(self):
        return complex(self.val)


    # unary operators ---------------------------------------------------------------------------

    def __pos__(self):
        return self


    def __neg__(self):
        return Value(
            val=-self.val, 
            grad=defaultdict(float, {k: -v for k, v in self.grad.items()})
            )


    def __abs__(self):
        return Value(
            val=abs(self.val), 
            grad=defaultdict(float, {k: v * np.sign(self.val) for k, v in self.grad.items()})
            )


    # arithmetic operators ----------------------------------------------------------------------

    def __add__(self, other):
        if isinstance(other, Value):
            new_grad = defaultdict(float, self.grad)
            for k, v in other.grad.items(): new_grad[k] += v
            return Value(
                val=self.val + other.val, 
                grad=new_grad
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
        if isinstance(other, Value):
            self.val += other.val
            for k, v in other.grad.items(): self.grad[k] += v
            return self
        else:
            self.val += other
            return self


    def __mul__(self, other):
        if isinstance(other, Value):
            new_grad = defaultdict(float)
            for k, v in self.grad.items(): new_grad[k] += v * other.val
            for k, v in other.grad.items(): new_grad[k] += v * self.val
            return Value(
                val=self.val * other.val, 
                grad=new_grad
                )
        elif isinstance(other, np.ndarray):
            return np.array([self * x for x in other])
        else:
            return Value(
                val=self.val * other, 
                grad=defaultdict(float, {k: v * other for k, v in self.grad.items()})
                )


    def __rmul__(self, other):
        return self * other


    def __imul__(self, other):
        if isinstance(other, Value):
            self.val *= other.val
            for k, v in self.grad.items(): self.grad[k] *= other.val 
            for k, v in other.grad.items(): self.grad[k] += v * self.val
            return self
        else:
            self.val *= other
            return self


    def __sub__(self, other):
        if isinstance(other, Value):
            new_grad = defaultdict(float, self.grad)
            for k, v in other.grad.items(): new_grad[k] -= v
            return Value(
                val=self.val - other.val, 
                grad=new_grad
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
        if isinstance(other, Value):
            self.val -= other.val
            for k, v in other.grad.items(): self.grad[k] -= v
            return self
        else:
            self.val -= other
            return self


    def __truediv__(self, other):
        if isinstance(other, Value):
            new_grad = defaultdict(float)
            if other.val != 0.0:
                a, b = self.val/other.val**2, 1/other.val
                for k, v in self.grad.items(): new_grad[k] += v*b
                for k, v in other.grad.items(): new_grad[k] -= v*a
            return Value(
                val=self.val / other.val, 
                grad=new_grad
                )
        if isinstance(other, np.ndarray):
            return np.array([self / x for x in other])
        else:
            return Value(
                val=self.val / other, 
                grad=defaultdict(float, {k: v / other for k, v in self.grad.items()})
                )


    def __rtruediv__(self, other):
        if isinstance(other, Value):
            return other / self
        else:
            new_grad = defaultdict(float)
            if self.val != 0.0:
                a = -other / (self.val ** 2)
                for k, v in self.grad.items(): new_grad[k] += v*a
            return Value(
                val=other / self.val, 
                grad=new_grad
                )


    def __pow__(self, power):
        if isinstance(power, Value):
            new_val = self.val ** power.val
            new_grad = defaultdict(float)
            a, b = new_val * power.val / self.val, new_val * np.log(self.val)
            for k, v in self.grad.items(): new_grad[k] += v*a
            for k, v in power.grad.items(): new_grad[k] += v*b 
            return Value(
                val=new_val, 
                grad=new_grad
                )
        else:
            return Value(
                val=self.val ** power, 
                grad=defaultdict(float, {k: power * (self.val ** (power - 1)) * v for k, v in self.grad.items()})
                )


    def __rpow__(self, base):
        new_val = base ** self.val
        return Value(
            val=new_val, 
            grad=defaultdict(float, {k: new_val * np.log(base) * v for k, v in self.grad.items()})
            )