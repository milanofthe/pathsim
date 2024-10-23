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


# CLASS DEFINITION ======================================================================


class Value:
    """
    Dual number 'Value' definition for small autograd framework. 

    The dunder methods of the 'Value' class are overloaded to simultaneously compute 
    the partial derivatives with respect to the instance itself and other instances 
    of the value class. 

    This is realized by a dictionary that handles the reference tracking via the unique 
    identifiers of the 'Value' instances.
    """

    def __init__(self, val=0.0, grad=None):
        self.val = val
        self.grad = {self:1.0} if grad is None else grad

    def d(self, other):
        #partial derivative with respect to 'other'
        return self.grad.get(other, 0.0)

    def set(self, val):
        self.val = val

    def __hash__(self):
        return id(self)

    def __iter__(self):
        yield self

    def __repr__(self):
        return f"Value(val={self.val}, grad={self.grad})"    

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

    def __bool__(self):
        return bool(self.val)

    def __int__(self):
        return int(self.val)

    def __float__(self):
        return float(self.val)

    def __pos__(self):
        return self

    def __neg__(self):
        return Value(val=-self.val, 
                     grad={k: -v for k, v in self.grad.items()})

    def __abs__(self):
        return Value(val=abs(self.val), 
                     grad={k: v * np.sign(self.val) 
                           for k, v in self.grad.items()})

    def __truediv__(self, other):
        if isinstance(other, Value):
            new_grad = {k: (self.grad.get(k, 0) * other.val - self.val * other.grad.get(k, 0)) / (other.val ** 2) 
                        if other.val != 0.0 else 0.0 for k in set(self.grad) | set(other.grad)}
            return Value(val=self.val / other.val, grad=new_grad)
        else:
            return Value(val=self.val / other, 
                         grad={k: v / other for k, v in self.grad.items()})

    def __rtruediv__(self, other):
        if isinstance(other, Value):
            return other / self
        else:
            return Value(val=other / self.val, 
                         grad={k: -other * v / (self.val ** 2) if self.val != 0.0 else 0.0
                               for k, v in self.grad.items()})

    def __add__(self, other):
        if isinstance(other, Value):
            new_grad = {k: self.grad.get(k, 0) + other.grad.get(k, 0) 
                        for k in set(self.grad) | set(other.grad)}
            return Value(val=self.val + other.val, grad=new_grad)
        else:
            return Value(val=self.val + other, grad=self.grad)

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, Value):
            new_grad = {k: self.grad.get(k, 0) * other.val + self.val * other.grad.get(k, 0) 
                        for k in set(self.grad) | set(other.grad)}
            return Value(val=self.val * other.val, grad=new_grad)
        else:
            return Value(val=self.val * other, 
                         grad={k: v * other for k, v in self.grad.items()})

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        if isinstance(other, Value):
            new_grad = {k: self.grad.get(k, 0) - other.grad.get(k, 0) 
                        for k in set(self.grad) | set(other.grad)}
            return Value(val=self.val - other.val, grad=new_grad)
        else:
            return Value(val=self.val - other, grad=self.grad)

    def __rsub__(self, other):
        return -(self - other)

    def __isub__(self, other):
        return self - other

    def __pow__(self, power):
        if isinstance(power, Value):
            new_val = self.val ** power.val
            new_grad = {k: new_val * (power.val * self.grad.get(k, 0) / self.val + np.log(self.val) * power.grad.get(k, 0))
                        for k in set(self.grad) | set(power.grad)}
            return Value(val=new_val, grad=new_grad)
        else:
            return Value(val=self.val ** power, 
                         grad={k: power * (self.val ** (power - 1)) * v 
                               for k, v in self.grad.items()})

    def __rpow__(self, base):
        new_val = base ** self.val
        new_grad = {k: new_val * np.log(base) * v for k, v in self.grad.items()}
        return Value(val=new_val, grad=new_grad)


class Parameter(Value):
    """
    Class that enhances the 'Value' class by some additional parameters and 
    methods that make it suitable to be used within an optimization framework.
    """

    def __init__(self, val=0.0, min_val=0, max_val=1, grad=None):
        self.val = val
        self.min_val = min_val
        self.max_val = max_val
        self.grad = {self:1.0} if grad is None else grad

    def shuffle(self):
        self.val = self.min_val + (self.max_val - self.min_val) * np.random.rand()