#########################################################################################
##
##                                      MATH BLOCKS 
##                                    (blocks/math.py)
##
##                  definitions of elementary math and function blocks
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..optim.operator import Operator


# BASE MATH BLOCK =======================================================================

class Math(Block):


    def __len__(self):
        """Purely algebraic block"""
        return 1


    def update(self, t):
        """update algebraic component of system equation 

        Parameters
        ----------
        t : float
            evaluation time
        """
        u = self.inputs.to_array()
        y = self.op_alg(u)
        self.outputs.update_from_array(y)


# BLOCKS ================================================================================

class Sin(Math):
    """Sine operator block
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.sin, 
            jac=lambda x: np.diag(np.cos(x))
            )


class Cos(Math):
    """Cosine operator block
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=np.cos, 
            jac=lambda x: -np.diag(np.sin(x))
            )


class Sqrt(Math):
    """Square root operator block
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=lambda x: np.sqrt(abs(x)), 
            jac=lambda x: np.diag(1/np.sqrt(abs(x)))
            )


class Abs(Math):
    """Absolute value operator block
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self):
        super().__init__()

        #create internal algebraic operator
        self.op_alg = Operator(
            func=lambda x: abs(x), 
            jac=lambda x: np.diag(np.sign(x))
            )


class Pow(Math):
    """Raise to power operator block

    Parameters
    ----------
    exponent : float, array_like
        exponent to raise the input to the power of
        
    Attributes
    ----------
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self, exponent=2):
        super().__init__()

        self.exponent = exponent

        #create internal algebraic operator
        self.op_alg = Operator(
            func=lambda x: np.power(x, self.exponent), 
            jac=lambda x: np.diag(self.exponent * np.power(x, self.exponent - 1))
            )