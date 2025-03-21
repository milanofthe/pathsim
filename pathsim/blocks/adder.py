#########################################################################################
##
##                                      ADDER BLOCK 
##                                   (blocks/adder.py)
##
##                                   Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..utils.utils import dict_to_array
from ..optim.operator import Operator


# MISO BLOCKS ===========================================================================

class Adder(Block):
    """Summs / adds up all input signals to a single output signal (MISO)
    
    This is how it works in the default case

    .. math::
        
        y(t) = \\sum_i u_i(t)


    and like this when additional operations are defined

    .. math::
        
        y(t) = \\sum_i \\mathrm{op}_i \\cdot u_i(t)
    

    Example
    -------

    This is the default initialization that just adds up all the inputs:

    .. code-block:: python

        A = Adder()


    and this is the initialization with specific operations that subtracts 
    the second from first input and neglects all others:

    .. code-block:: python
    
        A = Adder('+-')


    Parameters
    ----------
    operations : str, optional
        optional string of operations to be applied before 
        summation, i.e. '+-' will compute the difference, 
        'None' will just perform regular sum
    
    
    Attributes
    ----------
    _ops : dict
        dict that maps string operations to numerical
    op_alg : Operator
        internal algebraic operator
    """

    def __init__(self, operations=None):
        super().__init__()

        #allowed arithmetic operations
        self._ops = {"+":1.0, "-":-1.0, "0":0.0}

        #input validation
        if operations is not None:
            if not isinstance(operations, str):
                raise ValueError("'operations' must be string or 'None'")
            for op in operations:
                if op not in self._ops:
                    raise ValueError(f"operation '{op}' not in {self._ops}")
        
        self.operations = operations    

        def sum_ops(X):
            return sum(x*self._ops[op] for op, x in zip(self.operations, X))
        def jac_ops(X):
            return np.array([self._ops[op] for op in self.operations])

        #create internal algebraic operator
        self.op_alg = Operator(
            func=sum if self.operations is None else sum_ops, 
            jac=lambda x: np.ones(len(x)) if self.operations is None else jac_ops
            )


    def update(self, t):
        """update system equation in fixed point loop

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        error : float
            absolute error to previous iteration for convergence control
        """
        u = dict_to_array(self.inputs)
        _out, self.outputs[0] = self.outputs[0], self.op_alg(u)
        return abs(_out - self.outputs[0])
