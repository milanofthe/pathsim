########################################################################################
##
##                               ConnectionBooster CLASS 
##                                  (optim/booster.py)
##
##       class to boost connections, injecting a fixed point acelerator for loop 
##             closing connections to simplify the algebraic loop solver
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from .anderson import Anderson


# CLASS =================================================================================

class ConnectionBooster:
    """Wraps a `Connection` instance and injects a fixed point accelerator. 

    This class is part of the solver structure and intended to improve the 
    algebraic loop solver of the simulation.

    Parameters
    ----------
    connection : Connection
        connection instance to be boosted with an algebraic loop accelerator

    Attributes
    ----------
    accelerator : Anderson
        internal fixed point accelerator instance
    history : float | int | array_like
        history, previous evaliation of the connection value
    """

    def __init__(self, connection):
        self.connection = connection
        self.accelerator = Anderson()
        self.history = self.get()


    def __bool__(self):
        return len(self.connections) > 0


    def get(self):
        """Return the output values of the source block that is referenced in 
        the connection.

        Return 
        ------
        out : float | int | array_like
            output values of source, referenced in connection
        """
        return self.connection.source.get_outputs()


    def set(self, val): 
        """Set targets input values.

        Parameters
        ----------
        val : float | int | array_like
            input values to set at inputs of the targets, referenced by the 
            connection

        """
        for trg in self.connection.targets:
            trg.set_inputs(val)


    def reset(self):
        """Reset the internal fixed point accelerator and update the history 
        to the most recent value
        """
        self.accelerator.reset()
        self.history = self.get()


    def update(self):
        """Wraps the `Connection.update` method for data transfer from source 
        to targets and injects a solver step of the fixed point accelerator, 
        updates the history required for the next solver step, returns the 
        fixed point residual.

        Returns
        -------
        res : float
            fixed point residual of internal lixed point accelerator
        """
        _val, res = self.accelerator.step(self.history, self.get())
        self.set(_val)
        self.history = _val
        return res