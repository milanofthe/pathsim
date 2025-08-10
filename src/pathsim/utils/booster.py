########################################################################################
##
##                               ConnectionBooster CLASS 
##                                  (utils/booster.py)
##
##       class to boost connections, injecting a fixed point acelerator for loop 
##             closing connections to simplify the algebraic loop solver
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ..optim.anderson import Anderson


# CLASS =================================================================================

class ConnectionBooster:

    def __init__(self, connection):
        self.connection = connection
        self.accelerator = Anderson()
        self.history = self.get()


    def __bool__(self):
        return len(self.connections) > 0


    def get(self):
        return self.connection.source.get_outputs()


    def set(self, val):    
        for trg in self.connection.targets:
            trg.set_inputs(val)


    def reset(self):
        self.accelerator.reset()
        self.history = self.get()


    def update(self):
        _val, res = self.accelerator.step(self.history, self.get())
        self.set(_val)
        self.history = _val
        return res






        


