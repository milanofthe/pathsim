########################################################################################
##
##                                    BUNDLE CLASS 
##                                  (utils/bundle.py)
##
##          class to bundle together multiple connections, specifically loop 
##             closing connections to simplify the algebraic loop solver
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from ..optim.anderson import Anderson
from ..connection import Connection


# CLASS =================================================================================

class Bundle:

    def __init__(self, connections):
        self.connections = connections

        if connections:
            self.splits = np.cumsum([len(con) for con in connections])
            self.accelerator = Anderson()
            self.history = self.get()


    def __bool__(self):
        return len(self.connections) > 0


    def get(self):
        return np.hstack([
            con.source.get_outputs() for con in self.connections
            ])


    def set(self, vals):
        for val, con in zip(np.split(vals, self.splits), self.connections):
            for trg in con.targets:
                trg.set_inputs(val)


    def reset(self):
        self.accelerator.reset()
        self.history = self.get()


    def update(self):
        _vals, res = self.accelerator.step(self.history, self.get())
        self.set(_vals)
        self.history = _vals
        return res






        


