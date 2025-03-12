########################################################################################
##
##                    TEST LINEAR FEEDBACK SYSTEM SIMULATION
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Constant, Function, Adder, Scope

from pathsim.events import ZeroCrossing

from pathsim.solvers import RKBS32


# TESTS ================================================================================

class TestExampleBouncingBall(unittest.TestCase):
    """
    testing zero crossing detection for bouncing ball
    """

    def setUp(self):    

        #blocks that define the system
        self.Ix = Integrator(1)     # v -> x
        self.Iv = Integrator(0)     # a -> v 
        self.Cn = Constant(-9.81)       # gravitational acceleration
        self.Sc = Scope(labels=["x", "v"])


        blocks = [self.Ix, self.Iv, self.Cn, self.Sc]

        #the connections between the blocks
        connections = [
            Connection(self.Cn, self.Iv),
            Connection(self.Iv, self.Ix),
            Connection(self.Ix, self.Sc[0]),
            Connection(self.Iv, self.Sc[1])
            ]


        #event function for zero crossing detection
        def func_evt(t):
            *_, x = self.Ix() #get block outputs and states
            return x

        #action function for state transformation
        def func_act(t):
            *_, x = self.Ix()
            *_, v = self.Iv()
            self.Ix.engine.set(abs(x))
            self.Iv.engine.set(-v)

        #events (zero crossing)
        self.E1 = ZeroCrossing(
            func_evt=func_evt,                 
            func_act=func_act, 
            tolerance=1e-4
            )

        events = [self.E1]

        #initialize simulation 
        self.Sim = Simulation(
            blocks, 
            connections, 
            events, 
            dt=0.01, 
            dt_max=0.05,
            log=False, 
            Solver=RKBS32, 
            tolerance_lte_rel=1e-4, 
            tolerance_lte_abs=1e-6
            )



    def test_run(self):

        self.Sim.run(10)

        #check number of events
        self.assertEqual(len(self.E1), 11)

        #check height remains bounded
        _time, [height, _] = self.Sc.read()
        for h in height[1:]:
            self.assertTrue(h <= height[0])



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)