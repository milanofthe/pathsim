########################################################################################
##
##                          Testing bouncing ball system
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Constant, Function, Adder, Scope
from pathsim.events import ZeroCrossing

from pathsim.solvers import (
    RKF21, RKBS32, RKF45, RKCK54, RKDP54, RKV65, RKF78, RKDP87, 
    GEAR52A, GEAR32, GEAR43
    )


# TESTCASE =============================================================================

class TestBouncingBallSystem(unittest.TestCase):

    def setUp(self):

        """Set up the system with everything thats needed 
        for the evaluation exposed"""

        #gravitational acceleration
        self.g = 9.81

        #initial values
        self.h = 1

        #blocks that define the system
        Ix = Integrator(self.h)   
        Iv = Integrator()   
        Cn = Constant(-self.g)     
        Sc = Scope(labels=["x", "v"])

        blocks = [Ix, Iv, Cn, Sc]

        #the connections between the blocks
        connections = [
            Connection(Cn, Iv),
            Connection(Iv, Ix),
            Connection(Ix, Sc[0])
            ]

        #event function for zero crossing detection
        def func_evt(t):
            *_, x = Ix() #get block outputs and states
            return x

        #action function for state transformation
        def func_act(t):
            *_, x = Ix()
            *_, v = Iv()
            Ix.engine.set(abs(x))
            Iv.engine.set(-v)

        #events (zero crossing)
        self.E1 = ZeroCrossing(
            func_evt=func_evt,                 
            func_act=func_act, 
            tolerance=1e-6
            )

        events = [self.E1]

        #initialize simulation
        self.Sim = Simulation(
            blocks, 
            connections, 
            events,
            log=False
            )


    def test_eval_explicit_solvers(self):

        #fall time
        T = np.sqrt(2*self.h/self.g)

        #test for different solvers
        for SOL in [RKF21, RKBS32, RKF45, RKCK54, RKDP54, RKV65, RKF78, RKDP87]:

            with self.subTest("subtest solver", SOL=str(SOL)):            

                self.Sim.reset()
                self.Sim._set_solver(SOL)
                self.Sim.run(10)

                #check if all the bounce times are detected correctly
                for i, t in enumerate(self.E1):
                    t_ref = T + i*2*T

                    self.assertAlmostEqual(t, t_ref, 4)  


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
