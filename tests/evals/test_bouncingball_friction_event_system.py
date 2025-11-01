########################################################################################
##
##                      Testing bouncing ball with friction event system
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
    ESDIRK32, ESDIRK43, ESDIRK54, ESDIRK85
    )


# TESTCASE =============================================================================

class TestBouncingBallFrictionEventSystem(unittest.TestCase):

    def setUp(self):

        """Set up the system with everything thats needed
        for the evaluation exposed"""

        #gravitational acceleration
        self.g = 9.81

        #elasticity of bounce
        self.b = 0.95

        #mass normalized friction coefficient
        self.k = 0.2

        #initial values
        x0, v0 = 1, 5

        #newton friction
        def fric(v):
            return -self.k * np.sign(v) * v**2

        #blocks that define the system
        self.Ix = Integrator(x0)     # v -> x
        self.Iv = Integrator(v0)     # a -> v
        Fr = Function(fric)          # newton friction
        Ad = Adder()
        Cn = Constant(-self.g)       # gravitational acceleration
        self.Sc = Scope(labels=["x", "v"])

        blocks = [self.Ix, self.Iv, Fr, Ad, Cn, self.Sc]

        #the connections between the blocks
        connections = [
            Connection(Cn, Ad[0]),
            Connection(Fr, Ad[1]),
            Connection(Ad, self.Iv),
            Connection(self.Iv, self.Ix, Fr),
            Connection(self.Ix, self.Sc[0])
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
            self.Iv.engine.set(-self.b*v)

        #events (zero crossing)
        self.E1 = ZeroCrossing(
            func_evt=func_evt,
            func_act=func_act,
            tolerance=1e-6
            )

        events = [self.E1]

        #initialize simulation with the blocks, connections
        self.Sim = Simulation(
            blocks,
            connections,
            events,
            log=False
            )


    def test_eval_explicit_solvers(self):

        #test for different solvers
        for SOL in [RKBS32, RKF45, RKCK54, RKV65, RKDP87]:

            with self.subTest("subtest solver", SOL=str(SOL)):

                self.Sim.reset()
                self.Sim._set_solver(SOL)
                self.Sim.run(8)

                time, [x] = self.Sc.read()

                #check that position stays non-negative (ball doesn't go underground)
                self.assertTrue(np.min(x) >= 0)

                #check that events were detected
                self.assertTrue(len(self.E1) > 0)

                #check that ball eventually settles (energy dissipation)
                final_height = np.max(x[time > 6])
                self.assertTrue(final_height < 0.5)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
