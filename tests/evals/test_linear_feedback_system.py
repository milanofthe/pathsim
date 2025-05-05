########################################################################################
##
##                          Testing linear feedback system
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source, 
    Integrator, 
    Amplifier, 
    Adder, 
    Scope
    )

from pathsim.solvers import (
    RKF21, RKBS32, RKF45, RKCK54, RKDP54, RKV65, RKF78, RKDP87, 
    ESDIRK32, ESDIRK43, ESDIRK54, ESDIRK85
    )


# TESTCASE =============================================================================

class TestLinearFeedbackSystem(unittest.TestCase):

    def setUp(self):

        """Set up the system with everything thats needed 
        for the evaluation exposed"""

        #blocks that define the system
        Src = Source(lambda t : int(t>3))
        Int = Integrator(2)
        Amp = Amplifier(-1)
        Add = Adder()
        self.Sco = Scope()

        blocks = [Src, Int, Amp, Add, self.Sco]

        #the connections between the blocks
        connections = [
            Connection(Src, Add[0], self.Sco[0]),
            Connection(Amp, Add[1]),
            Connection(Add, Int),
            Connection(Int, Amp, self.Sco[1])
            ]

        #initialize simulation
        self.Sim = Simulation(
            blocks, 
            connections, 
            dt=0.01,
            log=False
            )


    def _reference(self, t):
        #analytical reference solution of the system
        return 2 * np.exp(-t) + (t>3)*(1 - np.exp(-(t-3)))


    def test_eval_explicit_solvers(self):

        #test for different solvers
        for SOL in [RKBS32, RKF45, RKCK54, RKV65, RKDP87]:

            #test for different tolerances
            for tol in [1e-4, 1e-6, 1e-8]:

                with self.subTest("subtest solver with tolerance", SOL=str(SOL), tol=tol):            

                    self.Sim.reset()
                    self.Sim._set_solver(
                        SOL, 
                        tolerance_lte_rel=0.0, 
                        tolerance_lte_abs=tol
                        )

                    self.Sim.run(10)

                    time, [_, res] = self.Sco.read()
                    ref = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(ref-res)), tol, 2)  

            
    def test_eval_implicit_solvers(self):

        #test for different solvers
        for SOL in [ESDIRK32, ESDIRK43, ESDIRK54, ESDIRK85]:

            #test for different tolerances
            for tol in [1e-4, 1e-6, 1e-8]:

                with self.subTest("subtest solver with tolerance", SOL=str(SOL), tol=tol):    

                    self.Sim.reset()
                    self.Sim._set_solver(
                        SOL, 
                        tolerance_lte_rel=0.0, 
                        tolerance_lte_abs=tol,

                        )

                    self.Sim.run(10)

                    time, [_, res] = self.Sco.read()
                    ref = self._reference(time)

                    #checking the global truncation error -> larger
                    self.assertAlmostEqual(np.max(abs(ref-res)), tol, 2) 



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
