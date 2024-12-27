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
from pathsim.blocks import (
    Source, 
    Integrator, 
    Amplifier, 
    Adder, 
    Scope
    )


# TESTS ================================================================================

class TestExampleLinearFeedback(unittest.TestCase):
    """
    linear feedback system
    """

    def setUp(self):

        #step delay
        self.tau = 3
        
        #blocks that define the system
        self.Src = Source(lambda t: int(t>self.tau))
        self.Int = Integrator()
        self.Amp = Amplifier(-1)
        self.Add = Adder()
        self.Sco = Scope()

        blocks = [self.Int, self.Src, self.Amp, self.Add, self.Sco]

        #the connections between the blocks
        connections = [
            Connection(self.Src, self.Add[0], self.Sco[0]),
            Connection(self.Amp, self.Add[1]),
            Connection(self.Add, self.Int),
            Connection(self.Int, self.Amp, self.Sco[1])
            ]

        #initialize simulation with the blocks, connections, timestep and logging enabled
        self.Sim = Simulation(blocks, connections, dt=0.02, log=False)


    def test_run(self):

        #run the simulation for some time
        stats = self.Sim.run(3*self.tau, reset=True)

        steps = stats["total_steps"]
        evals = stats["function_evaluations"]
        iters = stats["solver_iterations"]

        self.assertEqual(steps, 3*self.tau/self.Sim.dt+1)
        self.assertGreater(evals, steps)
        self.assertEqual(iters, 0)

        time, [stp, dta] = self.Sco.read()

        self.assertEqual(len(time), steps+1)
        self.assertEqual(len(stp), steps+1)
        self.assertEqual(len(dta), steps+1)

        #step response stays below step
        for s, d in zip(stp, dta):
            self.assertTrue(s>=d)


    def test_reset(self):

        #run the simulation for some time
        stats = self.Sim.run(3*self.tau, reset=True)

        steps = stats["total_steps"]
        evals = stats["function_evaluations"]
        iters = stats["solver_iterations"]

        self.assertEqual(steps, 3*self.tau/self.Sim.dt+1)
        self.assertGreater(evals, steps)
        self.assertEqual(iters, 0)

        time, [stp, dta] = self.Sco.read()

        self.assertEqual(len(time), steps+1)
        self.assertEqual(len(stp), steps+1)
        self.assertEqual(len(dta), steps+1)

        self.Sim.reset()

        time, data = self.Sco.read()

        self.assertTrue(time is None)
        self.assertTrue(data is None)        


    def test_steadystate(self):

        #run the simulation for some time
        stats = self.Sim.run(1.1*self.tau, reset=True)

        #force to steady state
        self.Sim.steadystate(reset=False)

        time, [stp, dta] = self.Sco.read()

        self.assertEqual(dta[-1], 1.0)





# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)