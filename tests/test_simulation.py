########################################################################################
##
##                                  TESTS FOR 
##                               'simulation.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.simulation import Simulation

#for testing
from pathsim.blocks._block import Block
from pathsim.connection import Connection



# TESTS ================================================================================

class TestSimulation(unittest.TestCase):
    """
    Test the implementation of the 'Simulation' class

    only very minimal functonality
    """

    def test_init(self):

        #test default initialization
        Sim = Simulation(log=False)
        self.assertEqual(Sim.blocks, [])
        self.assertEqual(Sim.connections, [])
        self.assertEqual(Sim.dt, 0.01)
        self.assertEqual(Sim.dt_min, 0.0)
        self.assertEqual(Sim.dt_max, None)
        self.assertEqual(str(Sim.Solver()), "SSPRK22")
        self.assertEqual(Sim.tolerance_fpi, 1e-12)
        self.assertEqual(Sim.tolerance_lte, 1e-8)
        self.assertEqual(Sim.iterations_min, 1) # <-- determined from internal path length
        self.assertEqual(Sim.iterations_max, 200)
        self.assertFalse(Sim.log)

        #test specific initialization
        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(blocks=[B1, B2, B3], 
                         connections=[C1, C2, C3], 
                         dt=0.02, 
                         dt_min=0.001, 
                         dt_max=0.1, 
                         tolerance_fpi=1e-9, 
                         tolerance_lte=1e-6, 
                         iterations_min=None, 
                         iterations_max=100, 
                         log=False)
        self.assertEqual(len(Sim.blocks), 3)
        self.assertEqual(len(Sim.connections), 3)
        self.assertEqual(Sim.dt, 0.02)
        self.assertEqual(Sim.dt_min, 0.001)
        self.assertEqual(Sim.dt_max, 0.1)
        self.assertEqual(Sim.tolerance_fpi, 1e-9)
        self.assertEqual(Sim.tolerance_lte, 1e-6)
        self.assertEqual(Sim.iterations_min, 3) # <-- determined from internal path length
        self.assertEqual(Sim.iterations_max, 100)

        #test specific initialization with connection override
        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B2) # <-- overrides B2
        with self.assertRaises(ValueError):
            Sim = Simulation(blocks=[B1, B2, B3], 
                             connections=[C1, C2, C3],
                             log=False)


    def test_add_block(self):
        
        Sim = Simulation(log=False)

        self.assertEqual(Sim.blocks, [])

        #test adding a block
        B1 = Block()
        Sim.add_block(B1)
        self.assertEqual(Sim.blocks, [B1])

        #test adding the same block again
        with self.assertRaises(ValueError):
            Sim.add_block(B1)


    def test_add_connection(self): 

        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)

        Sim = Simulation(blocks=[B1, B2, B3], 
                         connections=[C1],
                         log=False)

        self.assertEqual(Sim.connections, [C1])

        #test adding a connection
        C2 = Connection(B2, B3)
        Sim.add_connection(C2)
        self.assertEqual(Sim.connections, [C1, C2])

        #test adding the same connection again
        with self.assertRaises(ValueError):
            Sim.add_connection(C2)
        self.assertEqual(Sim.connections, [C1, C2])

        #test adding a connection that overrides B3
        C3 = Connection(B1, B3) 
        with self.assertRaises(ValueError):
            Sim.add_connection(C3)
        self.assertEqual(Sim.connections, [C1, C2])


    def test_set_solver(self): pass
    def test_update(self): pass
    def test_step(self): pass
    def test_run(self): pass


class TestSimulationIVP(unittest.TestCase):
    """
    special test case:
    linear feedback initial value problem with default solver (SSPRK22)
    """

    def setUp(self):

        #modules from pathsim for test case
        from pathsim.blocks import (
            Integrator, Scope, 
            Amplifier, Adder
            )

        #blocks that define the system
        self.Int = Integrator(1.0)
        self.Amp = Amplifier(-1)
        self.Add = Adder()
        self.Sco = Scope(labels=["response"])

        blocks = [self.Int, self.Amp, self.Add, self.Sco]

        #the connections between the blocks
        connections = [
            Connection(self.Amp, self.Add[1]),
            Connection(self.Add, self.Int),
            Connection(self.Int, self.Amp, self.Sco)
            ]

        #initialize simulation with the blocks, connections, timestep and logging enabled
        self.Sim = Simulation(blocks, connections, dt=0.02, log=False)


    def test_init(self):

        from pathsim.solvers import SSPRK22

        #test initialization of simulation
        self.assertEqual(len(self.Sim.blocks), 4)
        self.assertEqual(len(self.Sim.connections), 3)
        self.assertEqual(self.Sim.dt, 0.02)
        self.assertEqual(self.Sim.iterations_min, 2)
        self.assertTrue(isinstance(self.Sim.engine, SSPRK22))
        self.assertTrue(self.Sim.is_explicit)
        self.assertFalse(self.Sim.is_adaptive)

        #test if engine setup was correct
        self.assertTrue(isinstance(self.Int.engine, SSPRK22)) # <-- only the Integrator needs an engine
        self.assertTrue(self.Amp.engine is None)
        self.assertTrue(self.Add.engine is None)
        self.assertTrue(self.Sco.engine is None)


    def test_step(self):

        #reset first
        self.Sim.reset()

        #check if reset was sucecssful
        self.assertEqual(self.Sim.time, 0.0)
        self.assertEqual(self.Int.get(0), self.Int.initial_value)

        #step using global timestep
        success, err, scl, te, ts = self.Sim.step()
        self.assertEqual(self.Sim.time, self.Sim.dt)
        self.assertEqual(err, 0.0) #fixed solver
        self.assertEqual(scl, 1.0) #fixed solver
        self.assertEqual(ts, 0) #no implicit solver
        self.assertGreaterEqual(te, self.Sim.iterations_min)
        
        #step again using custom timestep
        self.Sim.step(dt=2.2*self.Sim.dt)
        self.assertEqual(self.Sim.time, 3.2*self.Sim.dt)
        self.assertLess(self.Int.get(0), self.Int.initial_value)

        #test if scope recorded correctly
        time, data = self.Sco.read()
        for a, b in zip(time, [self.Sim.dt, 3.2*self.Sim.dt]):
            self.assertEqual(a, b)

        #reset again
        self.Sim.reset()

        #check if reset was sucecssful
        self.assertEqual(self.Sim.time, 0.0)
        self.assertEqual(self.Int.get(0), self.Int.initial_value)


    def test_run(self):

        #reset first
        self.Sim.reset()

        #check if reset was sucecssful
        self.assertEqual(self.Sim.time, 0.0)
        self.assertEqual(self.Int.get(0), self.Int.initial_value)

        #test running for some time
        self.Sim.run(duration=2, reset=True)
        self.assertEqual(self.Sim.time, 2)
        
        time, data = self.Sco.read()
        _time = np.arange(0, 2.02, 0.02)

        #time recording matches and solution decays
        self.assertLess(np.linalg.norm(time - _time), 1e-13) 
        self.assertTrue(np.all(np.diff(data) < 0.0))

        #test running for some time with reset
        self.Sim.run(duration=1, reset=True)
        self.assertEqual(self.Sim.time, 1)

        time, data = self.Sco.read()
        _time = np.arange(0, 1.02, 0.02)

        #time recording matches and solution decays
        self.assertLess(np.linalg.norm(time - _time), 1e-13) 

        #test running for some time without reset
        self.Sim.run(duration=2, reset=False)
        self.assertEqual(self.Sim.time, 3)

        time, data = self.Sco.read()
        _time = np.arange(0, 3.02, 0.02)

        #time recording matches and solution decays
        self.assertLess(np.linalg.norm(time - _time), 1e-13) 







# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)