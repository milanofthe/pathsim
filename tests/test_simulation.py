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


class TestSimulationFeedback(unittest.TestCase):
    """
    replicates 'example_feedback.py' with tests    
    """

    def setUp(self):

        #modules from pathsim for test case
        from pathsim.blocks import (
            Integrator, 
            Source, 
            Scope, 
            Amplifier, 
            Adder
            )

        #simulation timestep
        dt = 0.02

        #step delay
        tau = 3

        #blocks that define the system
        self.Src = Source(lambda t: int(t>tau))
        self.Int = Integrator(0)
        self.Amp = Amplifier(-1)
        self.Add = Adder()
        self.Sco = Scope(labels=["step", "response"])

        blocks = [self.Src, self.Int, self.Amp, self.Add, self.Sco]

        #the connections between the blocks
        connections = [
            Connection(self.Src, self.Add[0], self.Sco[0]),
            Connection(self.Amp, self.Add[1]),
            Connection(self.Add, self.Int),
            Connection(self.Int, self.Amp, self.Sco[1])
            ]

        #initialize simulation with the blocks, connections, timestep and logging enabled
        self.Sim = Simulation(blocks, connections, dt=dt, log=False)


    def test_init(self):

        from pathsim.solvers import SSPRK22

        #test initialization of simulation
        self.assertEqual(len(self.Sim.blocks), 5)
        self.assertEqual(len(self.Sim.connections), 4)
        self.assertEqual(self.Sim.dt, 0.02)
        self.assertEqual(self.Sim.iterations_min, 2)
        self.assertTrue(isinstance(self.Sim.engine, SSPRK22))
        self.assertTrue(self.Sim.is_explicit)
        self.assertFalse(self.Sim.is_adaptive)

        #test if engine setup was correct
        self.assertTrue(self.Src.engine is None)
        self.assertTrue(isinstance(self.Int.engine, SSPRK22)) # <-- only the Integrator needs an engine
        self.assertTrue(self.Amp.engine is None)
        self.assertTrue(self.Add.engine is None)
        self.assertTrue(self.Sco.engine is None)


    def test_step(self):

        #reset first
        self.Sim.reset()
        self.assertEqual(self.Sim.time, 0.0)

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



    def test_run(self): pass



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)