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


    def test_add_connection(self): pass
    def test_set_solver(self): pass
    def test_step(self): pass
    def test_run(self): pass


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)