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
        self.assertEqual(Sim.iterations_min, 1)
        self.assertEqual(Sim.iterations_max, 200)
        self.assertFalse(Sim.log)

        #test specific initialization



    def test_add_block(self): pass
    def test_add_connection(self): pass
    def test_set_solver(self): pass
    def test_step(self): pass
    def test_run(self): pass


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)