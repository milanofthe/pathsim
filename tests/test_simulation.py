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

    def test_init(self): pass
    def test_add_block(self): pass
    def test_add_connection(self): pass
    def test_set_solver(self): pass
    def test_step(self): pass
    def test_run(self): pass



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
