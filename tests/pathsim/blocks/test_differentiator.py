########################################################################################
##
##                                  TESTS FOR 
##                          'blocks.differentiator.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.differentiator import Differentiator

#base solver for testing
from pathsim.solvers._solver import Solver 


# TESTS ================================================================================

class TestDifferentiator(unittest.TestCase):
    """
    Test the implementation of the 'Differentiator' block class
    """

    def test_init(self):

        #test default initialization
        D = Differentiator()
        self.assertEqual(D.f_max, 1e2)
        self.assertEqual(D.engine, None)

        #test special initialization
        D = Differentiator(f_max=1e3)
        self.assertEqual(D.f_max, 1e3)


    def test_len(self):

        D = Differentiator()

        #has direct passthrough
        self.assertEqual(len(D), 1)


    def test_set_solver(self):

        D = Differentiator()

        #test that no solver is initialized
        self.assertEqual(D.engine, None)

        D.set_solver(Solver, None, tolerance_lte_rel=1e-3, tolerance_lte_abs=1e-6)

        #test that solver is now available
        self.assertTrue(isinstance(D.engine, Solver))
        self.assertEqual(D.engine.tolerance_lte_rel, 1e-3)
        self.assertEqual(D.engine.tolerance_lte_abs, 1e-6)

        D.set_solver(Solver, None, tolerance_lte_rel=1e-2, tolerance_lte_abs=1e-3)

        #test that solver tolerance is changed
        self.assertEqual(D.engine.tolerance_lte_rel, 1e-2)
        self.assertEqual(D.engine.tolerance_lte_abs, 1e-3)


    def test_update(self):

        D = Differentiator()
        D.set_solver(Solver, None)

        #test that input is zero
        self.assertEqual(D.inputs[0], 0.0)

        D.update(0)

        #test if state is retrieved correctly
        self.assertEqual(D.outputs[0], 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
