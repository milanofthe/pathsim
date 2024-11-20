########################################################################################
##
##                                  TESTS FOR 
##                            'blocks.integrator.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.integrator import Integrator

#base solver for testing
from pathsim.solvers._solver import Solver 


# TESTS ================================================================================

class TestIntegrator(unittest.TestCase):
    """
    Test the implementation of the 'Integrator' block class
    """

    def test_init(self):

        #test default initialization
        I = Integrator()
        self.assertEqual(I.initial_value, 0.0)
        self.assertEqual(I.engine, None)

        #test special initialization
        I = Integrator(initial_value=1.0)
        self.assertEqual(I.initial_value, 1.0)


    def test_len(self):

        I = Integrator()

        #no direct passthrough
        self.assertEqual(len(I), 0)


    def test_str(self):

        I = Integrator()

        #test default str method
        self.assertEqual(str(I), "Integrator")


    def test_set_solver(self):

        I = Integrator()

        #test that no solver is initialized
        self.assertEqual(I.engine, None)

        I.set_solver(Solver, tolerance_lte_abs=1e-6, tolerance_lte_rel=1e-3)

        #test that solver is now available
        self.assertTrue(isinstance(I.engine, Solver))
        self.assertEqual(I.engine.tolerance_lte_rel, 1e-3)
        self.assertEqual(I.engine.tolerance_lte_abs, 1e-6)

        I.set_solver(Solver, tolerance_lte_abs=1e-4, tolerance_lte_rel=1e-2)

        #test that solver tolerance is changed
        self.assertEqual(I.engine.tolerance_lte_rel, 1e-2)
        self.assertEqual(I.engine.tolerance_lte_abs, 1e-4)


    def test_update(self):

        I = Integrator(initial_value=5.5)
        I.set_solver(Solver)

        err = I.update(0)

        #test if error is correctly 0
        self.assertEqual(err, 0.0)

        #test if engine state is retrieved correctly
        self.assertEqual(I.get(0), 5.5)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
