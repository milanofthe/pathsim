########################################################################################
##
##                                  TESTS FOR 
##                              'solvers/gear.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.gear import GEAR32, GEAR43, GEAR43

from ._referenceproblems import problems


# TESTS ================================================================================

class TestGEAR32(unittest.TestCase):
    """
    Test the implementation of the 'GEAR32' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = GEAR32()

        self.assertTrue(callable(solver.func))
        self.assertEqual(solver.jac, None)
        self.assertEqual(solver.initial_value, 0)

        self.assertEqual(solver.stage, 0)
        self.assertTrue(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = GEAR32(initial_value=1, 
                        func=lambda x, u, t: -x, 
                        jac=lambda x, u, t: -1, 
                        tolerance_lte_rel=1e-3, 
                        tolerance_lte_abs=1e-6)

        self.assertEqual(solver.func(2, 0, 0), -2)
        self.assertEqual(solver.jac(2, 0, 0), -1)
        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_stages(self):

        solver = GEAR32()

        for i, t in enumerate(solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, solver.eval_stages[i])


    def test_buffer(self):

        solver = GEAR32()

        #perform some steps
        for k in range(10):

            #buffer state
            solver.buffer(0)

            #test bdf buffer length
            self.assertEqual(len(solver.B), k+1 if k < 2 else 2)
            self.assertEqual(len(solver.T), k+1 if k < 2 else 2)
            
            #make one step
            for i, t in enumerate(solver.stages(0, 1)):
                success, err, scale = solver.step(0.0, t, 1)