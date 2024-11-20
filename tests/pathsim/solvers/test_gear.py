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

from pathsim.solvers.gear import *

from ._referenceproblems import problems


# TESTS ================================================================================

class TestComputeBDFCoefficients(unittest.TestCase):
    """
    Test the implementation of 'compute_bdf_coefficients'
    """

    def test_order_1(self):

        n = 1

        F, K = 1.0, [1.0]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


    def test_order_2(self):

        n = 2

        F, K = 2/3, [-1/3, 4/3]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


    def test_order_3(self):

        n = 3

        F, K = 6/11, [2/11, -9/11, 18/11]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


    def test_order_4(self):

        n = 4

        F, K = 12/25, [-3/25, 16/25, -36/25, 48/25]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


    def test_order_5(self):

        n = 5

        F, K = 60/137, [12/137, -75/137, 200/137, -300/137, 300/137]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


    def test_order_6(self):

        n = 6

        F, K = 60/147, [-10/147, 72/147, -225/147, 400/147, -450/147, 360/147]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


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


    def test_buffer(self):

        solver = GEAR32()

        #perform some steps
        for k in range(10):

            #buffer state
            solver.buffer(0)

            #test bdf buffer length
            self.assertEqual(len(solver.B), k+1 if k < solver.n else solver.n)
            self.assertEqual(len(solver.T), k+1 if k < solver.n else solver.n)
            
            #make one step
            for i, t in enumerate(solver.stages(0, 1)):
                success, err, scale = solver.step(0.0, t, 1)


