########################################################################################
##
##                                     TESTS FOR 
##                                'optim/anderson.py'
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.optim.anderson import (
    Anderson, 
    NewtonAnderson
    )


# TESTS ================================================================================

class TestAnderson(unittest.TestCase):
    """
    test the implementation of the 'Anderson' class 
    """

    def test_init(self):
        m = 5
        aa = Anderson(m)
        # test initialization
        self.assertEqual(aa.m, m)
        self.assertFalse(aa.restart)
        self.assertEqual(len(aa.dx_buffer), 0)
        self.assertEqual(len(aa.dr_buffer), 0)

    def test_reset(self):
        aa = Anderson(5)
        aa.x_buffer = [1, 2, 3]
        aa.f_buffer = [4, 5, 6]
        aa.counter = 10
        aa.reset()
        # test reset
        self.assertEqual(len(aa.dx_buffer), 0)
        self.assertEqual(len(aa.dr_buffer), 0)

    def test_step_scalar(self):
        aa = Anderson(2)
        x, g = 1.0, 2.0
        result, residual = aa.step(x, g)
        # test scalar step
        self.assertEqual(result, g)
        self.assertEqual(residual, abs(g - x))

    def test_step_vector(self):
        aa = Anderson(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 3.0])
        result, residual = aa.step(x, g)
        # test vector step
        np.testing.assert_array_equal(result, g)
        self.assertAlmostEqual(residual, np.linalg.norm(g - x))

    def test_solve_scalar_equation(self):
        # Solve x = cos(x)
        aa = Anderson(m=3)
        x = 0.0
        for _ in range(100):
            g = np.cos(x)
            x, residual = aa.step(x, g)
            if residual < 1e-8:
                break
        self.assertAlmostEqual(x, 0.7390851332151607, places=7)


    def test_solve_vector_equation(self):
        # Solve the system:
        # x^2 + y^2 = 1
        aa = Anderson(m=3)
        x = np.array([1.0, 0.0])  # Start from a point on the circle
        for _ in range(100):
            g = x / np.linalg.norm(x)  # Project back onto the unit circle
            x, residual = aa.step(x, g)
            if residual < 1e-8:
                break
        # Check if the solution lies on the unit circle
        self.assertAlmostEqual(np.linalg.norm(x), 1.0, places=7)


class TestNewtonAnderson(unittest.TestCase):
    """
    test the implementation of the 'NewtonAnderson' class 
    """

    def test_init(self):
        m = 5
        naa = NewtonAnderson(m)
        # test initialization
        self.assertEqual(naa.m, m)
        self.assertFalse(naa.restart)
        self.assertEqual(len(naa.dx_buffer), 0)
        self.assertEqual(len(naa.dr_buffer), 0)

    def test_step_no_jacobian(self):
        naa = NewtonAnderson(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 3.0])
        result, residual = naa.step(x, g)
        # test step without jacobian (should be same as Anderson)
        np.testing.assert_array_equal(result, g)
        self.assertAlmostEqual(residual, np.linalg.norm(g - x))

    def test_step_with_jacobian(self):
        naa = NewtonAnderson(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 3.0])
        jac = np.array([[2.0, 0.0], [0.0, 2.0]])
        result, residual = naa.step(x, g, jac)
        # test step with jacobian
        self.assertIsNotNone(result)
        self.assertIsNotNone(residual)

    def test_solve_scalar_equation(self):
        # Solve x = cos(x)
        naa = NewtonAnderson(m=5)
        x = 0.0
        for i in range(100):
            g = np.cos(x)
            jac = -np.sin(x)
            x, residual = naa.step(x, g, jac)
            if residual < 1e-8:
                break
        self.assertAlmostEqual(x, 0.7390851332151607, places=7)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)