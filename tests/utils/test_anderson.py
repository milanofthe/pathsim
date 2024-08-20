########################################################################################
##
##                                     TESTS FOR 
##                                'utils/anderson.py'
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.utils.anderson import (
    AndersonAcceleration, 
    NewtonAndersonAcceleration
    )


# TESTS ================================================================================

class TestAndersonAcceleration(unittest.TestCase):
    """
    test the implementation of the 'AndersonAcceleration' class 
    """

    def test_init(self):
        m = 5
        aa = AndersonAcceleration(m)
        # test initialization
        self.assertEqual(aa.m, m)
        self.assertTrue(aa.restart)
        self.assertEqual(len(aa.x_buffer), 0)
        self.assertEqual(len(aa.f_buffer), 0)
        self.assertEqual(aa.counter, 0)

    def test_reset(self):
        aa = AndersonAcceleration(5)
        aa.x_buffer = [1, 2, 3]
        aa.f_buffer = [4, 5, 6]
        aa.counter = 10
        aa.reset()
        # test reset
        self.assertEqual(len(aa.x_buffer), 0)
        self.assertEqual(len(aa.f_buffer), 0)
        self.assertEqual(aa.counter, 0)

    def test_step_scalar(self):
        aa = AndersonAcceleration(2)
        x, g = 1.0, 2.0
        result, residual = aa.step(x, g)
        # test scalar step
        self.assertEqual(result, g)
        self.assertEqual(residual, abs(g - x))

    def test_step_vector(self):
        aa = AndersonAcceleration(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 3.0])
        result, residual = aa.step(x, g)
        # test vector step
        np.testing.assert_array_equal(result, g)
        self.assertAlmostEqual(residual, np.linalg.norm(g - x))

    def test_solve_scalar_equation(self):
        # Solve x = cos(x)
        aa = AndersonAcceleration(m=3)
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
        aa = AndersonAcceleration(m=3)
        x = np.array([1.0, 0.0])  # Start from a point on the circle
        for _ in range(100):
            g = x / np.linalg.norm(x)  # Project back onto the unit circle
            x, residual = aa.step(x, g)
            if residual < 1e-8:
                break
        # Check if the solution lies on the unit circle
        self.assertAlmostEqual(np.linalg.norm(x), 1.0, places=7)


class TestNewtonAndersonAcceleration(unittest.TestCase):
    """
    test the implementation of the 'NewtonAndersonAcceleration' class 
    """

    def test_init(self):
        m = 5
        naa = NewtonAndersonAcceleration(m)
        # test initialization
        self.assertEqual(naa.m, m)
        self.assertTrue(naa.restart)
        self.assertEqual(len(naa.x_buffer), 0)
        self.assertEqual(len(naa.f_buffer), 0)
        self.assertEqual(naa.counter, 0)

    def test_step_no_jacobian(self):
        naa = NewtonAndersonAcceleration(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 3.0])
        result, residual = naa.step(x, g)
        # test step without jacobian (should be same as AndersonAcceleration)
        np.testing.assert_array_equal(result, g)
        self.assertAlmostEqual(residual, np.linalg.norm(g - x))

    def test_step_with_jacobian(self):
        naa = NewtonAndersonAcceleration(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 3.0])
        jac = np.array([[2.0, 0.0], [0.0, 2.0]])
        result, residual = naa.step(x, g, jac)
        # test step with jacobian
        self.assertIsNotNone(result)
        self.assertIsNotNone(residual)

    def test_solve_scalar_equation(self):
        # Solve x = cos(x)
        naa = NewtonAndersonAcceleration(m=5)
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