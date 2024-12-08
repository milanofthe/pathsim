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


class TestAnderson(unittest.TestCase):
    """
    Extended tests for the 'Anderson' class.
    """

    def test_init(self):
        m = 5
        aa = Anderson(m)
        self.assertEqual(aa.m, m)
        self.assertFalse(aa.restart)
        self.assertEqual(len(aa.dx_buffer), 0)
        self.assertEqual(len(aa.dr_buffer), 0)
        self.assertIsNone(aa.x_prev)
        self.assertIsNone(aa.r_prev)

    def test_reset(self):
        aa = Anderson(5)
        # artificially add some entries
        aa.dx_buffer.append(np.array([1.0]))
        aa.dr_buffer.append(np.array([2.0]))
        aa.x_prev = np.array([1.0])
        aa.r_prev = np.array([2.0])
        aa.reset()
        self.assertEqual(len(aa.dx_buffer), 0)
        self.assertEqual(len(aa.dr_buffer), 0)
        self.assertIsNone(aa.x_prev)
        self.assertIsNone(aa.r_prev)

    def test_step_scalar(self):
        aa = Anderson(2)
        x, g = 1.0, 2.0
        result, residual = aa.step(x, g)
        self.assertEqual(result, g)
        self.assertEqual(residual, abs(g - x))

    def test_step_vector(self):
        aa = Anderson(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 3.0])
        result, residual = aa.step(x, g)
        np.testing.assert_array_equal(result, g)
        self.assertAlmostEqual(residual, np.linalg.norm(g - x))

    def test_solve_converge_scalar(self):
        # Solve x = cos(x) using solve method
        def func_scalar(x):
            return np.cos(x) - x  # f(x)=0 => x=cos(x)
        aa = Anderson(m=5)
        x0 = np.array([0.0])
        x_sol, res, iters = aa.solve(func_scalar, x0, iterations_max=200, tolerance=1e-10)
        self.assertAlmostEqual(x_sol[0], 0.7390851332151607, places=7)

    def test_solve_converge_vector(self):
        # Solve system: x = x / ||x||, so solution is any unit vector. Start from random point.
        def func_vec(x):
            norm = np.linalg.norm(x)
            return x / norm - x
        aa = Anderson(m=5)
        x0 = np.array([1.0, 1.0])
        x_sol, res, iters = aa.solve(func_vec, x0, iterations_max=200, tolerance=1e-10)
        # Check unit circle solution
        self.assertAlmostEqual(np.linalg.norm(x_sol), 1.0, places=7)

    def test_restart_behavior(self):
        # Check if restart clears the buffers after they are full
        aa = Anderson(m=2, restart=True)
        x = np.array([1.0, 2.0])
        g = np.array([1.5, 2.5])
        # step 1
        aa.step(x, g)
        # step 2 (fills buffer)
        x, res = aa.step(g, g+0.1)
        # step 3 (trigger restart)
        x, res = aa.step(x, x+0.2)
        self.assertEqual(len(aa.dx_buffer), 0)
        self.assertEqual(len(aa.dr_buffer), 0)


class TestNewtonAnderson(unittest.TestCase):
    """
    Extended tests for the 'NewtonAnderson' class.
    """

    def test_init(self):
        m = 5
        naa = NewtonAnderson(m)
        self.assertEqual(naa.m, m)
        self.assertFalse(naa.restart)
        self.assertEqual(len(naa.dx_buffer), 0)
        self.assertEqual(len(naa.dr_buffer), 0)

    def test_step_no_jacobian(self):
        naa = NewtonAnderson(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 3.0])
        result, residual = naa.step(x, g)
        # same behavior as Anderson if no jac
        np.testing.assert_array_equal(result, g)
        self.assertAlmostEqual(residual, np.linalg.norm(g - x))

    def test_step_with_jacobian_scalar(self):
        # Solve a scalar equation quickly:
        # Suppose g(x)=cos(x), jac= -sin(x)
        naa = NewtonAnderson(2)
        x = 0.0
        g = np.cos(x)
        j = -np.sin(x)
        result, residual = naa.step(x, g, j)
        # Just ensure it runs without error and returns valid result
        self.assertTrue(np.isscalar(result))
        self.assertTrue(np.isscalar(residual))

    def test_step_with_jacobian_vector(self):
        naa = NewtonAnderson(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 4.0])
        # jac of g(x)= [2,0;0,2] for a trivial linear system
        jac = np.array([[2.0, 0.0],[0.0, 2.0]])
        result, residual = naa.step(x, g, jac)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(residual >= 0)

    def test_solve_scalar_equation(self):
        # Solve x = cos(x)
        naa = NewtonAnderson(m=5)
        def func_scalar(x):
            return np.cos(x) - x
        def jac_scalar(x):
            return -np.sin(x) - 1.0  # derivative of (cos(x)-x)
        x0 = np.array([0.0])
        x_sol, res, iters = naa.solve(func_scalar, x0, jac=jac_scalar, iterations_max=200, tolerance=1e-10)
        self.assertAlmostEqual(x_sol[0], 0.7390851332151607, places=7)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)