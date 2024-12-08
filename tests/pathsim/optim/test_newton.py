########################################################################################
##
##                                     TESTS FOR 
##                                 'optim/newton.py'
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np


from pathsim.optim.newton import (
    NewtonRaphsonAD, 
    GaussNewtonAD, 
    LevenbergMarquardtAD
    )


class TestNewtonRaphsonAD(unittest.TestCase):
    """
    Tests for the 'NewtonRaphsonAD' class from newton.py.
    """

    def test_init(self):
        nr = NewtonRaphsonAD()
        self.assertIsNone(nr.x)

    def test_reset(self):
        nr = NewtonRaphsonAD()
        nr.x = np.array([1.0, 2.0])
        nr.reset()
        self.assertIsNone(nr.x)

    def test_solve_scalar_equation(self):
        # Solve x=cos(x)
        def func(x):
            return np.cos(x) - x
        nr = NewtonRaphsonAD()
        x0 = np.array([0.0])
        x_sol, res, iters = nr.solve(func, x0, iterations_max=200, tolerance=1e-10)
        self.assertAlmostEqual(x_sol[0], 0.7390851332151607, places=7)
        self.assertTrue(res < 1e-10)

    def test_solve_vector_equation(self):
        # Solve a simple linear system:
        # f(x) = A x - b = 0, A= [[2,0],[0,2]], b=[2,2]
        # solution x=[1,1]
        def func(x):
            A = np.array([[2,0],[0,2]])
            b = np.array([2,2])
            return A.dot(x) - b
        nr = NewtonRaphsonAD()
        x0 = np.array([0.0,0.0])
        x_sol, res, iters = nr.solve(func, x0, iterations_max=50, tolerance=1e-12)
        np.testing.assert_allclose(x_sol, [1.0,1.0], atol=1e-9)

    def test_singular_jacobian_fallback(self):
        # This function has zero derivative: f(x)=x^3 constant derivative at zero is 0 for x=0
        # The solver should fallback to a fixed-point step or handle gracefully.
        def func(x):
            return x**3
        nr = NewtonRaphsonAD()
        x0 = np.array([0.0])
        # This won't converge easily, but we just want to ensure no error
        try:
            nr.solve(func, x0, iterations_max=10, tolerance=1e-6)
        except RuntimeError:
            # It's expected to not converge
            pass


class TestGaussNewtonAD(unittest.TestCase):
    """
    Tests for the 'GaussNewtonAD' class.
    """

    def test_solve_scalar_least_squares(self):
        # Solve a least squares problem: We want x s.t. f(x)= (x-3)
        # Minimizing (x-3)^2 leads to x=3 as solution for f(x)=0
        def func(x):
            return x - 3.0
        gn = GaussNewtonAD()
        x0 = np.array([0.0])
        x_sol, res, iters = gn.solve(func, x0, iterations_max=50, tolerance=1e-10)
        self.assertAlmostEqual(x_sol[0], 3.0, places=9)

    def test_solve_vector_least_squares(self):
        # f(x) = [x_1 - 1, x_2 - 2], solution: x=[1,2]
        def func(x):
            return x - np.array([1.0,2.0])
        gn = GaussNewtonAD()
        x0 = np.array([10.0, 10.0])
        x_sol, res, iters = gn.solve(func, x0)
        np.testing.assert_allclose(x_sol, [1.0,2.0], atol=1e-8)


class TestLevenbergMarquardtAD(unittest.TestCase):
    """
    Tests for the 'LevenbergMarquardtAD' class.
    """

    def test_reset(self):
        lm = LevenbergMarquardtAD()
        lm.x = np.array([1.0])
        lm.cost = 10.0
        lm.alpha = 1e-2
        lm.reset()
        self.assertIsNone(lm.x)
        self.assertIsNone(lm.cost)
        self.assertEqual(lm.alpha, 1e-6)

    def test_solve_scalar_nonlinear_least_squares(self):
        # Try to solve: minimize (sin(x))^2 = 0 => sin(x)=0 => x=0 is a solution
        # f(x)=sin(x), want f(x)=0
        def func(x):
            return np.sin(x)
        lm = LevenbergMarquardtAD()
        x0 = np.array([3.0]) # start from pi
        x_sol, res, iters = lm.solve(func, x0, iterations_max=200, tolerance=1e-10)
        # Nearest zero to pi is pi, but we might also find zero near 0 if step large.
        # LM tries a gradient-based approach. It's likely to find something close to pi (3.14159...) or 0.
        # We'll accept a solution near pi
        close_to_zero = abs(np.sin(x_sol[0])) < 1e-7
        self.assertTrue(close_to_zero)

    def test_solve_vector_problem(self):
        # Solve: f(x)= [ (x_1-2), (x_2-5) ], we want x=[2,5]
        def func(x):
            return np.array([x[0]-2.0, x[1]-5.0])
        lm = LevenbergMarquardtAD()
        x0 = np.array([0.0,0.0])
        x_sol, res, iters = lm.solve(func, x0, iterations_max=200, tolerance=1e-12)
        np.testing.assert_allclose(x_sol, [2.0,5.0], atol=1e-9)

    def test_alpha_adjustment(self):
        # We will test if alpha parameter gets adjusted when cost changes
        lm = LevenbergMarquardtAD()
        # fake a step
        lm.x = np.array([1.0])
        lm._adjust_params(10.0)  # cost=10
        old_alpha = lm.alpha
        lm._adjust_params(5.0)   # cost decreased
        self.assertTrue(lm.alpha < old_alpha) # alpha should shrink if cost improved
        old_alpha = lm.alpha
        lm._adjust_params(20.0)  # cost increased
        self.assertTrue(lm.alpha > old_alpha) # alpha should grow if cost got worse



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)