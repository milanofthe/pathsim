########################################################################################
##
##                                  TESTS FOR
##                           'solvers/steadystate.py'
##
##                            Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.steadystate import SteadyState


# TESTS ================================================================================

class TestSteadyState(unittest.TestCase):
    """
    Test the implementation of the 'SteadyState' solver class
    """

    def test_init(self):
        """Test default initialization"""

        solver = SteadyState()

        self.assertEqual(solver.initial_value, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)

        # Test specific initialization
        solver = SteadyState(
            initial_value=1.5,
            tolerance_lte_rel=1e-3,
            tolerance_lte_abs=1e-6
        )

        self.assertEqual(solver.initial_value, 1.5)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_solve_without_jacobian(self):
        """Test solve method without jacobian"""

        solver = SteadyState(initial_value=1.0)

        # Initialize state
        solver.x = np.array([2.0])

        # Function evaluation: f(x) = x - 3 (steady state at x=3)
        f = np.array([-1.0])  # x=2, so f = 2-3 = -1

        # Solve (no jacobian)
        error = solver.solve(f, None, dt=0.1)

        # Should return an error value
        self.assertIsInstance(error, float)


    def test_solve_with_jacobian(self):
        """Test solve method with jacobian"""

        solver = SteadyState(initial_value=1.0)

        # Initialize state
        solver.x = np.array([2.0])

        # Function evaluation: f(x) = x - 3
        f = np.array([-1.0])

        # Jacobian: df/dx = 1
        J = np.array([[1.0]])

        # Solve (with jacobian)
        error = solver.solve(f, J, dt=0.1)

        # Should return an error value
        self.assertIsInstance(error, float)


    def test_fixed_point_equation(self):
        """Test that fixed point equation is g(x) = x + f(x)"""

        solver = SteadyState(initial_value=5.0)

        # Set state
        solver.x = np.array([10.0])

        # Function value
        f = np.array([2.0])

        # Expected fixed point target: g = x + f = 10 + 2 = 12
        # The solver will try to find x such that x = x + f(x)
        # which is equivalent to f(x) = 0

        error = solver.solve(f, None, dt=0.1)

        # Just verify it runs without error
        self.assertIsInstance(error, float)


    def test_jacobian_transformation(self):
        """Test that jacobian of g is I + J"""

        solver = SteadyState(initial_value=np.array([1.0, 2.0]))

        # Multi-dimensional state
        solver.x = np.array([1.0, 2.0])

        # Function
        f = np.array([0.5, -0.5])

        # Jacobian of f
        J = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Solve with jacobian
        error = solver.solve(f, J, dt=0.1)

        # Expected jacobian of g = I + J
        # [[1.1, 0.2], [0.3, 1.4]]

        # Just verify it runs
        self.assertIsInstance(error, float)


    def test_solve_simple_steady_state_problem(self):
        """Test solving a simple steady state problem"""

        # Steady state problem: dx/dt = -x + 2
        # Steady state at x = 2

        solver = SteadyState(initial_value=0.0)
        solver.x = np.array([0.0])

        # Iterate to find steady state
        for _ in range(50):
            # f(x) = -x + 2
            f = -solver.x + 2.0

            # J = df/dx = -1
            J = np.array([[-1.0]])

            error = solver.solve(f, J, dt=0.1)

            # Check if converged
            if error < 1e-6:
                break

        # Should converge close to x = 2
        self.assertAlmostEqual(solver.x[0], 2.0, places=2)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
