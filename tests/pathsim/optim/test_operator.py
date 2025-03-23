########################################################################################
##
##                                  TESTS FOR 
##                               'optim/operator.py'
##
##                              Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.optim.operator import Operator, DynamicOperator


# TESTS ================================================================================

class TestOperator(unittest.TestCase):
    """
    Test the 'Operator' class for function evaluation and linearization
    """

    def test_init(self):
        """Test initialization of Operator instances."""
        # Basic initialization
        def func(x):
            return x**2 + np.sin(x)
            
        op = Operator(func)
        self.assertEqual(op._func, func)
        self.assertIsNone(op._jac)
        self.assertIsNone(op.f0)
        self.assertIsNone(op.x0)
        self.assertIsNone(op.J)
        
        # Initialization with analytical jacobian
        def jac(x):
            return 2*x + np.cos(x)
            
        op = Operator(func, jac=jac)
        self.assertEqual(op._func, func)
        self.assertEqual(op._jac, jac)
        self.assertIsNone(op.f0)
        self.assertIsNone(op.x0)
        self.assertIsNone(op.J)
        
    def test_bool_cast(self):
        """Test boolean cast operation."""
        def func(x):
            return x
        
        op = Operator(func)
        self.assertTrue(bool(op))
        
    def test_call_direct(self):
        """Test direct function evaluation via __call__."""
        def func(x):
            return x**2 + 2*x + 1
            
        op = Operator(func)
        x = 2.0
        expected = x**2 + 2*x + 1
        
        result = op(x)
        self.assertEqual(result, expected)
        
    def test_call_linearized(self):
        """Test linearized function evaluation via __call__."""
        def func(x):
            return x**2 + 2*x + 1
            
        def jac(x):
            return 2*x + 2
            
        op = Operator(func, jac=jac)
        
        # Linearize at x0 = 2.0
        x0 = 2.0
        op.linearize(x0)
        
        # Evaluate at x = 3.0
        x = 3.0
        
        # Expected: f(x0) + J(x0) * (x - x0)
        # f(2.0) = 9, J(2.0) = 6, (3.0 - 2.0) = 1
        # 9 + 6*1 = 15
        expected = 9 + 6 * (x - x0)
        
        result = op(x)
        self.assertEqual(result, expected)
        
    def test_jac_analytical(self):
        """Test Jacobian computation with analytical function."""
        def func(x):
            return np.array([x**2, np.sin(x)])
            
        def analytical_jac(x):
            return np.array([2*x, np.cos(x)])
            
        op = Operator(func, jac=analytical_jac)
        
        x = 1.0
        expected_jac = np.array([2.0, np.cos(1.0)])
        
        result_jac = op.jac(x)
        self.assertTrue(np.allclose(result_jac, expected_jac))
        
    def test_jac_automatic(self):
        """Test Jacobian computation with automatic differentiation."""
        def func(x):
            return np.array([x**2, np.sin(x)])
            
        op = Operator(func)
        
        x = 1.0
        expected_jac = np.array([[2.0], [np.cos(1.0)]])
        result_jac = op.jac(x)

        self.assertTrue(np.allclose(result_jac, expected_jac))
        
    def test_linearize(self):
        """Test linearization of an operator."""
        def func(x):
            return x**2 + 2*x + 1
            
        def jac(x):
            return 2*x + 2
            
        op = Operator(func, jac=jac)
        
        x0 = 2.0
        op.linearize(x0)
        
        # Check stored values
        self.assertEqual(op.x0, x0)
        self.assertEqual(op.f0, func(x0))
        self.assertEqual(op.J, jac(x0))
        
    def test_reset(self):
        """Test resetting the linearization."""
        def func(x):
            return x**2 + 2*x + 1
            
        op = Operator(func)
        
        x0 = 2.0
        op.linearize(x0)
        
        # Should have values after linearization
        self.assertIsNotNone(op.x0)
        self.assertIsNotNone(op.f0)
        self.assertIsNotNone(op.J)
        
        op.reset()
        
        # Should be None after reset
        self.assertIsNone(op.x0)
        self.assertIsNone(op.f0)
        self.assertIsNone(op.J)
        
    def test_multi_input_output(self):
        """Test with multi-dimensional inputs and outputs."""
        def func(x):
            return np.array([x[0]**2 + x[1], np.sin(x[0]) * x[1]])
            
        def jac(x):
            return np.array([
                [2*x[0], 1],
                [np.cos(x[0])*x[1], np.sin(x[0])]
            ])
            
        op = Operator(func, jac=jac)
        
        x0 = np.array([1.0, 2.0])
        op.linearize(x0)
        
        x = np.array([1.5, 2.5])
        
        # Expected linearized result
        expected_f0 = func(x0)
        expected_J = jac(x0)
        expected = expected_f0 + np.dot(expected_J, (x - x0))
        
        result = op(x)
        self.assertTrue(np.allclose(result, expected))









class TestDynamicOperator(unittest.TestCase):
    """
    Test the 'DynamicOperator' class for dynamic system function evaluation and linearization
    """

    def test_init(self):
        """Test initialization of DynamicOperator instances."""
        # Basic initialization
        def func(x, u, t):
            return -0.5*x + 2*u
            
        op = DynamicOperator(func)
        self.assertEqual(op._func, func)
        self.assertIsNone(op._jac_x)
        self.assertIsNone(op._jac_u)
        self.assertIsNone(op.f0)
        self.assertIsNone(op.x0)
        self.assertIsNone(op.u0)
        self.assertIsNone(op.Jx)
        self.assertIsNone(op.Ju)
        
        # Initialization with analytical jacobians
        def jac_x(x, u, t):
            return -0.5
            
        def jac_u(x, u, t):
            return 2.0
            
        op = DynamicOperator(func, jac_x=jac_x, jac_u=jac_u)
        self.assertEqual(op._func, func)
        self.assertEqual(op._jac_x, jac_x)
        self.assertEqual(op._jac_u, jac_u)
        self.assertIsNone(op.f0)
        self.assertIsNone(op.x0)
        self.assertIsNone(op.u0)
        self.assertIsNone(op.Jx)
        self.assertIsNone(op.Ju)
        
    def test_bool_cast(self):
        """Test boolean cast operation."""
        def func(x, u, t):
            return x
        
        op = DynamicOperator(func)
        self.assertTrue(bool(op))
        
    def test_call_direct(self):
        """Test direct function evaluation via __call__."""
        def func(x, u, t):
            return -0.5*x + 2*u
            
        op = DynamicOperator(func)
        x = 2.0
        u = 1.0
        t = 0.0
        expected = -0.5*x + 2*u
        
        result = op(x, u, t)
        self.assertEqual(result, expected)
        
    def test_call_linearized(self):
        """Test linearized function evaluation via __call__."""
        def func(x, u, t):
            return -0.5*x + 2*u
            
        def jac_x(x, u, t):
            return -0.5
            
        def jac_u(x, u, t):
            return 2.0
            
        op = DynamicOperator(func, jac_x=jac_x, jac_u=jac_u)
        
        # Linearize at x0 = 2.0, u0 = 1.0, t = 0.0
        x0 = 2.0
        u0 = 1.0
        t0 = 0.0
        op.linearize(x0, u0, t0)
        
        # Evaluate at x = 3.0, u = 1.5, t = 0.1
        x = 3.0
        u = 1.5
        t = 0.1
        
        # Expected: f(x0, u0, t0) + Jx * (x - x0) + Ju * (u - u0)
        # f(2.0, 1.0, 0.0) = -1 + 2 = 1
        # Jx = -0.5, (3.0 - 2.0) = 1
        # Ju = 2.0, (1.5 - 1.0) = 0.5
        # 1 + (-0.5)*1 + 2.0*0.5 = 1 - 0.5 + 1 = 1.5
        expected = 1 + (-0.5) * (x - x0) + 2.0 * (u - u0)
        
        result = op(x, u, t)
        self.assertEqual(result, expected)
        
    def test_jac_x_analytical(self):
        """Test Jacobian computation for x with analytical function."""
        def func(x, u, t):
            return x**2 + u
            
        def analytical_jac_x(x, u, t):
            return 2*x
            
        op = DynamicOperator(func, jac_x=analytical_jac_x)
        
        x = 2.0
        u = 1.0
        t = 0.0
        expected_jac = 2*x
        
        result_jac = op.jac_x(x, u, t)
        self.assertEqual(result_jac, expected_jac)
        
    def test_jac_u_analytical(self):
        """Test Jacobian computation for u with analytical function."""
        def func(x, u, t):
            return x + u**2
            
        def analytical_jac_u(x, u, t):
            return 2*u
            
        op = DynamicOperator(func, jac_u=analytical_jac_u)
        
        x = 2.0
        u = 1.0
        t = 0.0
        expected_jac = 2*u
        
        result_jac = op.jac_u(x, u, t)
        self.assertEqual(result_jac, expected_jac)
        
    def test_jac_x_automatic(self):
        """Test Jacobian computation for x with automatic differentiation."""
        def func(x, u, t):
            return x**2 + u
            
        op = DynamicOperator(func)
        
        x = 2.0
        u = 1.0
        t = 0.0
        expected_jac = 4.0
        
        result_jac = op.jac_x(x, u, t)
        self.assertAlmostEqual(result_jac, expected_jac, places=6)
        
    def test_jac_u_automatic(self):
        """Test Jacobian computation for u with automatic differentiation."""
        def func(x, u, t):
            return x + u**2
            
        op = DynamicOperator(func)
        
        x = 2.0
        u = 1.0
        t = 0.0
        expected_jac = 2.0
        
        result_jac = op.jac_u(x, u, t)
        self.assertAlmostEqual(result_jac, expected_jac, places=6)
        
    def test_linearize(self):
        """Test linearization of a dynamic operator."""
        def func(x, u, t):
            return -0.5*x + 2*u
            
        def jac_x(x, u, t):
            return -0.5
            
        def jac_u(x, u, t):
            return 2.0
            
        op = DynamicOperator(func, jac_x=jac_x, jac_u=jac_u)
        
        x0 = 2.0
        u0 = 1.0
        t0 = 0.0
        op.linearize(x0, u0, t0)
        
        # Check stored values
        self.assertEqual(op.f0, func(x0, u0, t0))
        self.assertEqual(op.x0[0], x0)  # Check first element since it's atleast_1d
        self.assertEqual(op.u0[0], u0)  # Check first element since it's atleast_1d
        self.assertEqual(op.Jx, jac_x(x0, u0, t0))
        self.assertEqual(op.Ju, jac_u(x0, u0, t0))
        
    def test_reset(self):
        """Test resetting the linearization."""
        def func(x, u, t):
            return -0.5*x + 2*u
            
        op = DynamicOperator(func)
        
        x0 = 2.0
        u0 = 1.0
        t0 = 0.0
        op.linearize(x0, u0, t0)
        
        # Should have values after linearization
        self.assertIsNotNone(op.f0)
        self.assertIsNotNone(op.x0)
        self.assertIsNotNone(op.u0)
        self.assertIsNotNone(op.Jx)
        self.assertIsNotNone(op.Ju)
        
        op.reset()
        
        # Should be None after reset
        self.assertIsNone(op.f0)
        self.assertIsNone(op.x0)
        self.assertIsNone(op.u0)
        self.assertIsNone(op.Jx)
        self.assertIsNone(op.Ju)
        
    def test_multi_input_output(self):
        """Test with multi-dimensional inputs and outputs."""
        def func(x, u, t):
            return np.array([x[0] + x[1] + u[0], x[0]*x[1]*u[1]])
            
        def jac_x(x, u, t):
            return np.array([
                [1, 1],
                [x[1]*u[1], x[0]*u[1]]
            ])
            
        def jac_u(x, u, t):
            return np.array([
                [1, 0],
                [0, x[0]*x[1]]
            ])
            
        op = DynamicOperator(func, jac_x=jac_x, jac_u=jac_u)
        
        x0 = np.array([1.0, 2.0])
        u0 = np.array([0.5, 1.5])
        t0 = 0.0
        op.linearize(x0, u0, t0)
        
        x = np.array([1.5, 2.5])
        u = np.array([1.0, 2.0])
        t = 0.5
        
        # Expected linearized result
        expected_f0 = func(x0, u0, t0)
        expected_Jx = jac_x(x0, u0, t0)
        expected_Ju = jac_u(x0, u0, t0)
        expected = expected_f0 + np.dot(expected_Jx, (x - x0)) + np.dot(expected_Ju, (u - u0))
        
        result = op(x, u, t)
        self.assertTrue(np.allclose(result, expected))



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)