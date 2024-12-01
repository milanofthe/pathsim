########################################################################################
##
##                                  TESTS FOR 
##                               'optim/value.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.optim.value import Value, jac, der, autojac


# TESTS ================================================================================

class TestAutojac(unittest.TestCase):
    """
    Test the 'autojac' decorator 
    """

    def test_autojac_decorator(self):
        """Test the autojac decorator for a function with multiple outputs."""
        @autojac
        def my_func(a, b):
            return [a + b, a * b, np.exp(a - b)]

        vals = (Value(1.0), Value(2.0))
        f_vals, jacobian = my_func(*vals)
        expected_vals = [1.0 + 2.0, 1.0 * 2.0, np.exp(1.0 - 2.0)]
        self.assertTrue(np.allclose(f_vals, expected_vals))

        # Manually compute derivatives
        df1_da = 1.0
        df1_db = 1.0

        df2_da = vals[1].val
        df2_db = vals[0].val

        df3_da = np.exp(1.0 - 2.0)
        df3_db = -np.exp(1.0 - 2.0)

        expected_jacobian = np.array([
            [df1_da, df1_db],
            [df2_da, df2_db],
            [df3_da, df3_db]
        ])
        self.assertTrue(np.allclose(jacobian, expected_jacobian))


    def test_autojac_with_multiple_outputs(self):
        """Test the autojac decorator with a function that returns multiple outputs as a tuple."""
        @autojac
        def func(a, b):
            return a + b, a * b, np.sin(a * b)

        x = Value(1.0)
        y = Value(2.0)
        f_vals, jacobian = func(x, y)

        # Expected values
        expected_vals = (x.val + y.val, x.val * y.val, np.sin(x.val * y.val))
        self.assertEqual(f_vals[0], expected_vals[0])
        self.assertEqual(f_vals[1], expected_vals[1])
        self.assertEqual(f_vals[2], expected_vals[2])

        # Expected Jacobian
        df1_da = 1.0
        df1_db = 1.0

        df2_da = y.val
        df2_db = x.val

        df3_da = y.val * np.cos(x.val * y.val)
        df3_db = x.val * np.cos(x.val * y.val)

        expected_jacobian = np.array([
            [df1_da, df1_db],
            [df2_da, df2_db],
            [df3_da, df3_db]
        ])
        self.assertTrue(np.allclose(jacobian, expected_jacobian))











class TestJacDer(unittest.TestCase):
    """
    Test the 'jac' and 'der' functions that operate on the 'Value' class 
    """

    def test_der_function(self):
        """Test the der function for computing derivatives with respect to a Value."""
        # Single Value object
        x = Value(2.0)
        f = [x ** 2, np.sin(x), x * 3]
        derivatives = der(f, x)
        expected_derivatives = np.array([4.0, np.cos(2.0), 3.0])
        self.assertTrue(np.allclose(derivatives, expected_derivatives))

        # Multiple Value objects in an array
        x = Value(1.0)
        y = Value(2.0)
        f = np.array([x + y, x * y, np.exp(x)])
        derivatives_x = der(f, x)
        expected_derivatives_x = np.array([1.0, y.val, np.exp(1.0)])
        self.assertTrue(np.allclose(derivatives_x, expected_derivatives_x))

        derivatives_y = der(f, y)
        expected_derivatives_y = np.array([1.0, x.val, 0.0])
        self.assertTrue(np.allclose(derivatives_y, expected_derivatives_y))


    def test_jac_function(self):
        """Test the jac function for computing Jacobians with respect to multiple Values."""
        x = Value(1.0)
        y = Value(2.0)
        z = Value(3.0)
        f = [x + y + z, x * y * z, np.sin(x) * np.cos(y) * np.exp(z)]
        variables = [x, y, z]
        jacobian = jac(f, variables)
        # Manually compute derivatives
        df1_dx = 1.0
        df1_dy = 1.0
        df1_dz = 1.0

        df2_dx = y.val * z.val
        df2_dy = x.val * z.val
        df2_dz = x.val * y.val

        df3_dx = np.cos(x.val) * np.cos(y.val) * np.exp(z.val)
        df3_dy = -np.sin(x.val) * np.sin(y.val) * np.exp(z.val)
        df3_dz = np.sin(x.val) * np.cos(y.val) * np.exp(z.val)

        expected_jacobian = np.array([
            [df1_dx, df1_dy, df1_dz],
            [df2_dx, df2_dy, df2_dz],
            [df3_dx, df3_dy, df3_dz]
        ])
        self.assertTrue(np.allclose(jacobian, expected_jacobian))


    def test_der_and_jac_with_scalars_and_arrays(self):
        """Test der and jac functions with a mix of scalar and array inputs."""
        x = Value(1.0)
        y = Value(2.0)
        f = np.array([x * y, x + y, x / y])

        # Compute derivative with respect to x
        derivatives_x = der(f, x)
        expected_derivatives_x = np.array([y.val, 1.0, 1 / y.val])
        self.assertTrue(np.allclose(derivatives_x, expected_derivatives_x))

        # Compute derivative with respect to y
        derivatives_y = der(f, y)
        expected_derivatives_y = np.array([x.val, 1.0, -x.val / y.val**2])
        self.assertTrue(np.allclose(derivatives_y, expected_derivatives_y))

        # Compute Jacobian
        variables = [x, y]
        jacobian = jac(f, variables)
        expected_jacobian = np.array([
            [y.val, x.val],
            [1.0, 1.0],
            [1 / y.val, -x.val / y.val**2]
        ])
        self.assertTrue(np.allclose(jacobian, expected_jacobian))


    def test_der_with_non_value_objects(self):
        """Test the der function when the function output contains non-Value objects."""
        x = Value(1.0)
        f = [x, 2 * x.val, x.val ** 2]
        derivatives = der(f, x)
        expected_derivatives = np.array([1.0, 0.0, 0.0])
        self.assertTrue(np.allclose(derivatives, expected_derivatives))


    def test_jac_with_non_value_objects(self):
        """Test the jac function when variables include non-Value objects."""
        x = Value(1.0)
        y = 2.0  # Not a Value object
        f = [x * y, x + y, np.sin(x)]
        variables = [x]
        jacobian = jac(f, variables)
        expected_jacobian = np.array([
            [y],
            [1.0],
            [np.cos(x.val)]
        ])
        self.assertTrue(np.allclose(jacobian, expected_jacobian))


    def test_der_with_list_input(self):
        """Test the der function when the function output is a list."""
        x = Value(2.0)
        f = [x ** 2, np.exp(x), np.log(x)]
        derivatives = der(f, x)
        expected_derivatives = np.array([4.0, np.exp(2.0), 1 / x.val])
        self.assertTrue(np.allclose(derivatives, expected_derivatives))


    def test_jac_with_list_of_variables(self):
        """Test the jac function with a list of variables."""
        x = Value(1.0)
        y = Value(2.0)
        z = Value(3.0)
        f = [x * y + z, x / y, np.sin(z)]
        variables = [x, y, z]
        jacobian = jac(f, variables)
        # Manually compute derivatives
        df1_dx = y.val
        df1_dy = x.val
        df1_dz = 1.0

        df2_dx = 1 / y.val
        df2_dy = -x.val / y.val**2
        df2_dz = 0.0

        df3_dx = 0.0
        df3_dy = 0.0
        df3_dz = np.cos(z.val)

        expected_jacobian = np.array([
            [df1_dx, df1_dy, df1_dz],
            [df2_dx, df2_dy, df2_dz],
            [df3_dx, df3_dy, df3_dz]
        ])
        self.assertTrue(np.allclose(jacobian, expected_jacobian))


    def test_der_with_scalar_output(self):
        """Test the der function when the function output is a scalar Value."""
        x = Value(2.0)
        f = x ** 3
        derivative = der(f, x)
        expected_derivative = 3 * x.val ** 2
        self.assertEqual(derivative, expected_derivative)


    def test_jac_with_scalar_output(self):
        """Test the jac function when the function output is a scalar Value."""
        x = Value(2.0)
        y = Value(3.0)
        f = x * y + np.sin(x)
        jacobian = jac(f, [x, y])
        expected_jacobian = np.array([
            y.val + np.cos(x.val),
            x.val
        ])
        self.assertTrue(np.allclose(jacobian, expected_jacobian))


    def test_der_with_non_value_variable(self):
        """Test the der function when variable is not a Value object."""
        x = Value(2.0)
        f = [x ** 2, np.sin(x)]
        with self.assertRaises(AttributeError):
            der(f, 2.0)  # Should raise AttributeError because 2.0 has no '_id' attribute


    def test_jac_with_non_value_variable(self):
        """Test the jac function when variables include non-Value objects."""
        x = Value(1.0)
        y = 2.0  # Not a Value object
        f = [x + y, x * y]
        with self.assertRaises(AttributeError):
            jac(f, [x, y])  # Should raise AttributeError because y has no '_id' attribute


    def test_der_with_mixed_function_output(self):
        """Test the der function when the function output contains mixed Value and non-Value objects."""
        x = Value(2.0)
        f = [x ** 2, 3.0, np.sin(x)]
        derivatives = der(f, x)
        expected_derivatives = np.array([4.0, 0.0, np.cos(2.0)])
        self.assertTrue(np.allclose(derivatives, expected_derivatives))


    def test_jac_with_mixed_function_output(self):
        """Test the jac function when the function output contains mixed Value and non-Value objects."""
        x = Value(2.0)
        y = Value(3.0)
        f = [x + y, 5.0, x * y]
        jacobian = jac(f, [x, y])
        expected_jacobian = np.array([
            [1.0, 1.0],
            [0.0, 0.0],
            [y.val, x.val]
        ])
        self.assertTrue(np.allclose(jacobian, expected_jacobian))


    def test_der_with_function_output_as_scalar(self):
        """Test the der function when the function output is a scalar Value."""
        x = Value(3.0)
        f = x ** 2
        derivative = der(f, x)
        expected_derivative = 6.0
        self.assertEqual(derivative, expected_derivative)


    def test_jac_with_function_output_as_scalar(self):
        """Test the jac function when the function output is a scalar Value."""
        x = Value(3.0)
        y = Value(4.0)
        f = x * y
        jacobian = jac(f, [x, y])
        expected_jacobian = np.array([y.val, x.val])
        self.assertTrue(np.allclose(jacobian, expected_jacobian))


    def test_der_with_multiple_variables(self):
        """Test the der function when computing derivatives with respect to multiple variables."""
        x = Value(1.0)
        y = Value(2.0)
        z = Value(3.0)
        f = [x * y * z]
        derivative_x = der(f, x)
        derivative_y = der(f, y)
        derivative_z = der(f, z)
        expected_derivative_x = y.val * z.val
        expected_derivative_y = x.val * z.val
        expected_derivative_z = x.val * y.val
        self.assertEqual(derivative_x, expected_derivative_x)
        self.assertEqual(derivative_y, expected_derivative_y)
        self.assertEqual(derivative_z, expected_derivative_z)











class TestValue(unittest.TestCase):
    """
    Test the implementation of the 'Value' class that 
    overloads operations for automatic differentiation.
    """

    def test_init(self):

        #test default initialization
        v = Value()
        self.assertEqual(v.val, 0.0)
        self.assertEqual(v.grad, {v._id: 1.0})

        #test special initialization
        v = Value(3.2)
        self.assertEqual(v.val, 3.2)
        self.assertEqual(v.grad, {v._id: 1.0})


    def test_d(self):
        #test derivative retrieval
        v = Value(0.01)
        w = Value(3.2)

        self.assertEqual(v(v), 1.0)
        self.assertEqual(v(w), 0.0)
        self.assertEqual(w(v), 0.0)
        self.assertEqual(w(w), 1.0)


    def test_add(self):
        v = Value(0.01)
        w = Value(3.2)

        # Test Value + Value
        z = v + w
        self.assertEqual(z.val, 3.21)
        self.assertEqual(z(v), 1.0)
        self.assertEqual(z(w), 1.0)

        # Test Value + scalar
        z = v + 2
        self.assertEqual(z.val, 2.01)
        self.assertEqual(z(v), 1.0)

        # Test scalar + Value
        z = 2 + v
        self.assertEqual(z.val, 2.01)
        self.assertEqual(z(v), 1.0)

        # Test Value + np.ndarray
        arr = np.array([1, 2, 3])
        z = v + arr
        expected_vals = np.array([v.val + 1, v.val + 2, v.val + 3])
        self.assertTrue(np.allclose([zi.val for zi in z], expected_vals))


    def test_sub(self):
        v = Value(5)
        w = Value(3)

        # Test Value - Value
        z = v - w
        self.assertEqual(z.val, 2)
        self.assertEqual(z(v), 1.0)
        self.assertEqual(z(w), -1.0)

        # Test Value - scalar
        z = v - 2
        self.assertEqual(z.val, 3)
        self.assertEqual(z(v), 1.0)

        # Test scalar - Value
        z = 10 - v
        self.assertEqual(z.val, 5)
        self.assertEqual(z(v), -1.0)


    def test_mul(self):
        v = Value(0.1)
        w = Value(3.2)

        # Test Value * Value
        z = v * w
        self.assertAlmostEqual(z.val, 0.32)
        self.assertEqual(z(v), 3.2)
        self.assertEqual(z(w), 0.1)

        # Test Value * scalar
        z = v * 2
        self.assertEqual(z.val, 0.2)
        self.assertEqual(z(v), 2)

        # Test scalar * Value
        z = 2 * v
        self.assertEqual(z.val, 0.2)
        self.assertEqual(z(v), 2)

        # Test Value * np.ndarray
        arr = np.array([1, 2, 3])
        z = v * arr
        expected = np.array([v.val * 1, v.val * 2, v.val * 3])
        self.assertTrue(np.allclose([zi.val for zi in z], expected))


    def test_div(self):
        v = Value(4)
        w = Value(2)

        # Test Value / Value
        z = v / w
        self.assertEqual(z.val, 2)
        self.assertEqual(z(v), 0.5)
        self.assertEqual(z(w), -1.0)

        # Test Value / scalar
        z = v / 2
        self.assertEqual(z.val, 2)
        self.assertEqual(z(v), 0.5)

        # Test scalar / Value
        z = 8 / v
        self.assertEqual(z.val, 2)
        self.assertEqual(z(v), -0.5)


    def test_pow(self):
        v = Value(2)
        w = Value(3)

        # Test Value ** Value
        z = v ** w
        self.assertEqual(z.val, 8)
        self.assertEqual(z(v), 12.0)
        self.assertEqual(z(w), 8 * np.log(2))

        # Test Value ** scalar
        z = v ** 3
        self.assertEqual(z.val, 8)
        self.assertEqual(z(v), 12.0)

        # Test scalar ** Value
        z = 2 ** w
        self.assertEqual(z.val, 8)
        self.assertEqual(z(w), 8 * np.log(2))


    def test_unary_ops(self):
        v = Value(-3)

        # Test negation
        z = -v
        self.assertEqual(z.val, 3)
        self.assertEqual(z(v), -1.0)

        # Test absolute value
        z = abs(v)
        self.assertEqual(z.val, 3)
        self.assertEqual(z(v), -1.0)


    def test_comparison_ops(self):
        v = Value(2)
        w = Value(3)

        self.assertTrue(v < w)
        self.assertFalse(v > w)
        self.assertTrue(v <= w)
        self.assertFalse(v >= w)
        self.assertFalse(v == w)
        self.assertTrue(v != w)


    def test_bool_cast(self):
        v = Value(0)
        w = Value(5)

        self.assertFalse(bool(v))
        self.assertTrue(bool(w))


    def test_type_cast(self):
        v = Value(3.7)

        self.assertEqual(int(v), 3)
        self.assertEqual(float(v), 3.7)


    def test_properties(self):
        v = Value(3 + 4j)

        real_part = v.real
        imag_part = v.imag

        self.assertEqual(real_part.val, 3)
        self.assertEqual(imag_part.val, 4)

        self.assertEqual(real_part(v), np.real(1))
        self.assertEqual(imag_part(v), np.imag(1))


    def test_numpy_functions(self):
        v = Value(0.5)

        # Test np.sin
        z = np.sin(v)
        self.assertAlmostEqual(z.val, np.sin(0.5))
        self.assertAlmostEqual(z(v), np.cos(0.5))

        # Test np.exp
        z = np.exp(v)
        self.assertAlmostEqual(z.val, np.exp(0.5))
        self.assertAlmostEqual(z(v), np.exp(0.5))

        # Test np.log
        v = Value(2)
        z = np.log(v)
        self.assertAlmostEqual(z.val, np.log(2))
        self.assertAlmostEqual(z(v), 0.5)


    def test_chain_rule(self):
        v = Value(0.5)
        w = Value(2.0)

        # Function: z = sin(v * w)
        z = (v * w).sin()
        self.assertAlmostEqual(z.val, np.sin(1.0))
        dz_dv = z(v)
        dz_dw = z(w)
        self.assertAlmostEqual(dz_dv, w.val * np.cos(1.0))
        self.assertAlmostEqual(dz_dw, v.val * np.cos(1.0))


    def test_array_operations(self):
        v = Value(2)
        arr = np.array([1, 2, 3])

        # Test addition
        z = v + arr
        expected_vals = np.array([3, 4, 5])
        self.assertTrue(np.allclose([zi.val for zi in z], expected_vals))

        # Test multiplication
        z = v * arr
        expected_vals = np.array([2, 4, 6])
        self.assertTrue(np.allclose([zi.val for zi in z], expected_vals))

        # Test gradient
        for zi, a in zip(z, arr):
            self.assertEqual(zi(v), a)


    def test_multiple_variables(self):
        """Test derivative computation with respect to multiple variables."""
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + np.sin(x)
        self.assertEqual(z.val, 2.0 * 3.0 + np.sin(2.0))
        self.assertAlmostEqual(z(x), y.val + np.cos(2.0))
        self.assertEqual(z(y), x.val)


    def test_zero_division(self):
        """Test division by zero handling."""
        x = Value(1.0)
        y = Value(0.0)
        with self.assertRaises(ZeroDivisionError):
            z = x / y
            _ = z.val  # Force evaluation

    
    def test_multidimensional_arrays(self):
        """Test operations with multidimensional numpy arrays."""
        x = Value(2.0)
        arr = np.array([[1, 2], [3, 4]])
        z = x * arr
        expected_vals = x.val * arr
        self.assertTrue(np.allclose([[zi.val for zi in row] for row in z], expected_vals))
        self.assertTrue(np.all([[zi(x) == arr[i, j] for j, zi in enumerate(row)] for i, row in enumerate(z)]))


    def test_numpy_power_function(self):
        """Test numpy power function overloading."""
        x = Value(2.0)
        z = np.power(x, 3)
        self.assertEqual(z.val, 8.0)
        self.assertEqual(z(x), 12.0)


    def test_numpy_log_functions(self):
        """Test numpy logarithmic functions."""
        x = Value(np.e)
        z = np.log(x)
        self.assertEqual(z.val, 1.0)
        self.assertEqual(z(x), 1 / x.val)

        z = np.log2(x)
        self.assertAlmostEqual(z.val, 1.44269504089)
        self.assertAlmostEqual(z(x), 1 / (x.val * np.log(2)))

        z = np.log10(x)
        self.assertAlmostEqual(z.val, 0.4342944819)
        self.assertAlmostEqual(z(x), 1 / (x.val * np.log(10)))


    def test_numpy_exponential_functions(self):
        """Test numpy exponential functions."""
        x = Value(1.0)
        z = np.exp(x)
        self.assertEqual(z.val, np.e)
        self.assertEqual(z(x), np.exp(1.0))

        z = np.exp2(x)
        self.assertEqual(z.val, 2.0)
        self.assertEqual(z(x), 2.0 * np.log(2))


    def test_underflow_handling(self):
        """Test handling of underflow in exponential functions."""
        x = Value(-1000)
        z = np.exp(x)
        self.assertEqual(z.val, 0.0)
        self.assertEqual(z(x), 0.0)


    def test_cbrt_function(self):
        """Test the cube root function."""
        x = Value(8.0)
        z = np.cbrt(x)
        self.assertAlmostEqual(z.val, 2.0)
        self.assertAlmostEqual(z(x), 1 / (3 * (np.cbrt(8.0) ** 2)))


    def test_arctrig_functions(self):
        """Test inverse trigonometric functions."""
        x = Value(0.5)
        z = np.arcsin(x)
        self.assertAlmostEqual(z.val, np.arcsin(0.5))
        self.assertAlmostEqual(z(x), 1 / np.sqrt(1 - 0.5 ** 2))

        z = np.arccos(x)
        self.assertAlmostEqual(z.val, np.arccos(0.5))
        self.assertAlmostEqual(z(x), -1 / np.sqrt(1 - 0.5 ** 2))

        z = np.arctan(x)
        self.assertAlmostEqual(z.val, np.arctan(0.5))
        self.assertAlmostEqual(z(x), 1 / (1 + 0.5 ** 2))


    def test_hyperbolic_functions(self):
        """Test hyperbolic functions and their derivatives."""
        x = Value(0.5)
        z = np.sinh(x)
        self.assertAlmostEqual(z.val, np.sinh(0.5))
        self.assertAlmostEqual(z(x), np.cosh(0.5))

        z = np.cosh(x)
        self.assertAlmostEqual(z.val, np.cosh(0.5))
        self.assertAlmostEqual(z(x), np.sinh(0.5))

        z = np.tanh(x)
        self.assertAlmostEqual(z.val, np.tanh(0.5))
        self.assertAlmostEqual(z(x), 1 - np.tanh(0.5) ** 2)


    def test_arcsinh_function(self):
        """Test the inverse hyperbolic sine function."""
        x = Value(1.0)
        z = np.arcsinh(x)
        self.assertAlmostEqual(z.val, np.arcsinh(1.0))
        self.assertAlmostEqual(z(x), 1 / np.sqrt(1 + x.val ** 2))

    def test_arccosh_function(self):
        """Test the inverse hyperbolic cosine function."""
        x = Value(2.0)
        z = np.arccosh(x)
        self.assertAlmostEqual(z.val, np.arccosh(2.0))
        self.assertAlmostEqual(z(x), 1 / (np.sqrt(x.val - 1) * np.sqrt(x.val + 1)))

    def test_arctanh_function(self):
        """Test the inverse hyperbolic tangent function."""
        x = Value(0.5)
        z = np.arctanh(x)
        self.assertAlmostEqual(z.val, np.arctanh(0.5))
        self.assertAlmostEqual(z(x), 1 / (1 - x.val ** 2))

    def test_abs_function(self):
        """Test the absolute value function."""
        x = Value(-3.0)
        z = np.abs(x)
        self.assertEqual(z.val, 3.0)
        self.assertEqual(z(x), -1.0)


    def test_array_value_initialization(self):
        """Test the Value.array class method for initializing arrays of Values."""
        arr = [1, 2, 3]
        values = Value.array(arr)
        self.assertTrue(all(isinstance(v, Value) for v in values))
        self.assertEqual([v.val for v in values], arr)


    def test_numeric_conversion(self):
        """Test the Value.numeric class method for converting Values to numerics."""
        arr = [Value(1), Value(2), Value(3)]
        numerics = Value.numeric(arr)
        self.assertEqual(list(numerics), [1, 2, 3])


    def test_inplace_operations(self):
        """Test in-place operations like +=, -=, *=, /=."""
        x = Value(2.0)
        y = Value(3.0)
        x += y
        self.assertEqual(x.val, 5.0)
        self.assertEqual(x(x), 1.0)
        self.assertEqual(x(y), 1.0)

        x = Value(2.0)
        x *= y
        self.assertEqual(x.val, 6.0)
        self.assertEqual(x(x), y.val)

        x = Value(6.0)
        x -= y
        self.assertEqual(x.val, 3.0)
        self.assertEqual(x(x), 1.0)
        self.assertEqual(x(y), -1.0)

        x = Value(6.0)
        x /= y
        self.assertEqual(x.val, 2.0)


    def test_custom_function(self):
        """Test a custom function using Value objects."""
        def custom_func(a, b):
            return a ** 2 + np.sin(b * a)

        x = Value(1.0)
        y = Value(0.5)
        z = custom_func(x, y)
        self.assertEqual(z.val, x.val ** 2 + np.sin(y.val * x.val))
        dz_dx = 2 * x.val + y.val * np.cos(y.val * x.val)
        dz_dy = x.val * np.cos(y.val * x.val)
        self.assertAlmostEqual(z(x), dz_dx)
        self.assertAlmostEqual(z(y), dz_dy)


    def test_log1p_function(self):
        """Test the np.log1p function."""
        x = Value(1e-6)
        z = np.log1p(x)
        self.assertAlmostEqual(z.val, np.log1p(1e-6))
        self.assertAlmostEqual(z(x), 1 / (1 + x.val))


    def test_expm1_function(self):
        """Test the np.expm1 function."""
        x = Value(1e-6)
        z = np.expm1(x)
        self.assertAlmostEqual(z.val, np.expm1(1e-6))
        self.assertAlmostEqual(z(x), np.exp(x.val))


    def test_square_function(self):
        """Test the np.square function."""
        x = Value(3.0)
        z = np.square(x)
        self.assertEqual(z.val, 9.0)
        self.assertEqual(z(x), 6.0)


    def test_sqrt_function(self):
        """Test the np.sqrt function."""
        x = Value(4.0)
        z = np.sqrt(x)
        self.assertEqual(z.val, 2.0)
        self.assertEqual(z(x), 0.25)


    def test_invalid_operations(self):
        """Test operations that should fail."""
        x = Value(1.0)
        with self.assertRaises(AttributeError):
            z = x.non_existent_method()


    def test_repr_method(self):
        """Test the __repr__ method for readability."""
        x = Value(3.0)
        expected_repr = f"Value(val=3.0, grad={{x._id: 1.0}})"
        self.assertIn("Value(val=3.0", repr(x))


    def test_value_array_classmethod(self):
        """Test the Value.array classmethod with various inputs."""
        # Test with a list of numbers
        input_list = [1, 2, 3]
        values = Value.array(input_list)
        self.assertTrue(all(isinstance(v, Value) for v in values))
        self.assertEqual([v.val for v in values], input_list)
        self.assertTrue(all(len(v.grad) == 1 for v in values))

        # Test with a numpy array
        input_array = np.array([4, 5, 6])
        values = Value.array(input_array)
        self.assertTrue(isinstance(values, np.ndarray))
        self.assertTrue(all(isinstance(v, Value) for v in values))
        self.assertEqual([v.val for v in values], list(input_array))

        # Test with a mixed list
        mixed_list = [Value(7), 8, Value(9)]
        values = Value.array(mixed_list)
        self.assertEqual([v.val for v in values], [7, 8, 9])

        # Test with nested arrays
        nested_list = [[1, 2], [3, 4]]
        values = Value.array(nested_list)
        self.assertEqual(values.shape, (2, 2))
        self.assertTrue(isinstance(values[0, 0], Value))
        self.assertEqual(values[0, 0].val, 1)


    def test_value_numeric_classmethod(self):
        """Test the Value.numeric classmethod with various inputs."""
        # Test with a list of Value objects
        values = [Value(1), Value(2), Value(3)]
        numerics = Value.numeric(values)
        self.assertTrue(np.array_equal(numerics, np.array([1, 2, 3])))

        # Test with a numpy array of Value objects
        values_array = np.array([Value(4), Value(5), Value(6)])
        numerics = Value.numeric(values_array)
        self.assertTrue(np.array_equal(numerics, np.array([4, 5, 6])))

        # Test with a mixed list
        mixed_list = [Value(7), 8, Value(9)]
        numerics = Value.numeric(mixed_list)
        self.assertEqual(list(numerics), [7, 8, 9])

        # Test with nested arrays
        nested_values = np.array([[Value(1), Value(2)], [Value(3), Value(4)]])
        numerics = Value.numeric(nested_values)
        self.assertTrue(np.array_equal(numerics, np.array([[1, 2], [3, 4]])))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
