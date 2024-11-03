########################################################################################
##
##                                  TESTS FOR 
##                               'diff.value.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.diff.value import Value, Parameter


# TESTS ================================================================================

class TestValue(unittest.TestCase):
    """
    Test the implementation of the 'Value' class that 
    overloads operations for automatic differentiation.
    """

    def test_init(self):

        #test default initialization
        v = Value()
        self.assertEqual(v.val, 0.0)
        self.assertEqual(v.grad, {v: 1.0})

        #test special initialization
        v = Value(3.2)
        self.assertEqual(v.val, 3.2)
        self.assertEqual(v.grad, {v: 1.0})


    def test_d(self):
        #test derivative retrieval
        v = Value(0.01)
        w = Value(3.2)

        self.assertEqual(v.d(v), 1.0)
        self.assertEqual(v.d(w), 0.0)
        self.assertEqual(w.d(v), 0.0)
        self.assertEqual(w.d(w), 1.0)


    def test_add(self):
        v = Value(0.01)
        w = Value(3.2)

        # Test Value + Value
        z = v + w
        self.assertEqual(z.val, 3.21)
        self.assertEqual(z.d(v), 1.0)
        self.assertEqual(z.d(w), 1.0)

        # Test Value + scalar
        z = v + 2
        self.assertEqual(z.val, 2.01)
        self.assertEqual(z.d(v), 1.0)

        # Test scalar + Value
        z = 2 + v
        self.assertEqual(z.val, 2.01)
        self.assertEqual(z.d(v), 1.0)

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
        self.assertEqual(z.d(v), 1.0)
        self.assertEqual(z.d(w), -1.0)

        # Test Value - scalar
        z = v - 2
        self.assertEqual(z.val, 3)
        self.assertEqual(z.d(v), 1.0)

        # Test scalar - Value
        z = 10 - v
        self.assertEqual(z.val, 5)
        self.assertEqual(z.d(v), -1.0)


    def test_mul(self):
        v = Value(0.1)
        w = Value(3.2)

        # Test Value * Value
        z = v * w
        self.assertAlmostEqual(z.val, 0.32)
        self.assertEqual(z.d(v), 3.2)
        self.assertEqual(z.d(w), 0.1)

        # Test Value * scalar
        z = v * 2
        self.assertEqual(z.val, 0.2)
        self.assertEqual(z.d(v), 2)

        # Test scalar * Value
        z = 2 * v
        self.assertEqual(z.val, 0.2)
        self.assertEqual(z.d(v), 2)

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
        self.assertEqual(z.d(v), 0.5)
        self.assertEqual(z.d(w), -1.0)

        # Test Value / scalar
        z = v / 2
        self.assertEqual(z.val, 2)
        self.assertEqual(z.d(v), 0.5)

        # Test scalar / Value
        z = 8 / v
        self.assertEqual(z.val, 2)
        self.assertEqual(z.d(v), -0.5)


    def test_pow(self):
        v = Value(2)
        w = Value(3)

        # Test Value ** Value
        z = v ** w
        self.assertEqual(z.val, 8)
        self.assertEqual(z.d(v), 12.0)
        self.assertEqual(z.d(w), 8 * np.log(2))

        # Test Value ** scalar
        z = v ** 3
        self.assertEqual(z.val, 8)
        self.assertEqual(z.d(v), 12.0)

        # Test scalar ** Value
        z = 2 ** w
        self.assertEqual(z.val, 8)
        self.assertEqual(z.d(w), 8 * np.log(2))


    def test_unary_ops(self):
        v = Value(-3)

        # Test negation
        z = -v
        self.assertEqual(z.val, 3)
        self.assertEqual(z.d(v), -1.0)

        # Test absolute value
        z = abs(v)
        self.assertEqual(z.val, 3)
        self.assertEqual(z.d(v), -1.0)


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

        self.assertEqual(real_part.d(v), np.real(1))
        self.assertEqual(imag_part.d(v), np.imag(1))


    def test_numpy_functions(self):
        v = Value(0.5)

        # Test np.sin
        z = np.sin(v)
        self.assertAlmostEqual(z.val, np.sin(0.5))
        self.assertAlmostEqual(z.d(v), np.cos(0.5))

        # Test np.exp
        z = np.exp(v)
        self.assertAlmostEqual(z.val, np.exp(0.5))
        self.assertAlmostEqual(z.d(v), np.exp(0.5))

        # Test np.log
        v = Value(2)
        z = np.log(v)
        self.assertAlmostEqual(z.val, np.log(2))
        self.assertAlmostEqual(z.d(v), 0.5)


    def test_chain_rule(self):
        v = Value(0.5)
        w = Value(2.0)

        # Function: z = sin(v * w)
        z = (v * w).sin()
        self.assertAlmostEqual(z.val, np.sin(1.0))
        dz_dv = z.d(v)
        dz_dw = z.d(w)
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
            self.assertEqual(zi.d(v), a)
   
   
class TestParameter(unittest.TestCase):
    """
    Test the implementation of the 'Parameter' class that 
    inherits from the 'Value' class but adds ranges and 
    randomizer for optimization.
    """

    def test_init(self):
        # Test default initialization
        p = Parameter()
        self.assertEqual(p.val, 0.0)
        self.assertEqual(p.min_val, 0)
        self.assertEqual(p.max_val, 1)
        self.assertEqual(p.grad, {p: 1.0})

        # Test initialization with specific values
        p = Parameter(val=5.0, min_val=1.0, max_val=10.0)
        self.assertEqual(p.val, 5.0)
        self.assertEqual(p.min_val, 1.0)
        self.assertEqual(p.max_val, 10.0)

    def test_shuffle(self):
        p = Parameter(val=5.0, min_val=1.0, max_val=10.0)

        # Shuffle the parameter multiple times and check if it stays within bounds
        for _ in range(100):
            p.shuffle()
            self.assertGreaterEqual(p.val, p.min_val)
            self.assertLessEqual(p.val, p.max_val)

    def test_inheritance(self):
        # Test that Parameter behaves like Value
        p = Parameter(2)
        w = Value(3)

        z = p + w
        self.assertEqual(z.val, 5)
        self.assertEqual(z.d(p), 1.0)
        self.assertEqual(z.d(w), 1.0)

        z = p * w
        self.assertEqual(z.val, 6)
        self.assertEqual(z.d(p), 3)
        self.assertEqual(z.d(w), 2)

    def test_gradient(self):
        # Test gradient propagation through Parameter
        p = Parameter(2)
        z = p ** 2
        self.assertEqual(z.val, 4)
        self.assertEqual(z.d(p), 4)

    def test_bounds(self):
        # Test that min_val and max_val are attributes and can be used
        p = Parameter(5, min_val=0, max_val=10)
        self.assertEqual(p.min_val, 0)
        self.assertEqual(p.max_val, 10)

        # Test that shuffle respects new bounds
        p.min_val = -5
        p.max_val = 5
        p.shuffle()
        self.assertGreaterEqual(p.val, -5)
        self.assertLessEqual(p.val, 5)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
