########################################################################################
##
##                                  TESTS FOR
##                               'blocks.fir.py'
##
##                              Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.fir import FIR
from pathsim.events.schedule import Schedule


# TESTS ================================================================================

class TestFIR(unittest.TestCase):
    """
    Test the implementation of the 'FIR' block class
    """

    def test_init(self):
        """Test initialization with default and custom parameters"""

        # Default initialization
        F = FIR()
        np.testing.assert_array_equal(F.coeffs, [1.0])
        self.assertEqual(F.T, 1)
        self.assertEqual(F.tau, 0)
        self.assertEqual(len(F.events), 1)
        self.assertIsInstance(F.events[0], Schedule)

        # Custom initialization
        coeffs = [0.5, 0.3, 0.2]
        F = FIR(coeffs=coeffs, T=0.1, tau=0.05)
        np.testing.assert_array_equal(F.coeffs, coeffs)
        self.assertEqual(F.T, 0.1)
        self.assertEqual(F.tau, 0.05)


    def test_len(self):
        """Test that FIR filter has no direct passthrough"""

        F = FIR()
        self.assertEqual(len(F), 0)


    def test_event_scheduling(self):
        """Test that the schedule event is configured correctly"""

        F = FIR(coeffs=[1.0], T=2.0, tau=0.5)

        # Check event properties
        event = F.events[0]
        self.assertIsInstance(event, Schedule)
        self.assertEqual(event.t_start, 0.5)
        self.assertEqual(event.t_period, 2.0)


    def test_buffer_initialization(self):
        """Test that buffer is initialized with correct size"""

        coeffs = [0.5, 0.3, 0.2]
        F = FIR(coeffs=coeffs)

        # Buffer should have length equal to number of coefficients
        self.assertEqual(len(F._buffer), 3)

        # Buffer should be initialized with zeros
        for val in F._buffer:
            self.assertEqual(val, 0.0)


    def test_simple_filter_passthrough(self):
        """Test FIR filter with single coefficient (passthrough)"""

        F = FIR(coeffs=[1.0])

        # Set input and trigger filter update
        F.inputs[0] = 5.0
        F.events[0].func_act(0)

        # Output should equal input for passthrough
        self.assertEqual(F.outputs[0], 5.0)


    def test_simple_filter_gain(self):
        """Test FIR filter with gain"""

        F = FIR(coeffs=[2.0])

        F.inputs[0] = 3.0
        F.events[0].func_act(0)

        # Output should be input * gain
        self.assertEqual(F.outputs[0], 6.0)


    def test_moving_average_filter(self):
        """Test FIR filter as moving average"""

        # Moving average of 3 samples
        F = FIR(coeffs=[1/3, 1/3, 1/3])

        # First sample
        F.inputs[0] = 3.0
        F.events[0].func_act(0)
        # Output: 3*1/3 + 0*1/3 + 0*1/3 = 1.0
        self.assertAlmostEqual(F.outputs[0], 1.0, places=10)

        # Second sample
        F.inputs[0] = 6.0
        F.events[0].func_act(1)
        # Output: 6*1/3 + 3*1/3 + 0*1/3 = 3.0
        self.assertAlmostEqual(F.outputs[0], 3.0, places=10)

        # Third sample
        F.inputs[0] = 9.0
        F.events[0].func_act(2)
        # Output: 9*1/3 + 6*1/3 + 3*1/3 = 6.0
        self.assertAlmostEqual(F.outputs[0], 6.0, places=10)


    def test_fir_with_memory(self):
        """Test FIR filter with multiple taps"""

        # Simple FIR: y[n] = x[n] + 0.5*x[n-1]
        F = FIR(coeffs=[1.0, 0.5])

        # First sample
        F.inputs[0] = 2.0
        F.events[0].func_act(0)
        # Output: 2.0*1.0 + 0.0*0.5 = 2.0
        self.assertEqual(F.outputs[0], 2.0)

        # Second sample
        F.inputs[0] = 4.0
        F.events[0].func_act(1)
        # Output: 4.0*1.0 + 2.0*0.5 = 5.0
        self.assertEqual(F.outputs[0], 5.0)

        # Third sample
        F.inputs[0] = 6.0
        F.events[0].func_act(2)
        # Output: 6.0*1.0 + 4.0*0.5 = 8.0
        self.assertEqual(F.outputs[0], 8.0)


    def test_reset(self):
        """Test that reset clears the buffer"""

        F = FIR(coeffs=[1.0, 0.5])

        # Add some data
        F.inputs[0] = 10.0
        F.events[0].func_act(0)
        self.assertEqual(F.outputs[0], 10.0)

        # Reset
        F.reset()

        # Buffer should be cleared
        for val in F._buffer:
            self.assertEqual(val, 0.0)


    def test_difference_filter(self):
        """Test FIR filter as difference operator"""

        # Difference: y[n] = x[n] - x[n-1]
        F = FIR(coeffs=[1.0, -1.0])

        # First sample
        F.inputs[0] = 5.0
        F.events[0].func_act(0)
        # Output: 5.0*1.0 + 0.0*(-1.0) = 5.0
        self.assertEqual(F.outputs[0], 5.0)

        # Second sample
        F.inputs[0] = 8.0
        F.events[0].func_act(1)
        # Output: 8.0*1.0 + 5.0*(-1.0) = 3.0
        self.assertEqual(F.outputs[0], 3.0)

        # Third sample
        F.inputs[0] = 10.0
        F.events[0].func_act(2)
        # Output: 10.0*1.0 + 8.0*(-1.0) = 2.0
        self.assertEqual(F.outputs[0], 2.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
