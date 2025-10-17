########################################################################################
##
##                                  TESTS FOR
##                            'blocks.comparator.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.comparator import Comparator
from pathsim.events.zerocrossing import ZeroCrossing


# TESTS ================================================================================

class TestComparator(unittest.TestCase):
    """
    Test the implementation of the 'Comparator' block class
    """

    def test_init(self):
        """Test initialization with default and custom parameters"""

        # Default initialization
        C = Comparator()
        self.assertEqual(C.threshold, 0)
        self.assertEqual(C.tolerance, 1e-4)
        self.assertEqual(C.span, [-1, 1])
        self.assertEqual(len(C.events), 1)
        self.assertIsInstance(C.events[0], ZeroCrossing)

        # Custom initialization
        C = Comparator(threshold=5.0, tolerance=1e-6, span=[0, 10])
        self.assertEqual(C.threshold, 5.0)
        self.assertEqual(C.tolerance, 1e-6)
        self.assertEqual(C.span, [0, 10])


    def test_len(self):
        """Test the length of the comparator block"""

        C = Comparator()
        self.assertEqual(len(C), 1)


    def test_update_above_threshold(self):
        """Test output when input is above threshold"""

        C = Comparator(threshold=0, span=[-1, 1])

        # Input above threshold
        C.inputs[0] = 5.0
        C.update(0)
        self.assertEqual(C.outputs[0], 1)

        # Input exactly at threshold
        C.inputs[0] = 0.0
        C.update(0)
        self.assertEqual(C.outputs[0], 1)


    def test_update_below_threshold(self):
        """Test output when input is below threshold"""

        C = Comparator(threshold=0, span=[-1, 1])

        # Input below threshold
        C.inputs[0] = -5.0
        C.update(0)
        self.assertEqual(C.outputs[0], -1)


    def test_custom_threshold(self):
        """Test with custom threshold values"""

        C = Comparator(threshold=10.0, span=[-1, 1])

        C.inputs[0] = 15.0
        C.update(0)
        self.assertEqual(C.outputs[0], 1)

        C.inputs[0] = 5.0
        C.update(0)
        self.assertEqual(C.outputs[0], -1)


    def test_custom_span(self):
        """Test with custom output span"""

        C = Comparator(threshold=0, span=[0, 10])

        # Above threshold
        C.inputs[0] = 5.0
        C.update(0)
        self.assertEqual(C.outputs[0], 10)

        # Below threshold
        C.inputs[0] = -5.0
        C.update(0)
        self.assertEqual(C.outputs[0], 0)


    def test_event_function(self):
        """Test that the zero-crossing event function works correctly"""

        C = Comparator(threshold=5.0)

        # Set input and check event function evaluates correctly
        C.inputs[0] = 10.0
        event_val = C.events[0].func_evt(0)
        self.assertEqual(event_val, 5.0)  # 10 - 5

        C.inputs[0] = 3.0
        event_val = C.events[0].func_evt(0)
        self.assertEqual(event_val, -2.0)  # 3 - 5


    def test_asymmetric_span(self):
        """Test with asymmetric span values"""

        C = Comparator(threshold=0, span=[-5, 3])

        C.inputs[0] = 1.0
        C.update(0)
        self.assertEqual(C.outputs[0], 3)

        C.inputs[0] = -1.0
        C.update(0)
        self.assertEqual(C.outputs[0], -5)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
