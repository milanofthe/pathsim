########################################################################################
##
##                                  TESTS FOR
##                             'blocks.counter.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.counter import Counter, CounterUp, CounterDown
from pathsim.events.zerocrossing import ZeroCrossing, ZeroCrossingUp, ZeroCrossingDown


# TESTS ================================================================================

class TestCounter(unittest.TestCase):
    """
    Test the implementation of the 'Counter' block class (bidirectional)
    """

    def test_init(self):
        """Test initialization with default and custom parameters"""

        # Default initialization
        C = Counter()
        self.assertEqual(C.start, 0)
        self.assertEqual(C.threshold, 0.0)
        self.assertEqual(len(C.events), 1)
        self.assertIsInstance(C.E, ZeroCrossing)
        self.assertIsInstance(C.events[0], ZeroCrossing)

        # Custom initialization
        C = Counter(start=10, threshold=5.0)
        self.assertEqual(C.start, 10)
        self.assertEqual(C.threshold, 5.0)


    def test_len(self):
        """Test that counter has no direct passthrough"""

        C = Counter()
        self.assertEqual(len(C), 0)


    def test_update_initial(self):
        """Test initial output before any events"""

        C = Counter(start=5)
        C.inputs[0] = 0.0
        C.update(0)

        # Output should be start value
        self.assertEqual(C.outputs[0], 5)


    def test_output_formula(self):
        """Test that output equals start + event count"""

        C = Counter(start=5)

        # Initially no events, output = start
        C.update(0)
        self.assertEqual(C.outputs[0], 5 + len(C.E))

        # After initialization, len(C.E) should be 0
        self.assertEqual(C.outputs[0], 5)


    def test_custom_start_values(self):
        """Test counter with various custom start values"""

        for start_val in [0, 10, -5, 100]:
            C = Counter(start=start_val)
            C.update(0)
            # Output should be start + number of events (initially 0)
            self.assertEqual(C.outputs[0], start_val)


    def test_threshold(self):
        """Test event function with threshold"""

        C = Counter(start=0, threshold=10.0)

        C.inputs[0] = 15.0
        event_val = C.E.func_evt(0)
        self.assertEqual(event_val, 5.0)  # 15 - 10

        C.inputs[0] = 5.0
        event_val = C.E.func_evt(0)
        self.assertEqual(event_val, -5.0)  # 5 - 10


class TestCounterUp(unittest.TestCase):
    """
    Test the implementation of the 'CounterUp' block class (unidirectional up)
    """

    def test_init(self):
        """Test initialization"""

        CU = CounterUp(start=5, threshold=2.0)
        self.assertEqual(CU.start, 5)
        self.assertEqual(CU.threshold, 2.0)
        self.assertIsInstance(CU.E, ZeroCrossingUp)
        self.assertIsInstance(CU.events[0], ZeroCrossingUp)


    def test_len(self):
        """Test that counter has no direct passthrough"""

        CU = CounterUp()
        self.assertEqual(len(CU), 0)


    def test_update(self):
        """Test output updates"""

        CU = CounterUp(start=10)

        CU.update(0)
        # Output should be start + number of events (initially 0)
        self.assertEqual(CU.outputs[0], 10)


class TestCounterDown(unittest.TestCase):
    """
    Test the implementation of the 'CounterDown' block class (unidirectional down)
    """

    def test_init(self):
        """Test initialization"""

        CD = CounterDown(start=5, threshold=2.0)
        self.assertEqual(CD.start, 5)
        self.assertEqual(CD.threshold, 2.0)
        self.assertIsInstance(CD.E, ZeroCrossingDown)
        self.assertIsInstance(CD.events[0], ZeroCrossingDown)


    def test_len(self):
        """Test that counter has no direct passthrough"""

        CD = CounterDown()
        self.assertEqual(len(CD), 0)


    def test_update(self):
        """Test output updates"""

        CD = CounterDown(start=20)

        CD.update(0)
        # Output should be start + number of events (initially 0)
        self.assertEqual(CD.outputs[0], 20)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
