########################################################################################
##
##                                   TESTS FOR
##                             'events.condition.py'
##
##                               Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.events.condition import Condition


# TESTS ================================================================================

class TestCondition(unittest.TestCase):
    """
    Test the implementation of the 'Condition' event class.
    """

    def test_detect_false_condition(self):
        """Test detect when condition is false"""

        # Condition: t > 5
        e = Condition(func_evt=lambda t: t > 5)

        # Before condition is met
        e.buffer(1.0)
        detected, close, ratio = e.detect(2.0)

        self.assertFalse(detected)
        self.assertFalse(close)
        self.assertEqual(ratio, 0.5)  # Bisection halves the step


    def test_detect_true_condition_not_close(self):
        """Test detect when condition becomes true but not close enough"""

        # Condition: t > 5
        e = Condition(func_evt=lambda t: t > 5, tolerance=0.1)

        # Condition just became true, but time gap is large
        e.buffer(4.0)
        detected, close, ratio = e.detect(6.0)

        self.assertTrue(detected)
        self.assertFalse(close)  # 6.0 - 4.0 = 2.0 > tolerance
        self.assertEqual(ratio, 0.5)


    def test_detect_true_condition_close(self):
        """Test detect when condition is true and we're close enough"""

        # Condition: t > 5
        e = Condition(func_evt=lambda t: t > 5, tolerance=0.1)

        # Condition true and time gap is small
        e.buffer(5.05)
        detected, close, ratio = e.detect(5.08)

        self.assertTrue(detected)
        self.assertTrue(close)  # 5.08 - 5.05 = 0.03 < tolerance
        self.assertEqual(ratio, 1.0)


    def test_resolve_without_action(self):
        """Test resolve without action function"""

        e = Condition(func_evt=lambda t: t > 5)

        # Initially active (using __bool__)
        self.assertTrue(bool(e))

        # Resolve event
        e.resolve(5.5)

        # Should be deactivated after resolution
        self.assertFalse(bool(e))

        # Event time should be recorded
        self.assertEqual(len(e._times), 1)
        self.assertEqual(e._times[0], 5.5)


    def test_resolve_with_action(self):
        """Test resolve with action function"""

        action_called = []

        def action(t):
            action_called.append(t)

        e = Condition(func_evt=lambda t: t > 5, func_act=action)

        # Initially active (using __bool__)
        self.assertTrue(bool(e))

        # Resolve event
        e.resolve(5.5)

        # Action should have been called
        self.assertEqual(len(action_called), 1)
        self.assertEqual(action_called[0], 5.5)

        # Should be deactivated
        self.assertFalse(bool(e))


    def test_bisection_behavior(self):
        """Test that bisection narrows down to tolerance"""

        e = Condition(func_evt=lambda t: t > 10, tolerance=0.1)

        # Start before condition (large gap)
        e.buffer(9.0)
        detected, close, ratio = e.detect(11.0)
        self.assertTrue(detected)
        self.assertFalse(close)  # 11.0 - 9.0 = 2.0 > tolerance
        self.assertEqual(ratio, 0.5)

        # Narrow down (small gap)
        e.buffer(10.05)
        detected, close, ratio = e.detect(10.08)
        self.assertTrue(detected)
        self.assertTrue(close)  # 10.08 - 10.05 = 0.03 < tolerance
        self.assertEqual(ratio, 1.0)


    def test_len(self):
        """Test that len() returns number of times event occurred"""

        e = Condition(func_evt=lambda t: t > 5)

        self.assertEqual(len(e), 0)

        e.resolve(5.5)
        self.assertEqual(len(e), 1)

        e.resolve(10.5)
        self.assertEqual(len(e), 2)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
