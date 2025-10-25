########################################################################################
##
##                             TESTS FOR ANALYSIS TOOLS
##                              'utils/analysis.py'
##
##                              Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import time
import logging

from pathsim.utils.analysis import Timer, timer
from pathsim.utils.logger import LoggerManager


# TESTS ================================================================================

class TestTimer(unittest.TestCase):
    """
    Test the implementation of the 'Timer' context manager and decorator
    """

    def setUp(self):
        """Setup logger for tests"""
        # Configure logging to capture debug messages
        LoggerManager._instance = None
        LoggerManager._initialized = False
        mgr = LoggerManager()
        mgr.configure(enabled=True, level=logging.DEBUG)


    def test_timer_context_manager_verbose(self):
        """Test Timer as context manager with verbose=True"""
        with Timer(verbose=True) as T:
            time.sleep(0.01)  # Sleep for 10ms

        # Should have recorded time
        self.assertIsNotNone(T.time)

        # Time should be approximately 10ms (allow for some variance)
        self.assertGreater(T.time, 0.008)  # At least 8ms
        self.assertLess(T.time, 0.020)     # At most 20ms

        # __repr__ should return formatted string
        repr_str = repr(T)
        self.assertIsNotNone(repr_str)
        self.assertIn("ms", repr_str)

        # __float__ should return time
        self.assertEqual(float(T), T.time)


    def test_timer_context_manager_non_verbose(self):
        """Test Timer as context manager with verbose=False"""
        with Timer(verbose=False) as T:
            time.sleep(0.01)

        # Should still record time
        self.assertIsNotNone(T.time)
        self.assertGreater(T.time, 0.008)


    def test_timer_repr_before_execution(self):
        """Test timer before execution has None time"""
        T = Timer()
        self.assertIsNone(T.time)
        # __repr__ returns None internally when time is None
        self.assertIsNone(T.__repr__())


    def test_timer_as_decorator(self):
        """Test Timer as function decorator"""
        @Timer(verbose=False)
        def test_function():
            time.sleep(0.01)
            return "result"

        result = test_function()

        # Function should still return correctly
        self.assertEqual(result, "result")


    def test_timer_measures_correctly(self):
        """Test that Timer measures time accurately"""
        with Timer(verbose=False) as T:
            time.sleep(0.05)  # 50ms

        # Should be close to 50ms
        self.assertGreater(T.time, 0.045)
        self.assertLess(T.time, 0.060)


class TestTimerDecorator(unittest.TestCase):
    """
    Test the 'timer' decorator function
    """

    def setUp(self):
        """Setup logger for tests"""
        LoggerManager._instance = None
        LoggerManager._initialized = False
        mgr = LoggerManager()
        mgr.configure(enabled=True, level=logging.DEBUG)


    def test_timer_decorator_basic(self):
        """Test timer decorator on simple function"""
        @timer
        def add(a, b):
            return a + b

        result = add(2, 3)

        # Function should work correctly
        self.assertEqual(result, 5)


    def test_timer_decorator_with_sleep(self):
        """Test timer decorator measures time"""
        @timer
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()

        # Should return correct result
        self.assertEqual(result, "done")


    def test_timer_decorator_preserves_function_name(self):
        """Test that timer decorator preserves function metadata"""
        @timer
        def my_function():
            """My docstring"""
            return True

        # Should preserve name
        self.assertEqual(my_function.__name__, "my_function")

        # Should preserve docstring
        self.assertEqual(my_function.__doc__, "My docstring")


    def test_timer_decorator_with_args_kwargs(self):
        """Test timer decorator with args and kwargs"""
        @timer
        def complex_function(a, b, c=0, d=0):
            return a + b + c + d

        result = complex_function(1, 2, c=3, d=4)
        self.assertEqual(result, 10)


    def test_timer_decorator_with_exception(self):
        """Test timer decorator when function raises exception"""
        @timer
        def failing_function():
            raise ValueError("test error")

        # Exception should propagate
        with self.assertRaises(ValueError):
            failing_function()


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
