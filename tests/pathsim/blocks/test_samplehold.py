########################################################################################
##
##                                  TESTS FOR
##                            'blocks.samplehold.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.samplehold import SampleHold
from pathsim.events.schedule import Schedule


# TESTS ================================================================================

class TestSampleHold(unittest.TestCase):
    """
    Test the implementation of the 'SampleHold' block class
    """

    def test_init(self):
        """Test initialization with default and custom parameters"""

        # Default initialization
        SH = SampleHold()
        self.assertEqual(SH.T, 1)
        self.assertEqual(SH.tau, 0)
        self.assertEqual(len(SH.events), 1)
        self.assertIsInstance(SH.events[0], Schedule)

        # Custom initialization
        SH = SampleHold(T=0.5, tau=0.1)
        self.assertEqual(SH.T, 0.5)
        self.assertEqual(SH.tau, 0.1)
        self.assertEqual(len(SH.events), 1)


    def test_len(self):
        """Test that sample-hold has no direct passthrough"""

        SH = SampleHold()
        self.assertEqual(len(SH), 0)


    def test_event_scheduling(self):
        """Test that the schedule event is configured correctly"""

        SH = SampleHold(T=2.0, tau=0.5)

        # Check event properties
        event = SH.events[0]
        self.assertIsInstance(event, Schedule)
        self.assertEqual(event.t_start, 0.5)
        self.assertEqual(event.t_period, 2.0)


    def test_single_input_sample(self):
        """Test sampling a single input"""

        SH = SampleHold(T=1.0)

        # Set input
        SH.inputs[0] = 5.0

        # Manually trigger the sampling function (simulates scheduled event)
        SH.events[0].func_act(0)

        # Output should now match input
        self.assertEqual(SH.outputs[0], 5.0)


    def test_multiple_samples(self):
        """Test multiple sampling events"""

        SH = SampleHold(T=1.0)

        # First sample
        SH.inputs[0] = 3.0
        SH.events[0].func_act(0)
        self.assertEqual(SH.outputs[0], 3.0)

        # Second sample with different input
        SH.inputs[0] = 7.0
        SH.events[0].func_act(1)
        self.assertEqual(SH.outputs[0], 7.0)

        # Third sample
        SH.inputs[0] = -2.5
        SH.events[0].func_act(2)
        self.assertEqual(SH.outputs[0], -2.5)


    def test_hold_behavior(self):
        """Test that output holds value between samples"""

        SH = SampleHold(T=1.0)

        # Initial sample
        SH.inputs[0] = 10.0
        SH.events[0].func_act(0)
        self.assertEqual(SH.outputs[0], 10.0)

        # Change input but don't sample - output should hold
        SH.inputs[0] = 20.0
        self.assertEqual(SH.outputs[0], 10.0)  # Still holding previous value


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
