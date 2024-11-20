########################################################################################
##
##                                     TESTS FOR 
##                             'utils/adaptivebuffer.py'
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.utils.adaptivebuffer import (
    AdaptiveBuffer
    )


# TESTS ================================================================================

class TestAdaptiveBuffer(unittest.TestCase):
    """
    test the implementation of the 'AdaptiveBuffer' class 
    """

    def test_init(self):

        d = 100

        buffer = AdaptiveBuffer(d)

        #test initialization
        self.assertEqual(buffer.delay, d)
        self.assertEqual(len(buffer.buffer), 0)
        self.assertEqual(buffer.counter, 0)


    def test_add(self):

        d = 100

        buffer = AdaptiveBuffer(d)

        for i in range(10*d):

            #test the counter
            self.assertLessEqual(buffer.counter, buffer.clean_every+1)

            buffer.add(i, i)


    def test_get_empty(self):

        d = 100

        buffer = AdaptiveBuffer(d)

        #test default empty buffer
        self.assertEqual(buffer.get(123), 0)


    def test_get(self):

        d = 100

        buffer = AdaptiveBuffer(d)

        for i in range(10*d):

            buffer.add(i, i)

            #test buffer readout
            self.assertEqual(buffer.get(i), 0 if i < d else i-d)


    def test_get_too_large(self):

        d = 100

        buffer = AdaptiveBuffer(d)

        for i in range(10*d):

            buffer.add(i, i)

            #test buffer readout out of bounds (most recent value)
            self.assertEqual(buffer.get(100*d), i)


    def test_clear(self):

        d = 100

        buffer = AdaptiveBuffer(d)

        for i in range(10*d):

            buffer.add(i, i)

        buffer.clear()

        #test buffer clearing
        self.assertEqual(len(buffer.buffer), 0)
        self.assertEqual(buffer.counter, 0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)