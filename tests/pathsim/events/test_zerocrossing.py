########################################################################################
##
##                                   TESTS FOR 
##                            'events.zerocrossing.py'
##
##                               Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.events.zerocrossing import (
    ZeroCrossing, 
    ZeroCrossingUp, 
    ZeroCrossingDown
    )


# TESTS ================================================================================

class TestZeroCrossing(unittest.TestCase):
    """
    Test the implementation of the 'ZeroCrossing' event class.
    """

    def test_detect_up(self):

        #upwards
        e = ZeroCrossing(func_evt=lambda t: t-2)

        #before crossing
        e.buffer(0)
        de, cl, ra = e.detect(1)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)

        #crossing 1
        de, cl, ra = e.detect(3)
        self.assertTrue(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 2/3)

        #crossing 2
        e.buffer(1)
        de, cl, ra = e.detect(3)
        self.assertTrue(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1/2)

        #after crossing
        e.buffer(3)
        de, cl, ra = e.detect(4)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)


    def test_detect_down(self):

        #downwards
        e = ZeroCrossing(func_evt=lambda t: t-2)

        #before crossing
        e.buffer(0)
        de, cl, ra = e.detect(1)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)

        #crossing 1
        de, cl, ra = e.detect(3)
        self.assertTrue(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 2/3)

        #crossing 2
        e.buffer(1)
        de, cl, ra = e.detect(3)
        self.assertTrue(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1/2)

        #after crossing
        e.buffer(3)
        de, cl, ra = e.detect(4)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)


class TestZeroCrossingUp(unittest.TestCase):
    """
    Test the implementation of the 'ZeroCrossingUp' event class.
    """

    def test_detect_up(self):

        #upwards
        e = ZeroCrossingUp(func_evt=lambda t: t-2)

        #before crossing
        e.buffer(0)
        de, cl, ra = e.detect(1)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)

        #crossing 1
        de, cl, ra = e.detect(3)
        self.assertTrue(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 2/3)

        #crossing 2
        e.buffer(1)
        de, cl, ra = e.detect(3)
        self.assertTrue(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1/2)

        #after crossing
        e.buffer(3)
        de, cl, ra = e.detect(4)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)


    def test_detect_down(self):

        #downwards
        e = ZeroCrossingUp(func_evt=lambda t: t-2)

        #before crossing
        e.buffer(0)
        de, cl, ra = e.detect(1)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)

        #after crossing
        e.buffer(3)
        de, cl, ra = e.detect(4)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)


class TestZeroCrossingDown(unittest.TestCase):
    """
    Test the implementation of the 'ZeroCrossingDown' event class.
    """

    def test_detect_up(self):

        #upwards
        e = ZeroCrossingDown(func_evt=lambda t: t-2)

        #before crossing
        e.buffer(0)
        de, cl, ra = e.detect(1)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)

        #crossing 1 -> not triggering
        de, cl, ra = e.detect(3)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)

        #crossing 2 -> not triggering
        e.buffer(1)
        de, cl, ra = e.detect(3)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)

        #after crossing
        e.buffer(3)
        de, cl, ra = e.detect(4)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)


    def test_detect_down(self):

        #downwards
        e = ZeroCrossingDown(func_evt=lambda t: t-2)

        #before crossing
        e.buffer(0)

        de, cl, ra = e.detect(1)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)


        #after crossing
        e.buffer(3)
        de, cl, ra = e.detect(4)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
