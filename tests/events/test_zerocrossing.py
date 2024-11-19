########################################################################################
##
##                                  TESTS FOR 
##                              'events._event.py'
##
##                               Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.events.zerocrossing import ZeroCrossing

from pathsim.blocks._block import Block


# TESTS ================================================================================

class TestZeroCrossing(unittest.TestCase):
    """
    Test the implementation of the 'ZeroCrossing' event class.
    """

    def test_detect_up(self):

        #upwards
        e = ZeroCrossing(func_evt=lambda y, x, t: t-2)

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
        e = ZeroCrossing(func_evt=lambda y, x, t: 2-t)

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


    def test_detect_on_off(self):

        e = ZeroCrossing(func_evt=lambda y, x, t: t-2)

        #reference
        e.buffer(1)
        de, cl, ra = e.detect(3)
        self.assertTrue(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1/2)

        #turn off
        e.off()
        de, cl, ra = e.detect(3)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)

        #turn on again
        e.on()
        de, cl, ra = e.detect(3)
        self.assertTrue(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1/2)






  


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
