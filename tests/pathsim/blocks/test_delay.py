########################################################################################
##
##                                  TESTS FOR 
##                              'blocks.delay.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.delay import Delay
from pathsim.utils.adaptivebuffer import AdaptiveBuffer


# TESTS ================================================================================

class TestDelay(unittest.TestCase):
    """
    Test the implementation of the 'Delay' block class
    """

    def test_init(self):

        #test specific initialization
        D = Delay(tau=1)

        self.assertTrue(isinstance(D._buffer, AdaptiveBuffer))
        self.assertEqual(D.tau, 1)


    def test_len(self):
        
        D = Delay()

        #no passthrough
        self.assertEqual(len(D), 0)


    def test_str(self):

        D = Delay()

        #test default str method
        self.assertEqual(str(D), "Delay")


    def test_reset(self):

        D = Delay(tau=100)

        for t in range(10):
            D.sample(t)

        self.assertEqual(len(D._buffer), 10)    

        D.reset()

        #test if reset worked
        self.assertEqual(len(D._buffer), 0)  
        

    def test_sample(self):

        D = Delay(tau=100)

        for t in range(10):

            #test internal buffer length
            self.assertEqual(len(D._buffer), t)

            D.sample(t)


    def test_update(self):

        #test delay without interpolation
        D = Delay(tau=10)

        for t in range(100):

            #test internal buffer length
            self.assertEqual(len(D._buffer), t)

            D.set(0, t)
            D.sample(t)

            err = D.update(t)

            #test if error returns correctly
            self.assertEqual(err, 0)

            #test if delay is correctly applied
            self.assertEqual(D.get(0), max(0, t-10))

        #test delay with local interpolation
        D = Delay(tau=10.5)

        for t in range(100):

            #test internal buffer length
            self.assertEqual(len(D._buffer), t)

            D.set(0, t)
            D.sample(t)

            err = D.update(t)

            #test if error returns correctly
            self.assertEqual(err, 0)

            #test if delay is correctly applied
            self.assertEqual(D.get(0), max(0, t-10.5))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)