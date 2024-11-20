########################################################################################
##
##                                  TESTS FOR 
##                               'blocks.rng.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.rng import RNG


# TESTS ================================================================================

class TestRNG(unittest.TestCase):
    """
    Test the implementation of the 'RNG' block class
    """

    def test_init(self):

        R = RNG()

        self.assertEqual(R.sampling_rate, None)

        R = RNG(sampling_rate=1)

        self.assertEqual(R.n_samples, 0)
        self.assertEqual(R.sampling_rate, 1)


    def test_len(self):
        
        R = RNG()

        #no passthrough
        self.assertEqual(len(R), 0)


    def test_str(self):

        R = RNG()

        #test default str method
        self.assertEqual(str(R), "RNG")


    def test_reset(self):

        R = RNG()

        for t in range(10):
            R.sample(t)

        R.reset()

        #test if reset worked
        self.assertEqual(R.n_samples, 0)
        self.assertEqual(R.get(0), 0.0)


    def test_sample(self):

        #first test default 'sampling_rate=None'
        R = RNG()

        for t in range(10):

            #test sample counter
            self.assertEqual(R.n_samples, t)

            old = R.get(0)

            R.sample(t)

            #test if new random value is sampled
            self.assertNotEqual(old, R.get(0))


        #next test finite sampling rate (samples every two seconds)
        R = RNG(sampling_rate=0.5)

        for t in range(10):

            #test sample counter 
            self.assertEqual(R.n_samples, t//2)

            old = R.get(0)

            R.sample(t)

            if t%2 == 0:
                #test if value remains the same is sampled
                self.assertEqual(old, R.get(0))

            else:
                #test if new random value is sampled
                self.assertNotEqual(old, R.get(0))
                

# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)