########################################################################################
##
##                                  TESTS FOR 
##                            'blocks.amplifier.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.amplifier import Amplifier


# TESTS ================================================================================

class TestAmplifier(unittest.TestCase):
    """
    Test the implementation of the 'Amplifier' block class
    """

    def test_init(self):

        A = Amplifier(gain=5)

        self.assertEqual(A.gain, 5)


    def test_str(self):

        A = Amplifier()

        #test default str method
        self.assertEqual(str(A), "Amplifier")


    def test_update(self):
        
        A = Amplifier(gain=5)

        #set block inputs
        A.set(0, 1)

        #update block
        err = A.update(None)

        #test if update was correct
        self.assertEqual(A.get(0), 5)

        #test if error was computed correctly
        self.assertGreater(err, 0)

        #update block again
        err = A.update(None)

        #test error, now should be 0
        self.assertEqual(err, 0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)