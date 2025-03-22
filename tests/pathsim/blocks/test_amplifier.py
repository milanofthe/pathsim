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

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestAmplifier(unittest.TestCase):
    """
    Test the implementation of the 'Amplifier' block class
    """

    def test_init(self):

        A = Amplifier(gain=5)

        self.assertEqual(A.gain, 5)


    def test_len(self):

        A = Amplifier(gain=5)

        self.assertEqual(len(A), 1)


    def test_embedding(self):

        A = Amplifier(gain=5)
        E = Embedding(A, np.sin, lambda t: 5 * np.sin(t))
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        A = Amplifier(gain=0.5)
        E = Embedding(A, np.cos, lambda t: 0.5 * np.cos(t))
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        A = Amplifier(gain=-1e6)
        E = Embedding(A, np.exp, lambda t: -1e6 * np.exp(t))
        for t in range(10): self.assertEqual(*E.check_SISO(t))


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


    def test_linearize(self):

        A = Amplifier(gain=5)

        #set block inputs
        A.set(0, 1)
        err = A.update(None)

        #test if update was correct
        self.assertEqual(A.get(0), 5)

        #linearize gain (its already linear)
        A.linearize(3)

        #set block inputs
        A.set(0, 1)
        err = A.update(None)

        #test if update was correct
        self.assertEqual(A.get(0), 5)


    def test_linearize(self):

        A = Amplifier(gain=5)

        #set block inputs
        A.set(0, 1)
        err = A.update(None)

        #test if update was correct
        self.assertEqual(A.get(0), 5)

        #linearize gain (its already linear)
        A.linearize(3)

        #set block inputs
        A.set(0, 3)
        err = A.update(None)

        #test if update was correct
        self.assertEqual(A.get(0), 15)

        #reset linearization
        A.delinearize()

        #set block inputs
        A.set(0, 0.1)
        err = A.update(None)

        #test if update was correct
        self.assertEqual(A.get(0), 0.5)





# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)