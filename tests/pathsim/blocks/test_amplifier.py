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
        """test algebraic component via embedding"""

        A = Amplifier(gain=5)
        E = Embedding(A, np.sin, lambda t: 5 * np.sin(t))
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        A = Amplifier(gain=0.5)
        E = Embedding(A, np.cos, lambda t: 0.5 * np.cos(t))
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        A = Amplifier(gain=-1e6)
        E = Embedding(A, np.exp, lambda t: -1e6 * np.exp(t))
        for t in range(10): self.assertEqual(*E.check_SISO(t))


    def test_linearization(self):
        """test linearization and delinearization"""

        A = Amplifier(gain=5)

        def src(t): return np.cos(t)
        def ref(t): return 5*np.cos(t)

        E = Embedding(A, src, ref)

        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #linearize block
        A.linearize(3)

        for t in range(10): 
            a, b = E.check_SISO(t)
            self.assertAlmostEqual(np.linalg.norm(a-b), 0, 8)

        #linearize at differnt point in time block
        A.linearize(12)

        for t in range(10): 
            a, b = E.check_SISO(t)
            self.assertAlmostEqual(np.linalg.norm(a-b), 0, 8)

        #delinearize
        A.delinearize()

        for t in range(10): self.assertEqual(*E.check_SISO(t))


    def test_update(self):
        
        A = Amplifier(gain=5)

        #set block inputs
        A.inputs[0] = 1

        #update block
        A.update(None)

        #test if update was correct
        self.assertEqual(A.outputs[0], 5)




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)