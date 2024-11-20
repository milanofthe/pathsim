########################################################################################
##
##                                  TESTS FOR 
##                               'blocks.adder.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.adder import Adder


# TESTS ================================================================================

class TestAdder(unittest.TestCase):
    """
    Test the implementation of the 'Adder' block class
    """

    def test_str(self):

        A = Adder()

        #test default str method
        self.assertEqual(str(A), "Adder")


    def test_update_single(self):
        
        A = Adder()

        #set block inputs
        A.set(0, 1)

        #update block
        err = A.update(None)

        #test if update was correct
        self.assertEqual(A.get(0), 1)

        #test if error was computed correctly
        self.assertGreater(err, 0)

        #update block again
        err = A.update(None)

        #test error, now should be 0
        self.assertEqual(err, 0)


    def test_update_multi(self):

        A = Adder()

        #set block inputs
        A.set(0, 1)
        A.set(1, 2.0)
        A.set(2, 3.1)

        #update block
        err = A.update(None)

        #test if update was correct
        self.assertEqual(A.get(0), 6.1)

        #test if error was computed correctly
        self.assertGreater(err, 0)

        #update block again
        err = A.update(None)

        #test error, now should be 0
        self.assertEqual(err, 0)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)