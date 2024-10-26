########################################################################################
##
##                                  TESTS FOR 
##                              'blocks._block.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks._block import Block


# TESTS ================================================================================

class TestBlock(unittest.TestCase):
    """
    Test the implementation of the base 'Block' class
    """

    def test_init(self):

        B = Block()

        #test default inputs and outputs
        self.assertEqual(B.inputs, {0: 0.0})
        self.assertEqual(B.outputs, {0: 0.0})

        #test default engine
        self.assertEqual(B.engine, None)


    def test_len(self):

        B = Block()

        #test default len method
        self.assertEqual(len(B), 1)
            

    def test_str(self):

        B = Block()

        #test default str method
        self.assertEqual(str(B), "Block")


    def test_getitem(self):

        B = Block()

        #test default getitem method (for connection creation)
        self.assertEqual(B[0], (B, 0))
        self.assertEqual(B[1], (B, 1))
        self.assertEqual(B[2], (B, 2))

        #test input validation
        with self.assertRaises(ValueError): B[0.2]
        with self.assertRaises(ValueError): B[1j]
        with self.assertRaises(ValueError): B["a"]


    def test_reset(self):

        B = Block()

        B.inputs = {0:0, 2:2, 1:1}
        B.outputs = {1:1, 0:0, 2:2}

        B.reset()

        #test if inputs and outputs are reset correctly
        self.assertEqual(B.inputs, {0:0.0, 1:0.0, 2:0.0})
        self.assertEqual(B.outputs, {0:0.0, 1:0.0, 2:0.0})


    def test_set(self):

        B = Block()

        B.set(0, 1)
        self.assertEqual(B.inputs[0], 1)

        B.set(0, 2)
        self.assertEqual(B.inputs[0], 2)

        B.set(2, 3)
        self.assertEqual(B.inputs[2], 3)


    def test_get(self):

        B = Block()

        B.outputs = {0:0, 2:2, 1:1}

        self.assertEqual(B.get(0), 0)
        self.assertEqual(B.get(1), 1)
        self.assertEqual(B.get(2), 2)

        #undefined output -> defaults to 0.0
        self.assertEqual(B.get(100), 0.0)


    def test_update(self):

        B = Block()

        #test default implementation 
        self.assertEqual(B.update(None), 0.0)


    def test_solve(self):

        B = Block()

        #test default implementation 
        self.assertEqual(B.solve(None, None), 0.0)


    def test_step(self):

        B = Block()

        #test default implementation 
        self.assertEqual(B.step(None, None), (True, 0.0, 0.0, 1.0))



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)