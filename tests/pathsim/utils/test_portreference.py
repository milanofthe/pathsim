########################################################################################
##
##                                  TESTS FOR 
##                           'utils.portreference.py'
##
##                              Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks._block import Block
from pathsim.utils.portreference import PortReference


# TESTS ================================================================================

class TestPortReference(unittest.TestCase):
    """
    test the 'PortReference' container class
    """

    def test_init(self):

        B = Block()

        #default
        PR = PortReference(B)
        self.assertEqual(PR.block, B)
        self.assertEqual(PR.ports, [0])

        #special
        PR = PortReference(B, [0])
        self.assertEqual(PR.ports, [0]) #<- same as default

        PR = PortReference(B, [0, 1, 2, 3])
        self.assertEqual(PR.ports, [0, 1, 2, 3])

        PR = PortReference(B, [2, 3, 1])
        self.assertEqual(PR.ports, [2, 3, 1])

        #input validation
        with self.assertRaises(ValueError): PR = PortReference(B, [2, 2, 3])    #duplicates
        with self.assertRaises(ValueError): PR = PortReference(B, [2, 3, 2, 1]) #duplicates
        with self.assertRaises(ValueError): PR = PortReference(B, [-1, 2, 3])   #negative integer
        with self.assertRaises(ValueError): PR = PortReference(B, [1, 2.1])     #no int
        with self.assertRaises(ValueError): PR = PortReference(B, (3, 2))       #no list but iterable
        with self.assertRaises(ValueError): PR = PortReference(B, 1)            #no list but int
        with self.assertRaises(ValueError): PR = PortReference(B, "d")          #no list but str


    def test_len(self):

        B = Block()

        #default
        PR = PortReference(B)
        self.assertEqual(len(PR), 1)

        #special
        PR = PortReference(B, [0])
        self.assertEqual(len(PR), 1) #<- same as default

        PR = PortReference(B, [0, 1, 2, 3])
        self.assertEqual(len(PR), 4)

        PR = PortReference(B, [2, 3, 1])
        self.assertEqual(len(PR), 3)


    def test_to(self):

        #data transfer
        B1 = Block()
        B2 = Block()

        B1.outputs[0] = 123
        B1.outputs[1] = 231
        B1.outputs[2] = 312

        P1 = PortReference(B1, [0, 1, 2])
        P2 = PortReference(B2, [2, 3, 4])

        self.assertEqual(B1.inputs[2], 0.0)
        self.assertEqual(B1.inputs[3], 0.0)
        self.assertEqual(B1.inputs[4], 0.0)

        self.assertEqual(B2.inputs[2], 0.0)
        self.assertEqual(B2.inputs[3], 0.0)
        self.assertEqual(B2.inputs[4], 0.0)

        #to other
        P1.to(P2)

        self.assertEqual(B2.inputs[2], 123)
        self.assertEqual(B2.inputs[3], 231)
        self.assertEqual(B2.inputs[4], 312)

        #to self, directly from outputs -> inputs
        P1.to(P1)

        self.assertEqual(B1.inputs[0], 123)
        self.assertEqual(B1.inputs[1], 231)
        self.assertEqual(B1.inputs[2], 312)


    def test_get_inputs(self):

        B = Block()

        #default
        PR = PortReference(B)
        self.assertEqual(PR.get_inputs(), 0) 

        #special
        PR = PortReference(B, [0, 1, 2])
        self.assertTrue(np.allclose(PR.get_inputs(), np.zeros(3))) 

        B.inputs[0] = 3
        B.inputs[1] = 2
        self.assertTrue(np.allclose(PR.get_inputs(), np.array([3, 2, 0]))) 


    def test_get_outputs(self):

        B = Block()

        #default
        PR = PortReference(B)
        self.assertEqual(PR.get_outputs(), 0) 

        #special
        PR = PortReference(B, [0, 1, 2])
        self.assertTrue(np.allclose(PR.get_outputs(), np.zeros(3))) 

        B.outputs[0] = 3
        B.outputs[1] = 2
        self.assertTrue(np.allclose(PR.get_outputs(), np.array([3, 2, 0]))) 


    def test_set_inputs(self): 

        B = Block()

        #default
        PR = PortReference(B)
        PR.set_inputs([9])
        self.assertEqual(B.inputs[0], 9) 

        #special
        PR = PortReference(B, [0, 1, 2])
        PR.set_inputs([9, 8, 7])
        self.assertEqual(B.inputs[0], 9)
        self.assertEqual(B.inputs[1], 8)
        self.assertEqual(B.inputs[2], 7) 


    def test_set_outputs(self): 

        B = Block()

        #default
        PR = PortReference(B)
        PR.set_outputs([9])
        self.assertEqual(B.outputs[0], 9) 

        #special
        PR = PortReference(B, [0, 1, 2])
        PR.set_outputs([9, 8, 7])
        self.assertEqual(B.outputs[0], 9)
        self.assertEqual(B.outputs[1], 8)
        self.assertEqual(B.outputs[2], 7)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)