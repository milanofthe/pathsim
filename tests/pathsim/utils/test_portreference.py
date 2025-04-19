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
        with self.assertRaises(ValueError): PR = PortReference(B, ["e", 2, 3])  #list but no int
        with self.assertRaises(ValueError): PR = PortReference(B, [1, 2.1])     #no int
        with self.assertRaises(ValueError): PR = PortReference(B, (3, 2))       #no list but iterable
        with self.assertRaises(ValueError): PR = PortReference(B, 1)            #no list but int
        with self.assertRaises(ValueError): PR = PortReference(B, "d")          #no list



    def test_set(self):

        B = Block()

        #default
        PR = PortReference(B)
        PR.set([23])
        self.assertEqual(B.inputs[0], 23)
        PR.set([0.02])
        self.assertEqual(B.inputs[0], 0.02)
    
        #specific
        PR = PortReference(B, [0, 2, 3])
        PR.set([33, -0.3, 1e4])
        self.assertEqual(B.inputs[0], 33)
        self.assertEqual(B.inputs[2], -0.3)
        self.assertEqual(B.inputs[3], 1e4)

        PR.set([1, 3])
        self.assertEqual(B.inputs[0], 1)
        self.assertEqual(B.inputs[2], 3)
        self.assertEqual(B.inputs[3], 1)

        PR.set([1])
        self.assertEqual(B.inputs[0], 1)
        self.assertEqual(B.inputs[2], 1)
        self.assertEqual(B.inputs[3], 1)

        PR.set([1, 2, 3, 4, 5])
        self.assertEqual(B.inputs[0], 1)
        self.assertEqual(B.inputs[2], 2)
        self.assertEqual(B.inputs[3], 3)


    def test_get(self):

        B = Block()

        #default
        PR = PortReference(B)
        self.assertEqual(PR.get(), [0.0])

        #specific
        B.outputs = {0:0.0, 1:3.2, 2:-0.04}

        PR = PortReference(B, [0, 1, 2])
        self.assertEqual(PR.get(), [0.0, 3.2, -0.04])

        PR = PortReference(B, [1])
        self.assertEqual(PR.get(), [3.2])

        PR = PortReference(B, [0, 2])
        self.assertEqual(PR.get(), [0.0, -0.04])







# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)