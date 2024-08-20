########################################################################################
##
##                                  TESTS FOR 
##                                'subsystem.py'
##
##                               Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.subsystem import Subsystem, Interface

#for testing
from pathsim.blocks._block import Block
from pathsim.connection import Connection


# TESTS ================================================================================

class TestInterface(unittest.TestCase):
    """
    Test the implementation of the 'Interface' class
    """

    def test_set_output(self):
        I = Interface()

        #test that outputs are zero
        self.assertEqual(I.outputs, {0: 0.0})

        #test if output is correctly set
        I.set_output(0, 0.2)
        I.set_output(1, 1.3)
        I.set_output(2, 2.4)
        I.set_output(3, 3.5)
        
        self.assertEqual(I.get(0), 0.2)
        self.assertEqual(I.get(1), 1.3)
        self.assertEqual(I.get(2), 2.4)
        self.assertEqual(I.get(3), 3.5)


    def test_get_input(self):

        I = Interface()

        #test that inputs are zero
        self.assertEqual(I.inputs, {0: 0.0})

        #test default retrieval        
        self.assertEqual(I.get_input(2), 0.0)

        #test if input is correctly retrieved
        I.set(0, 0.2)
        I.set(1, 1.3)
        I.set(2, 2.4)
        I.set(3, 3.5)
        
        self.assertEqual(I.get_input(0), 0.2)
        self.assertEqual(I.get_input(1), 1.3)
        self.assertEqual(I.get_input(2), 2.4)
        self.assertEqual(I.get_input(3), 3.5)


class TestSubsystem(unittest.TestCase):
    """
    test implementation of the 'Subsystem' class
    """

    def test_init(self):

        #test default initialization
        with self.assertRaises(ValueError):
            S = Subsystem()

        #test initialization without interface
        with self.assertRaises(ValueError):
            S = Subsystem(blocks=[Block(), Block()])

        #test specific initialization with interface
        B1, B2, B3 = Block(), Block(), Block()
        I1 = Interface()
        C1 = Connection(I1, B1, B2, B3)
        C2 = Connection(B1, I1)
        S = Subsystem(blocks=[B1, B2, B3, I1], connections=[C1, C2])
        self.assertEqual(len(S.blocks), 3)
        self.assertEqual(len(S.connections), 2)


    def test_check_connections(self):

        #test specific initialization with connecion override
        B1, B2, B3 = Block(), Block(), Block()
        I1 = Interface()
        C1 = Connection(I1, B1, B2, B3)
        C2 = Connection(B1, I1)
        C3 = Connection(B2, B3) # <-- this one overrides B3
        with self.assertRaises(ValueError):
            S = Subsystem(blocks=[B1, B2, B3, I1], connections=[C1, C2, C3])


    def test_len(self):

        #test the len method for internal signal path estimation

        I1 = Interface()
        S = Subsystem(blocks=[I1],)
        self.assertEqual(len(S), 1)

        B1 = Block()
        I1 = Interface()
        C1 = Connection(I1, B1)
        S = Subsystem(blocks=[I1, B1], connections=[C1])
        self.assertEqual(len(S), 2)

        B1, B2 = Block(), Block()
        I1 = Interface()
        C1 = Connection(I1, B1)
        C2 = Connection(B1, B2)
        S = Subsystem(blocks=[I1, B1, B2], connections=[C1, C2])
        self.assertEqual(len(S), 3)

        B1, B2, B3 = Block(), Block(), Block()
        I1 = Interface()
        C1 = Connection(I1, B1, B2, B3)
        C2 = Connection(B1, B2[1])
        C3 = Connection(B2, B1[1], B3[1])
        C4 = Connection(B3, I1)
        S = Subsystem(blocks=[I1, B1, B2, B3], connections=[C1, C2, C3, C4])
        self.assertEqual(len(S), 5)


    def test_set(self): 

        B1 = Block()
        I1 = Interface()
        C1 = Connection(I1, B1, I1)
        S = Subsystem(blocks=[I1, B1], connections=[C1])

        self.assertEqual(S.interface.outputs, {0:0.0})

        S.set(0, 1.1)
        S.set(1, 2.2)
        S.set(2, 3.3)

        self.assertEqual(S.interface.outputs, {0:1.1, 1:2.2, 2:3.3})


    def test_get(self): 

        B1 = Block()
        I1 = Interface()
        C1 = Connection(I1, B1, I1)
        S = Subsystem(blocks=[I1, B1], connections=[C1])

        S.interface.inputs = {0:1.1, 1:2.2, 2:3.3}

        self.assertEqual(S.get(0), 1.1)
        self.assertEqual(S.get(1), 2.2)
        self.assertEqual(S.get(2), 3.3)


    def test_update(self): 

        B1 = Block()
        I1 = Interface()
        C1 = Connection(I1, B1)
        S = Subsystem(blocks=[I1, B1], connections=[C1])

        err = S.update(0)

        self.assertEqual(err, 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
