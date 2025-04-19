########################################################################################
##
##                                  TESTS FOR 
##                               'connection.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.utils.portreference import PortReference
from pathsim.connection import Connection, Duplex
from pathsim.blocks._block import Block


# TESTS ================================================================================

class TestConnection(unittest.TestCase):
    """
    Test the implementation of the 'Connection' class
    """

    def test_init_none(self):
        
        #default
        with self.assertRaises(TypeError):
            C = Connection()


    def test_init_single(self):

        B1, B2 = Block(), Block()

        #default
        C = Connection(B1, B2)

        self.assertTrue(isinstance(C.source, PortReference))
        self.assertTrue(isinstance(C.targets, list))

        #mixed


        #all



    def test_init_multi(self):

        B1, B2, B3 = Block(), Block(), Block()

        #default
        C = Connection(B1, B2, B3)

        #mixed

        #all


    def test_overwrites(self):

        B1, B2, B3 = Block(), Block(), Block()

        C1 = Connection(B1, B2) 
        C2 = Connection(B1, B3)  
        C3 = Connection(B2, B3) 

        self.assertFalse(C1.overwrites(C2))
        self.assertFalse(C2.overwrites(C1))
        self.assertTrue(C3.overwrites(C2))
        self.assertTrue(C2.overwrites(C3))

        C1 = Connection(B1, B2, B3) 
        C2 = Connection(B1, B3)  

        self.assertTrue(C1.overwrites(C2))
        self.assertTrue(C2.overwrites(C1))



    def test_update_single(self):

        B1, B2 = Block(), Block()

        #test data transfer with default ports
        C = Connection(B1, B2) 
        B1.outputs[0] = 3
        C.update()
        self.assertEqual(B2.inputs[0], 3)

        #test data transfer with specific ports

        #test data transfer with mixed ports


    def test_update_multi(self):

        B1, B2, B3 = Block(), Block(), Block()

        #test data transfer with default ports
        C = Connection(B1, B2, B3) 
        B1.outputs[0] = 3
        C.update()
        self.assertEqual(B2.inputs[0], 3)
        self.assertEqual(B3.inputs[0], 3)

        #test data transfer with specific ports



    def test_on_off_bool(self):
        
        B1, B2 = Block(), Block()

        #default
        C = Connection(B1, B2)

        #default active
        self.assertTrue(C)

        #deactivate
        C.off()
        self.assertFalse(C)

        #activate
        C.on()
        self.assertTrue(C)



class TestDuplex(unittest.TestCase):
    """
    Test the implementation of the 'Duplex' class (bidirectional connection)
    """

    def test_init_none(self):
        
        #default
        with self.assertRaises(TypeError):
            D = Duplex()


    def test_init_mixed(self):

        B1, B2 = Block(), Block()

        #default
        D = Duplex(B1, B2)

        #mixed
        
        #all

        #test too many


    def test_update(self):

        B1, B2 = Block(), Block()

        #test data transfer with default ports
        D = Duplex(B1, B2) 
        B1.outputs[0] = 3
        B2.outputs[0] = 1
        D.update()
        self.assertEqual(B1.inputs[0], 1)
        self.assertEqual(B2.inputs[0], 3)

        #test data transfer with special ports



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)