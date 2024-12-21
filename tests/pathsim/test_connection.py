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

        #test if ports assigned correctly
        self.assertEqual(C.source, (B1, 0))
        self.assertEqual(C.targets, [(B2, 0)])

        #mixed
        C1 = Connection(B1, (B2, 2))    
        C2 = Connection((B1, 3), B2)
        
        #test if ports assigned correctly
        self.assertEqual(C1.source, (B1, 0))
        self.assertEqual(C1.targets, [(B2, 2)])

        #test if ports assigned correctly
        self.assertEqual(C2.source, (B1, 3))
        self.assertEqual(C2.targets, [(B2, 0)])

        #all
        C = Connection((B1, 4), (B2, 1))

        #test if ports assigned correctly
        self.assertEqual(C.source, (B1, 4))
        self.assertEqual(C.targets, [(B2, 1)])


    def test_init_multi(self):

        B1, B2, B3 = Block(), Block(), Block()

        #default
        C = Connection(B1, B2, B3)

        #test if ports assigned correctly
        self.assertEqual(C.source, (B1, 0))
        self.assertEqual(C.targets, [(B2, 0), (B3, 0)])

        #mixed
        C1 = Connection(B1, (B2, 2), B3)    
        C2 = Connection((B1, 3), B2, (B3, 1))
        
        #test if ports assigned correctly
        self.assertEqual(C1.source, (B1, 0))
        self.assertEqual(C1.targets, [(B2, 2), (B3, 0)])

        #test if ports assigned correctly
        self.assertEqual(C2.source, (B1, 3))
        self.assertEqual(C2.targets, [(B2, 0), (B3, 1)])

        #all
        C = Connection((B1, 4), (B2, 1), (B3, 2))

        #test if ports assigned correctly
        self.assertEqual(C.source, (B1, 4))
        self.assertEqual(C.targets, [(B2, 1), (B3, 2)])


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

        C1 = Connection(B1, (B2, 1)) 
        C2 = Connection(B1, (B2, 2))  

        self.assertFalse(C1.overwrites(C2))
        self.assertFalse(C2.overwrites(C1))

        C1 = Connection((B1, 1), B3) 
        C2 = Connection((B1, 2), B3)  

        self.assertTrue(C1.overwrites(C2))
        self.assertTrue(C2.overwrites(C1))

        C1 = Connection((B1, 1), B3) 
        C2 = Connection((B1, 2), (B3, 0))  

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
        C = Connection((B1, 2), (B2, 2)) 
        B1.outputs[2] = 3
        C.update()
        self.assertEqual(B2.inputs[2], 3)

        #test data transfer with mixed ports
        C = Connection(B1, (B2, 2)) 
        B1.outputs[0] = 3
        C.update()
        self.assertEqual(B2.inputs[2], 3)

        C = Connection((B1, 2), B2) 
        B1.outputs[2] = 3
        C.update()
        self.assertEqual(B2.inputs[0], 3)


    def test_update_multi(self):

        B1, B2, B3 = Block(), Block(), Block()

        #test data transfer with default ports
        C = Connection(B1, B2, B3) 
        B1.outputs[0] = 3
        C.update()
        self.assertEqual(B2.inputs[0], 3)
        self.assertEqual(B3.inputs[0], 3)

        #test data transfer with specific ports
        C = Connection((B1, 2), (B2, 2), (B3, 1)) 
        B1.outputs[2] = 3
        C.update()
        self.assertEqual(B2.inputs[2], 3)
        self.assertEqual(B3.inputs[1], 3)


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

        #test if ports assigned correctly
        self.assertEqual(D.source, (B1, 0))
        self.assertEqual(D.target, (B2, 0))

        #mixed
        D1 = Duplex(B1, (B2, 2))
        D2 = Duplex((B1, 3), B2)
        
        #test if ports assigned correctly
        self.assertEqual(D1.source, (B1, 0))
        self.assertEqual(D1.target, (B2, 2))

        #test if ports assigned correctly
        self.assertEqual(D2.source, (B1, 3))
        self.assertEqual(D2.target, (B2, 0))

        #all
        D = Duplex((B1, 4), (B2, 1))

        #test if ports assigned correctly
        self.assertEqual(D.source, (B1, 4))
        self.assertEqual(D.target, (B2, 1))

        #test too many
        with self.assertRaises(TypeError):
            D = Duplex((B1, 4), (B2, 1), (B2, 3))


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
        D = Duplex((B1, 3), (B2, 2)) 
        B1.outputs[3] = 5.5
        B2.outputs[2] = 12
        D.update()
        self.assertEqual(B1.inputs[3], 12)
        self.assertEqual(B2.inputs[2], 5.5)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)