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
        self.assertEqual(C.source.ports, [0])
        self.assertTrue(isinstance(C.targets, list))

        #mixed
        C = Connection(B1[0], B2)
        self.assertEqual(C.source.ports, [0])
        self.assertEqual(C.targets[0].ports, [0])

        C = Connection(B1[2], B2)
        self.assertEqual(C.source.ports, [2])

        with self.assertRaises(ValueError):
            C = Connection(B1[0:3], B2)
        
        #all
        C = Connection(B1[2], B2[9])
        self.assertEqual(C.source.ports, [2])
        self.assertEqual(C.targets[0].ports, [9])

        C = Connection(B1[:8:3], B2[1:6:2])
        self.assertEqual(C.source.ports, [0, 3, 6])
        self.assertEqual(C.targets[0].ports, [1, 3, 5])


    def test_init_multi(self):

        B1, B2, B3 = Block(), Block(), Block()

        #default
        C = Connection(B1, B2, B3)
        self.assertTrue(isinstance(C.source, PortReference))
        self.assertEqual(C.source.ports, [0])
        self.assertTrue(isinstance(C.targets, list))
        self.assertEqual(len(C.targets), 2)
        self.assertEqual(C.targets[0].ports, [0])
        self.assertEqual(C.targets[1].ports, [0])

        #mixed
        C = Connection(B1[1], B2[3], B3)
        self.assertEqual(C.source.ports, [1])
        self.assertEqual(C.targets[0].ports, [3])
        self.assertEqual(C.targets[1].ports, [0])

        C = Connection(B1, B2[2], B3[0])
        self.assertEqual(C.source.ports, [0])
        self.assertEqual(C.targets[0].ports, [2])
        self.assertEqual(C.targets[1].ports, [0])

        #all
        C = Connection(B1[1:7], B3[3:9])
        self.assertEqual(C.source.ports, [1, 2, 3, 4, 5, 6])
        self.assertEqual(C.targets[0].ports, [3, 4, 5, 6, 7, 8])



    def test_overwrites(self):

        B1, B2, B3 = Block(), Block(), Block()

        #self overwrite

        C1 = Connection(B1, B2) 
        self.assertFalse(C1.overwrites(C1))

        #default ports

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

        #specific ports

        C1 = Connection(B1, B2, B3[1]) 
        C2 = Connection(B1, B3)  

        self.assertFalse(C1.overwrites(C2))
        self.assertFalse(C2.overwrites(C1))

        C1 = Connection(B1, B2, B3) 
        C2 = Connection(B1[2], B3)  

        self.assertTrue(C1.overwrites(C2))
        self.assertTrue(C2.overwrites(C1))

        C1 = Connection(B1, B2, B3[5]) 
        C2 = Connection(B1[2], B3)  

        self.assertFalse(C1.overwrites(C2))
        self.assertFalse(C2.overwrites(C1))    
            
        #test with sliced ports
        
        #test with tuple ports



    def test_update_single(self):

        B1, B2 = Block(), Block()

        #test data transfer with default ports
        C = Connection(B1, B2) 
        B1.outputs[0] = 3
        C.update()
        self.assertEqual(B2.inputs[0], 3)

        C = Connection(B1, B2[3]) 
        C.update()
        self.assertEqual(B2.inputs[3], 3)

        C = Connection(B1[1], B2[3]) 
        B1.outputs[1] = 2.5
        C.update()
        self.assertEqual(B2.inputs[3], 2.5)


    def test_update_multi(self):

        B1, B2, B3 = Block(), Block(), Block()

        #test data transfer with default ports
        C = Connection(B1, B2, B3) 
        B1.outputs[0] = 3
        C.update()
        self.assertEqual(B2.inputs[0], 3)
        self.assertEqual(B3.inputs[0], 3)

        #test data transfer with specific ports
        C = Connection(B1, B2[3], B3[2]) 
        B1.outputs[0] = 55
        C.update()
        self.assertEqual(B2.inputs[3], 55)
        self.assertEqual(B3.inputs[2], 55)

        #test with sliced ports
        
        #test with tuple ports



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
        self.assertTrue(isinstance(D.source, PortReference))
        self.assertTrue(isinstance(D.target, PortReference))
        self.assertTrue(isinstance(D.targets, list))
        self.assertEqual(len(D.targets), 2)
        self.assertEqual(D.source.ports, [0])
        self.assertEqual(D.target.ports, [0])

        #specific

        D = Duplex(B1[3], B2)
        self.assertEqual(D.source.ports, [3])
        self.assertEqual(D.target.ports, [0])

        D = Duplex(B1, B2[2])
        self.assertEqual(D.source.ports, [0])
        self.assertEqual(D.target.ports, [2])

        D = Duplex(B1[5], B2[1])
        self.assertEqual(D.source.ports, [5])
        self.assertEqual(D.target.ports, [1])

        #slicing
        
        D = Duplex(B1[1:4], B2[1:4])
        self.assertEqual(D.source.ports, [1, 2, 3])
        self.assertEqual(D.target.ports, [1, 2, 3])
        
        D = Duplex(B1[:3], B2[1:4])
        self.assertEqual(D.source.ports, [0, 1, 2])
        self.assertEqual(D.target.ports, [1, 2, 3])

        #test too many

        B3 = Block()
        with self.assertRaises(TypeError): 
            D = Duplex(B1[3], B2, B3)


    def test_update(self):

        B1, B2 = Block(), Block()

        #default

        D = Duplex(B1, B2) 
        B1.outputs[0] = 3
        B2.outputs[0] = 1
        D.update()
        self.assertEqual(B1.inputs[0], 1)
        self.assertEqual(B2.inputs[0], 3)

        #specific

        D = Duplex(B1[3], B2[1]) 
        B1.outputs[3] = 2
        B2.outputs[1] = -0.1
        D.update()
        self.assertEqual(B1.inputs[3], -0.1)
        self.assertEqual(B2.inputs[1], 2)

        #slicing

        D = Duplex(B1[1:4], B2[:3]) 
        B1.outputs = {0:33, 1:99, 2:44, 3:77, 4:11}
        B2.outputs = {0:0.33, 1:0.99, 2:0.44, 3:0.77, 4:0.11}
        D.update()
        self.assertEqual(B1.inputs[3], 0.44)
        self.assertEqual(B2.inputs[1], 44)




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)