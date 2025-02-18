########################################################################################
##
##                                  TESTS FOR 
##                              'events._event.py'
##
##                               Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.events._event import Event

from pathsim.blocks._block import Block


# TESTS ================================================================================

class TestEvent(unittest.TestCase):
    """
    Test the implementation of the base 'Event' class.
    """

    def test_init(self):

        #test default initialization
        e = Event()
        self.assertEqual(e.func_evt, None)
        self.assertEqual(e.func_act, None)
        self.assertEqual(e.tolerance, 1e-4)
        self.assertEqual(e._history, (None, 0))
        self.assertTrue(e._active)

        #test specific initialization 
        e = Event(func_evt=lambda *_ : True, 
                  func_act=lambda *_ : True, 
                  tolerance=1e-6
                  )
        self.assertTrue(e.func_evt(0))
        self.assertTrue(e.func_act(0))
        self.assertEqual(e.tolerance, 1e-6)
        self.assertEqual(e._history, (None, 0))
        self.assertTrue(e._active)


    def test_on_off(self):

        #activating and deactivating event tracking

        e = Event()
        self.assertTrue(e._active)

        e.off()
        self.assertFalse(e._active)

        e.on()
        self.assertTrue(e._active)


    def test_bool(self):

        e = Event()
        self.assertTrue(e)

        #turn off
        e.off()
        self.assertFalse(e)

        #turn on again
        e.on()
        self.assertTrue(e)


    def test_len(self):

        e = Event()
        self.assertEqual(len(e), 0)

        e._times = [1, 2, 3]
        self.assertEqual(len(e), 3)


    def test_iter(self):

        e = Event()
        e._times = [1, 2, 3]

        for i, t in enumerate(e):
            self.assertEqual(i+1, t)


    def test_detect(self):
        #default implementation of base class

        e = Event(func_evt=lambda *_: None)
        de, cl, ra = e.detect(0)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)


    def test_resolve(self):
        #default implementation of base class

        e = Event(func_evt=lambda *_: None)
        for t in range(5):
            e.resolve(t)
            self.assertEqual(len(e), t+1)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
