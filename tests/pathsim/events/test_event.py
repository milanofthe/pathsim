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

        #test default initialization (no event function)
        with self.assertRaises(ValueError):
            e = Event()


        #test default initialization (with event function)
        e = Event(func_evt=lambda *_: None)
        self.assertEqual(e.blocks, [])
        self.assertEqual(e.func_act, None)
        self.assertEqual(e.func_cbk, None)
        self.assertEqual(e.tolerance, 1e-4)
        self.assertEqual(e._history, None)
        self.assertTrue(e._active)


    def test_on_off(self):

        #activating and deactivating event tracking

        e = Event(func_evt=lambda *_: None)
        self.assertTrue(e._active)

        e.off()
        self.assertFalse(e._active)

        e.on()
        self.assertTrue(e._active)


    def test_len(self):

        e = Event(func_evt=lambda *_: None)
        self.assertEqual(len(e), 0)

        e._times = [1, 2, 3]
        self.assertEqual(len(e), 3)


    def test_iter(self):

        e = Event(func_evt=lambda *_: None)
        e._times = [1, 2, 3]

        for i, t in enumerate(e):
            self.assertEqual(i+1, t)


    def test_get(self):

        e = Event(func_evt=lambda *_: None)
        self.assertEqual(e._get(), ([], []))

        e = Event(blocks=[], func_evt=lambda *_: None)
        self.assertEqual(e._get(), ([], []))

        e = Event(blocks=[Block(), Block()], func_evt=lambda *_: None)
        _out, _sta = e._get()
        self.assertEqual(list(_out), [0.0, 0.0])
        self.assertEqual(_sta, ([], []))

        B1, B2 = Block(), Block()
        B1.outputs = {0: 23}
        B2.outputs = {0: 22, 1:12}

        e = Event(blocks=[B1, B2], func_evt=lambda *_: None)
        [_out1, _out2], _sta = e._get()
        self.assertEqual(_out1, np.array([23]))
        self.assertEqual(_out2[0], 22)
        self.assertEqual(_out2[1], 12)


    def test_evaluate(self):
        #default implementation of base class

        e = Event(func_evt=lambda *_: True)
        self.assertTrue(e._evaluate(0))

        e = Event(func_evt=lambda *_: False)
        self.assertFalse(e._evaluate(0))

        e = Event(func_evt=lambda y, x, t: t>3)
        self.assertTrue(e._evaluate(4))
        self.assertFalse(e._evaluate(2))


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
