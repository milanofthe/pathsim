########################################################################################
##
##                                  TESTS FOR 
##                            'blocks.wrapper.py'
##
##                                   2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks import Wrapper

# TESTS ================================================================================
class TestWrapper(unittest.TestCase):
    """
    Test the implementation of the 'Wrapper' block class
    """

    def test_init(self):
        W = Wrapper()
    
    def test_init_with_dec(self):
        @Wrapper.dec(T=2, tau=0.5)
        def func1(a, b, c):
            return a + 1, b + 2, c + 3
        self.assertEqual(func1.T,2)
        self.assertEqual(func1.tau,0.5)
        self.assertEqual(func1.wrapped(1,2,3), (2, 4, 6))
    
    def test_init_with_func(self):
        def func(a, b, c):
            return a + 1, b + 2, c + 3
        func1 = Wrapper(func=func, T=2, tau=0.5)
        self.assertEqual(func1.T,2)
        self.assertEqual(func1.tau,0.5)
        self.assertEqual(func1.wrapped(1,2,3), (2, 4, 6))
    
    def test_init_with_func_as_class(self):
        class Func(Wrapper):
            def wrapped(self, a, b, c):
                return a + 1, b + 2, c + 3
        func1 = Func(T=2, tau=0.5)
        self.assertEqual(func1.T,2)
        self.assertEqual(func1.tau,0.5)
        self.assertEqual(func1.wrapped(1,2,3), (2, 4, 6))

    def test_raise_not_overcharged(self):
        W = Wrapper()
        with self.assertRaises(AttributeError):
            W.wrapped()

    def test_overcharge(self):
        class SinSample(Wrapper):
            def wrapped(self, x):
              return np.sin(x)
        W = SinSample()

    def test_wrapped_func(self):
        class SinSample(Wrapper):
            def wrapped(self, x):
                return np.sin(x)

        W = SinSample()

        for t in range(10):
            self.assertEqual(W.wrapped(t), np.sin(t))

    def test_trigger_event_error(self):

        W = Wrapper()
        ev = W.Evt
        with self.assertRaises(AttributeError):
            ev.resolve(0)
    
    def test_assert_update_event_tau(self):
        W = Wrapper()
        W.tau = 2
        ev = W.Evt
        ev_l = W.events[0]
        self.assertEqual(W.tau,ev.t_start)
        self.assertEqual(W.tau,ev_l.t_start)
    
    def test_assert_update_event_period(self):
        W = Wrapper()
        W.T = 2
        ev = W.Evt
        ev_l = W.events[0]
        self.assertEqual(W.T,ev.t_period)
        self.assertEqual(W.T,ev_l.t_period)
    
    def test_wrong_tau(self):
        W = Wrapper()
        with self.assertRaises(ValueError):
            W.tau = -1

    def test_wrong_period(self):
        W = Wrapper()
        with self.assertRaises(ValueError):
            W.T = -1
        

    def test_trigger_event(self):

        class SinSample(Wrapper):
            def wrapped(self, x):
                return np.sin(x)

        W = SinSample()
        ev = W.Evt
        ev.buffer(0)
        ev.resolve(0)
        
        de, cl, ra = ev.detect(0.1)
        self.assertFalse(de)
        self.assertFalse(cl)
        self.assertEqual(ra, 1)

        for t in range(1,10):
            de, cl, ra = ev.detect(t)
            self.assertTrue(de)
            self.assertTrue(cl)
            self.assertEqual(ra, 0)
            ev.buffer(t)
            ev.resolve(t)

            de, cl, ra = ev.detect(t+0.1)
            self.assertFalse(de)
            self.assertFalse(cl)
            self.assertEqual(ra, 1)





# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
