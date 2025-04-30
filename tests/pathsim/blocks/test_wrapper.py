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

    def test_raise_not_overcharged(self):
        W = Wrapper()
        self.assertRaises(NotImplementedError, W._run_wrapper)


    def test_overcharge(self):
    	class SinSample(Wrapper):
    		def _run_wrapper(self, x):
    			return np.sin(x)
    	W = SinSample()

    def test_wrapped_func(self):
        class SinSample(Wrapper):
            def _run_wrapper(self, x):
                return np.sin(x)

        W = SinSample()

        for t in range(10):
            self.assertEqual(W._run_wrapper(t), np.sin(t))

    def test_trigger_event_error(self):

        W = Wrapper()
        ev = W.events[0]
        self.assertRaises(TypeError, ev.resolve,0) # I don't believe is is expected ? 
        # Must be NotImplementedError no ?
    
    def test_update_event_tau(self):
        W = Wrapper()
        W.tau = 2
        ev = W.events[0]
        self.assertEqual(W.tau,ev.t_start)
    
    def test_update_event_period(self):
        W = Wrapper()
        W.T = 2
        ev = W.events[0]
        self.assertEqual(W.T,ev.t_period)
    
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
            def _run_wrapper(self, x):
                return np.sin(x)

        W = SinSample()
        ev = W.events[0]
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
