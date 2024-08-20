########################################################################################
##
##                                  TESTS FOR 
##                              'blocks.scope.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.scope import Scope, RealtimeScope


# TESTS ================================================================================

class TestScope(unittest.TestCase):
    """
    Test the implementation of the 'Scope' block class
    """

    def test_init(self):

        #test default initialization
        S = Scope()

        self.assertEqual(S.sampling_rate, None)
        self.assertEqual(S.t_wait, 0.0)
        self.assertEqual(S.labels, [])
        self.assertEqual(S.recording, {})

        #test specific initialization
        S = Scope(sampling_rate=1, t_wait=1.0, labels=["1", "2"])

        self.assertEqual(S.sampling_rate, 1)
        self.assertEqual(S.t_wait, 1.0)
        self.assertEqual(S.labels, ["1", "2"])


    def test_len(self):
        
        S = Scope()

        #no passthrough
        self.assertEqual(len(S), 0)


    def test_str(self):

        S = Scope()

        #test default str method
        self.assertEqual(str(S), "Scope")


    def test_reset(self):

        S = Scope()

        for t in range(10):

            S.set(0, t)
            S.sample(t)

        #test that we have some recording
        self.assertGreater(len(S.recording), 0)

        S.reset()

        #test if reset was successful
        self.assertEqual(S.recording, {})


    def test_sample(self):

        #single input default initialization
        S = Scope()

        for t in range(10):

            S.set(0, t)
            S.sample(t)

            #test most recent recording
            self.assertEqual(S.recording[t], t)

        #multi input default initialization
        S = Scope()

        for t in range(10):

            S.set(0, t)
            S.set(1, 2*t)
            S.set(2, 3*t)
            S.sample(t)

            #test most recent recording
            self.assertTrue(np.all(np.equal(S.recording[t], [t, 2*t, 3*t])))


    def test_read(self):

        _time = np.arange(10)

        #single input default initialization
        S = Scope()

        for t in _time:

            S.set(0, t)
            S.sample(t)

        time, result = S.read()

        #test if time was recorded correctly
        self.assertTrue(np.all(np.equal(time, _time)))

        #test if input was recorded correctly
        self.assertTrue(np.all(np.equal(result, _time)))

        #multi input default initialization
        S = Scope()

        for t in _time:

            S.set(0, t)
            S.set(1, 2*t)
            S.set(2, 3*t)
            S.sample(t)

        time, result = S.read()

        #test if time was recorded correctly
        self.assertTrue(np.all(np.equal(time, _time)))

        #test if multi input was recorded correctly
        self.assertTrue(np.all(np.equal(result, [_time, 2*_time, 3*_time])))


    def test_sampling_rate(self):

        _time = np.arange(10)

        #single input special sampling rate
        S = Scope(sampling_rate=0.5)

        for t in _time:

            S.set(0, t)
            S.sample(t)

        time, result = S.read()

        #test if time was recorded correctly
        self.assertTrue(np.all(np.equal(time, _time[1::2])))

        #test if input was recorded correctly
        self.assertTrue(np.all(np.equal(result, _time[1::2])))


    def test_t_wait(self):

        _time = np.arange(10)

        #single input special t_wait
        S = Scope(t_wait=5)

        for t in _time:

            S.set(0, t)
            S.sample(t)

        time, result = S.read()

        #test if time was recorded correctly
        self.assertTrue(np.all(np.equal(time, _time[5:])))

        #test if input was recorded correctly
        self.assertTrue(np.all(np.equal(result, _time[5:])))


class TestRealtimeScope(unittest.TestCase):
    """
    Test the implementation of the 'RealtimeScope' block class
    """

    pass #no tests implemented yet, since it just inherits from 'Scope' and is not critical


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)