########################################################################################
##
##                                   TESTS FOR 
##                          'pathsim.events.schedule.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.events.schedule import (
    Schedule,
    ScheduleList
    )


# TESTS ================================================================================

class TestSchedule(unittest.TestCase):
    """
    Test the implementation of the 'Schedule' event class.
    """

    def test_init(self):

        S = Schedule(
            t_start=0.1, 
            t_end=200, 
            t_period=20 
            )

        self.assertEqual(S.t_start, 0.1)
        self.assertEqual(S.t_end, 200)
        self.assertEqual(S.t_period, 20)


    def test_next(self):

        S = Schedule(
            t_start=0, 
            t_period=20 
            )

        self.assertEqual(S._next(), 0)

        S.resolve(0)

        self.assertEqual(S._next(), 20)


    def test_estimate(self):

        S = Schedule(
            t_start=2, 
            t_period=20 
            )

        self.assertEqual(S.estimate(0), 2)
        self.assertEqual(S.estimate(1), 1)

        S.resolve(2)

        self.assertEqual(S.estimate(2), 20)
        self.assertEqual(S.estimate(13), 9)


    def test_detect(self):

        S = Schedule(
            t_start=2, 
            t_period=20 
            )

        S.buffer(0)

        d, c, r = S.detect(0)

        self.assertFalse(d)
        self.assertFalse(c)

        d, c, r = S.detect(4)

        self.assertTrue(d)
        self.assertFalse(c)
        self.assertEqual(r, 0.5)


class TestScheduleList(unittest.TestCase):
    """
    Test the implementation of the 'ScheduleList' event class.
    """

    def test_init(self):


        with self.assertRaises(ValueError):
            S = ScheduleList(times_evt=[1, 3, 5, 2, 7])

        S = ScheduleList(
            times_evt=[1, 3, 5, 7]
            )

        self.assertEqual(S.times_evt, [1, 3, 5, 7])


    def test_next(self):

        S = ScheduleList(
            times_evt=[1, 3, 5, 7]
            )

        self.assertEqual(S._next(), 1)

        S.resolve(1)

        self.assertEqual(S._next(), 3)

        S.resolve(3)

        self.assertEqual(S._next(), 5)


    def test_estimate(self):

        S = ScheduleList(
            times_evt=[1, 3, 5, 7]
            )

        self.assertEqual(S.estimate(0), 1)
        self.assertEqual(S.estimate(0.5), 0.5)

        S.resolve(1)

        self.assertEqual(S.estimate(1), 2)
        self.assertEqual(S.estimate(2), 1)


    def test_detect(self):

        S = ScheduleList(
            times_evt=[1, 3, 5, 7]
            )

        S.buffer(0)

        d, c, r = S.detect(0)

        self.assertFalse(d)
        self.assertFalse(c)

        d, c, r = S.detect(2)

        self.assertTrue(d)
        self.assertFalse(c)
        self.assertEqual(r, 0.5)

    def test_func_act_is_not_none(self):
        def func_act(_):
            pass

        event = ScheduleList(
            times_evt=[1, 2, 3], func_act=func_act
        )

        assert event.func_act is not None

        

# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
