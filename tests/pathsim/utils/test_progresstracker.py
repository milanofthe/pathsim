########################################################################################
##
##                                     TESTS FOR 
##                             'utils/progresstracker.py'
##
##                                Milan Rother 2023/24
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.utils.progresstracker import (
    ProgressTracker
    )


# TESTS ================================================================================

class TestProgressTracker(unittest.TestCase):
    """
    test the implementation of the 'ProgressTracker' class 
    """


    def test_iter_successful_5(self):

        #tracker with log interval 5/100 %
        tracker = ProgressTracker(log_interval=5)

        #test if display percentages are correctly computed
        self.assertEqual(tracker.display_percentages, 
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

        i, n = 0, 100

        #iterate the tracker
        for _ in tracker:
            i += 1

            #test tracker iteration condition
            self.assertTrue(tracker.condition)

            #check progress
            tracker.check(progress=i/n, success=True)

            #test tracker steps
            self.assertEqual(tracker.steps, i)

            #check successful steps tracker
            self.assertEqual(tracker.successful_steps, i)


    def test_iter_successful_10(self):

        #tracker with log interval 10/100 %
        tracker = ProgressTracker(log_interval=10)

        #test if display percentages are correctly computed
        self.assertEqual(tracker.display_percentages, 
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        i, n = 0, 100

        #iterate the tracker
        for _ in tracker:
            i += 1

            #test tracker iteration condition
            self.assertTrue(tracker.condition)

            #check progress
            tracker.check(progress=i/n, success=True)

            #test tracker steps
            self.assertEqual(tracker.steps, i)

            #check successful steps tracker
            self.assertEqual(tracker.successful_steps, i)


    def test_iter_mixed_success_5(self):

        #tracker with log interval 5/100 %
        tracker = ProgressTracker(log_interval=5)

        #test if display percentages are correctly computed
        self.assertEqual(tracker.display_percentages, 
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

        i, j, n = 0, 50, 100

        #iterate the tracker
        for _ in tracker:
            i += 1

            #test tracker iteration condition
            self.assertTrue(tracker.condition)

            #check progress
            tracker.check(progress=i/n, success=i>j)

            #test tracker steps
            self.assertEqual(tracker.steps, i)

            #check successful steps tracker
            self.assertEqual(tracker.successful_steps, max(0, i-j))


    def test_iter_mixed_success_10(self):

        #tracker with log interval 10/100 %
        tracker = ProgressTracker(log_interval=10)

        #test if display percentages are correctly computed
        self.assertEqual(tracker.display_percentages, 
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        i, j, n = 0, 50, 100

        #iterate the tracker
        for _ in tracker:
            i += 1

            #test tracker iteration condition
            self.assertTrue(tracker.condition)

            #check progress
            tracker.check(progress=i/n, success=i>j)

            #test tracker steps
            self.assertEqual(tracker.steps, i)

            #check successful steps tracker
            self.assertEqual(tracker.successful_steps, max(0, i-j))



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)