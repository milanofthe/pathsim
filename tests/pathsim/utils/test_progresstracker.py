########################################################################################
##
##                        TESTS FOR (UPDATED)
##                     'utils/progresstracker.py'
##
##                        Milan Rother 2023/24
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import logging
import time 

from pathsim.utils.progresstracker import ProgressTracker


# TESTS ================================================================================

class TestProgressTracker(unittest.TestCase):
    """
    Test the implementation of the updated 'ProgressTracker' class
    """

    # Helper to get a logger for testing (avoids duplicate handlers)
    def _get_test_logger(self, name):
        logger = logging.getLogger(name)

        # Prevent messages propagating to root logger during tests if not desired
        logger.propagate = False

        # Ensure it has a handler, but maybe NullHandler to suppress output unless debugging
        if not logger.hasHandlers():
            logger.addHandler(logging.NullHandler())

        return logger

    def test_iter_successful_5_percent(self):
        """Test iteration with 100% success, logging every 5% progress."""
        n = 100
        test_logger = self._get_test_logger("TestIterSuccess5")
        tracker = ProgressTracker(
            total_duration=1.0,      # Corresponds to progress 0.0 to 1.0
            update_log_every=0.05,   # Log every 5%
            description="Test Success 5%",
            logger=test_logger,
            min_log_interval=0.001 # Allow frequent logs for testing steps
        )

        i = 0
        with tracker: # Use context manager
            for _ in tracker: # Use iterator protocol
                i += 1
                progress = i / n

                # Update progress tracker
                tracker.update(progress=progress, success=True)

                # Test tracker steps accumulated so far
                self.assertEqual(tracker.stats["total_steps"], i)
                # Check successful steps tracker
                self.assertEqual(tracker.stats["successful_steps"], i)

                # Check internal progress state (optional)
                self.assertAlmostEqual(tracker.current_progress, progress)

                # Explicit break needed as loop condition depends on tracker's state
                if i >= n:
                    break # Exit loop once 100% progress is reported

        # After loop and context exit, check final stats
        self.assertEqual(tracker.stats["total_steps"], n)
        self.assertEqual(tracker.stats["successful_steps"], n)
        self.assertTrue(tracker._closed) # Check if close was called
        self.assertAlmostEqual(tracker.current_progress, 1.0) # Should be 1.0 after close


    def test_iter_successful_10_percent(self):
        """Test iteration with 100% success, logging every 10% progress."""
        n = 100
        test_logger = self._get_test_logger("TestIterSuccess10")
        tracker = ProgressTracker(
            total_duration=1.0,      # Corresponds to progress 0.0 to 1.0
            update_log_every=0.10,   # Log every 10%
            description="Test Success 10%",
            logger=test_logger,
            min_log_interval=0.001
        )

        i = 0
        with tracker:
            for _ in tracker:
                i += 1
                progress = i / n
                tracker.update(progress=progress, success=True)

                self.assertEqual(tracker.stats["total_steps"], i)
                self.assertEqual(tracker.stats["successful_steps"], i)
                self.assertAlmostEqual(tracker.current_progress, progress)

                if i >= n:
                    break

        self.assertEqual(tracker.stats["total_steps"], n)
        self.assertEqual(tracker.stats["successful_steps"], n)
        self.assertTrue(tracker._closed)
        self.assertAlmostEqual(tracker.current_progress, 1.0)


    def test_iter_mixed_success_5_percent(self):
        """Test iteration with mixed success, logging every 5% progress."""
        n = 100
        j = 50 # Point after which steps are successful
        test_logger = self._get_test_logger("TestIterMixed5")
        tracker = ProgressTracker(
            total_duration=1.0,
            update_log_every=0.05,
            description="Test Mixed 5%",
            logger=test_logger,
            min_log_interval=0.001
        )

        i = 0
        with tracker:
            for _ in tracker:
                i += 1
                progress = i / n
                is_successful = (i > j) # Success only for i > 50

                tracker.update(progress=progress, success=is_successful)

                self.assertEqual(tracker.stats["total_steps"], i)
                # Check successful steps: only count when i > j
                self.assertEqual(tracker.stats["successful_steps"], max(0, i - j))
                self.assertAlmostEqual(tracker.current_progress, progress)

                if i >= n:
                    break

        self.assertEqual(tracker.stats["total_steps"], n)
        self.assertEqual(tracker.stats["successful_steps"], n - j) # 50 successful steps
        self.assertTrue(tracker._closed)
        self.assertAlmostEqual(tracker.current_progress, 1.0)


    def test_iter_mixed_success_10_percent(self):
        """Test iteration with mixed success, logging every 10% progress."""
        n = 100
        j = 50 # Point after which steps are successful
        test_logger = self._get_test_logger("TestIterMixed10")
        tracker = ProgressTracker(
            total_duration=1.0,
            update_log_every=0.10,
            description="Test Mixed 10%",
            logger=test_logger,
            min_log_interval=0.001
        )

        i = 0
        with tracker:
            for _ in tracker:
                i += 1
                progress = i / n
                is_successful = (i > j)

                tracker.update(progress=progress, success=is_successful)

                self.assertEqual(tracker.stats["total_steps"], i)
                self.assertEqual(tracker.stats["successful_steps"], max(0, i - j))
                self.assertAlmostEqual(tracker.current_progress, progress)

                if i >= n:
                    break

        self.assertEqual(tracker.stats["total_steps"], n)
        self.assertEqual(tracker.stats["successful_steps"], n - j)
        self.assertTrue(tracker._closed)
        self.assertAlmostEqual(tracker.current_progress, 1.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)