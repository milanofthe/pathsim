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


    def test_interrupt_early(self):
        """Test interrupting the tracker before completion."""
        n = 100
        interrupt_at = 50
        test_logger = self._get_test_logger("TestInterruptEarly")
        tracker = ProgressTracker(
            total_duration=1.0,
            update_log_every=0.10,
            description="Test Interrupt Early",
            logger=test_logger,
            min_log_interval=0.001
        )

        i = 0
        with tracker:
            for _ in tracker:
                i += 1
                progress = i / n
                tracker.update(progress=progress, success=True)

                # Trigger interrupt at 50%
                if i == interrupt_at:
                    tracker.interrupt()
                    break

        # Check that interrupt was registered
        self.assertTrue(tracker._interrupted)
        
        # Check stats reflect partial completion
        self.assertEqual(tracker.stats["total_steps"], interrupt_at)
        self.assertEqual(tracker.stats["successful_steps"], interrupt_at)
        
        # Tracker should be closed
        self.assertTrue(tracker._closed)
        

    def test_normal_completion_not_interrupted(self):
        """Test that normal completion doesn't set interrupt flag."""
        n = 100
        test_logger = self._get_test_logger("TestNoInterrupt")
        tracker = ProgressTracker(
            total_duration=1.0,
            update_log_every=0.10,
            description="Test No Interrupt",
            logger=test_logger,
            min_log_interval=0.001
        )

        i = 0
        with tracker:
            for _ in tracker:
                i += 1
                progress = i / n
                tracker.update(progress=progress, success=True)
                if i >= n:
                    break

        # Should NOT be interrupted
        self.assertFalse(tracker._interrupted)
        self.assertEqual(tracker.stats["total_steps"], n)
        self.assertTrue(tracker._closed)


    def test_interrupt_with_mixed_success(self):
        """Test interrupt with some failed steps before interruption."""
        n = 100
        interrupt_at = 60
        fail_before = 30
        test_logger = self._get_test_logger("TestInterruptMixed")
        tracker = ProgressTracker(
            total_duration=1.0,
            update_log_every=0.10,
            description="Test Interrupt Mixed",
            logger=test_logger,
            min_log_interval=0.001
        )

        i = 0
        with tracker:
            for _ in tracker:
                i += 1
                progress = i / n
                is_successful = (i > fail_before)
                tracker.update(progress=progress, success=is_successful)

                if i == interrupt_at:
                    tracker.interrupt()
                    break

        self.assertTrue(tracker._interrupted)
        self.assertEqual(tracker.stats["total_steps"], interrupt_at)
        self.assertEqual(tracker.stats["successful_steps"], interrupt_at - fail_before)
        self.assertTrue(tracker._closed)


    def test_invalid_total_duration(self):
        """Test that invalid total_duration raises ValueError"""
        test_logger = self._get_test_logger("TestInvalidDuration")

        # Zero duration
        with self.assertRaises(ValueError):
            ProgressTracker(total_duration=0, logger=test_logger)

        # Negative duration
        with self.assertRaises(ValueError):
            ProgressTracker(total_duration=-1.0, logger=test_logger)


    def test_invalid_update_log_every(self):
        """Test that invalid update_log_every raises ValueError"""
        test_logger = self._get_test_logger("TestInvalidUpdateLog")

        # Zero
        with self.assertRaises(ValueError):
            ProgressTracker(total_duration=1.0, update_log_every=0, logger=test_logger)

        # Greater than 1
        with self.assertRaises(ValueError):
            ProgressTracker(total_duration=1.0, update_log_every=1.5, logger=test_logger)

        # Negative
        with self.assertRaises(ValueError):
            ProgressTracker(total_duration=1.0, update_log_every=-0.1, logger=test_logger)


    def test_invalid_min_log_interval(self):
        """Test that negative min_log_interval raises ValueError"""
        test_logger = self._get_test_logger("TestInvalidMinLog")

        with self.assertRaises(ValueError):
            ProgressTracker(total_duration=1.0, min_log_interval=-1, logger=test_logger)


    def test_progress_clamping(self):
        """Test that progress is clamped to [0.0, 1.0]"""
        test_logger = self._get_test_logger("TestProgressClamping")
        tracker = ProgressTracker(total_duration=1.0, logger=test_logger, log=False)
        tracker.start()

        # Test clamping to 1.0
        tracker.current_progress = 1.5
        self.assertEqual(tracker.current_progress, 1.0)

        # Test clamping to 0.0
        tracker.current_progress = -0.5
        self.assertEqual(tracker.current_progress, 0.0)


    def test_update_before_start_warning(self):
        """Test that update before start() triggers warning and auto-starts"""
        test_logger = self._get_test_logger("TestUpdateBeforeStart")
        tracker = ProgressTracker(total_duration=1.0, logger=test_logger, log=False)

        # Update without calling start()
        with self.assertWarns(UserWarning):
            tracker.update(0.5)

        # Should have auto-started
        self.assertIsNotNone(tracker.start_time)


    def test_update_after_close_warning(self):
        """Test that update after close() triggers warning"""
        test_logger = self._get_test_logger("TestUpdateAfterClose")
        tracker = ProgressTracker(total_duration=1.0, logger=test_logger, log=False)
        tracker.start()
        tracker.close()

        # Update after close
        with self.assertWarns(UserWarning):
            tracker.update(0.5)


    def test_iterator_without_start_warning(self):
        """Test that using iterator without start() triggers warning"""
        test_logger = self._get_test_logger("TestIterNoStart")
        tracker = ProgressTracker(total_duration=1.0, logger=test_logger, log=False)

        # Use iterator without start()
        with self.assertWarns(UserWarning):
            iterator = tracker.__iter__()
            tracker.update(1.0)  # Exit loop


    def test_close_multiple_times(self):
        """Test that closing multiple times is safe"""
        test_logger = self._get_test_logger("TestMultipleClose")
        tracker = ProgressTracker(total_duration=1.0, logger=test_logger, log=False)
        tracker.start()

        # Close multiple times should be safe
        tracker.close()
        tracker.close()  # Should not raise

        self.assertTrue(tracker._closed)


    def test_logging_disabled(self):
        """Test tracker with logging disabled"""
        test_logger = self._get_test_logger("TestLoggingDisabled")
        tracker = ProgressTracker(total_duration=1.0, logger=test_logger, log=False)

        with tracker:
            for _ in tracker:
                tracker.update(1.0)
                break

        # Should complete without errors
        self.assertTrue(tracker._closed)


    def test_format_time_edge_cases(self):
        """Test _format_time with edge cases"""
        test_logger = self._get_test_logger("TestFormatTime")
        tracker = ProgressTracker(total_duration=1.0, logger=test_logger, log=False)

        # None
        self.assertEqual(tracker._format_time(None), "--:--")

        # Negative
        self.assertEqual(tracker._format_time(-1), "--:--")

        # Infinite
        self.assertEqual(tracker._format_time(float('inf')), "--:--")

        # Less than 60 seconds
        result = tracker._format_time(5.234)
        self.assertIn("s", result)

        # Between 60 and 3600 seconds
        result = tracker._format_time(125)  # 2:05
        self.assertIn(":", result)

        # More than 3600 seconds
        result = tracker._format_time(3665)  # 1:01:05
        parts = result.split(":")
        self.assertEqual(len(parts), 3)


    def test_format_rate_edge_cases(self):
        """Test _format_rate with edge cases"""
        test_logger = self._get_test_logger("TestFormatRate")
        tracker = ProgressTracker(total_duration=1.0, logger=test_logger, log=False)

        # None
        self.assertEqual(tracker._format_rate(None), "N/A")

        # Zero
        self.assertEqual(tracker._format_rate(0), "N/A")

        # Negative
        self.assertEqual(tracker._format_rate(-1), "N/A")

        # Very slow (< 0.1 it/s)
        result = tracker._format_rate(0.05)
        self.assertIn("it/min", result)

        # Slow (< 1 it/s)
        result = tracker._format_rate(0.5)
        self.assertIn("it/s", result)

        # Fast (>= 1 it/s)
        result = tracker._format_rate(100)
        self.assertIn("it/s", result)


    def test_render_bar(self):
        """Test _render_bar rendering"""
        test_logger = self._get_test_logger("TestRenderBar")
        tracker = ProgressTracker(total_duration=1.0, logger=test_logger, log=False, bar_width=10)

        # 0% progress
        bar = tracker._render_bar(0.0)
        self.assertEqual(bar, "-" * 10)

        # 50% progress
        bar = tracker._render_bar(0.5)
        self.assertEqual(bar, "#" * 5 + "-" * 5)

        # 100% progress
        bar = tracker._render_bar(1.0)
        self.assertEqual(bar, "#" * 10)


    def test_ema_alpha_clamping(self):
        """Test that ema_alpha is clamped to valid range"""
        test_logger = self._get_test_logger("TestEMAAlpha")

        # Too low
        tracker1 = ProgressTracker(total_duration=1.0, logger=test_logger, log=False, ema_alpha=0.0)
        self.assertEqual(tracker1.ema_alpha, 0.01)

        # Too high
        tracker2 = ProgressTracker(total_duration=1.0, logger=test_logger, log=False, ema_alpha=2.0)
        self.assertEqual(tracker2.ema_alpha, 1.0)

        # Valid
        tracker3 = ProgressTracker(total_duration=1.0, logger=test_logger, log=False, ema_alpha=0.5)
        self.assertEqual(tracker3.ema_alpha, 0.5)


    def test_stats_tracking(self):
        """Test that stats are tracked correctly"""
        test_logger = self._get_test_logger("TestStats")
        tracker = ProgressTracker(total_duration=1.0, logger=test_logger, log=False)

        with tracker:
            # Successful step
            tracker.update(0.25, success=True)
            self.assertEqual(tracker.stats["total_steps"], 1)
            self.assertEqual(tracker.stats["successful_steps"], 1)

            # Failed step
            tracker.update(0.5, success=False)
            self.assertEqual(tracker.stats["total_steps"], 2)
            self.assertEqual(tracker.stats["successful_steps"], 1)

            # Another successful
            tracker.update(1.0, success=True)
            self.assertEqual(tracker.stats["total_steps"], 3)
            self.assertEqual(tracker.stats["successful_steps"], 2)

        # Runtime should be set
        self.assertGreater(tracker.stats["runtime_ms"], 0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)