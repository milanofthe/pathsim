########################################################################################
##
##                             PROGRESS TRACKER CLASS DEFINITION 
##                                (utils/progresstracker.py)
##
##                                   Milan Rother 2023/24
##
########################################################################################

# IMPORTS ==============================================================================

import logging
import time
import sys
import logging
import math
import warnings


# HELPER CLASS =========================================================================

class ProgressTracker:
    """
    A concise progress tracker suitable for simulations with variable timesteps,
    updated by progress fraction and integrated with standard logging.

    Logs progress updates periodically based on time and progress intervals.
    Can be used as both an iterator and a context manager:

    """

    def __init__(
        self, 
        total_duration, 
        description="", 
        logger=None,
        log_level=logging.INFO, 
        min_log_interval=2.0, 
        update_log_every=0.2
        ): 
        """
        Initializes the progress tracker.

        Parameters
        ----------
        total_duration : float
            The total simulation duration to track against.
        description : str, optional
            Prefix text for log messages.
        logger : logging.Logger, optional
            Logger instance to use. If None, creates a default logger printing to stdout.
        log_level : int, optional
            Logging level for progress messages (e.g., logging.INFO, logging.DEBUG).
        min_log_interval : float, optional
            Minimum time interval (seconds) between log updates.
        update_log_every : float, optional
            Log a message every time progress increases by this fraction (e.g., 0.1 for 10%).
        """
        if total_duration <= 0:
            raise ValueError("total_duration must be positive")
        if not (0 < update_log_every <= 1):
             raise ValueError("update_log_every must be between 0 (exclusive) and 1 (inclusive)")
        if min_log_interval < 0:
             raise ValueError("min_log_interval cannot be negative")

        self.total_duration = float(total_duration)
        self.description = description if description else "TRACKER"
        self.logger = logger or logging.getLogger(f"ProgressTracker_{description[:10]}") 

        #ensure logger has a handler if none exists to see output
        if not self.logger.hasHandlers() and not self._has_configured_handler(self.logger):
            # Configure logger only if it hasn't been configured elsewhere
            handler = logging.StreamHandler(sys.stdout) # Log to stdout
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO) # Default level if creating handler
            self.logger.propagate = False # Prevent duplicate messages if root logger also has handler

        self.log_level = log_level
        self.min_log_interval = min_log_interval
        self.update_log_every = update_log_every

        self.start_time = None
        self.last_log_time = 0
        # Initialize to slightly below zero to ensure the 0% mark is logged by first update
        self.last_log_progress_milestone = -self.update_log_every
        self._current_progress = 0.0 # Use property for clamping
        self.stats = { # Basic counters
            "total_steps": 0,
            "successful_steps": 0,
            "runtime_ms": 0.0
        }
        self.postfix_dict = {} # Store latest postfix info
        self._closed = False

    @staticmethod
    def _has_configured_handler(logger):
        """Check if logger or its ancestors have a configured handler."""
        l = logger
        while l:
            if l.handlers:
                # Check if any handler is configured (e.g., has a formatter)
                # This is a heuristic, might need refinement based on logging setup
                if any(h.formatter is not None for h in l.handlers):
                    return True
            if not l.propagate:
                break
            else:
                l = l.parent
        return False

    @property
    def current_progress(self):
        return self._current_progress

    @current_progress.setter
    def current_progress(self, value):
        # Clamp progress between 0 and 1
        self._current_progress = max(0.0, min(1.0, value))

    def __enter__(self):
        """Start timer upon entering context and return the iterator."""
        self.start()
        return self.__iter__() # Return the iterator object

    def __exit__(self, exc_type, exc_value, traceback):
        """Log final status upon exiting context."""
        self.close()
        # Return False to propagate exceptions if any occurred
        return False

    def __iter__(self):
        """Iterator protocol, yields until progress reaches 1.0."""
        if self.start_time is None:
             warnings.warn("Tracker iterator started before entering 'with' block or calling start().")
             self.start() # Attempt to start if not already

        while self.current_progress < 1.0:
            yield self # Yield control, allows external loop to run and call update()

        # Exiting the loop means progress >= 1.0

    def start(self):
        """Starts the timer and logs the initial message."""

        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.logger.log(
            self.log_level, 
            f"STARTING -> {self.description} (duration: {self.total_duration:.2f})"
            )

        #log initial 0% state immediately
        self._log_progress()

    def _format_time(self, seconds):
        """Helper to format seconds into H:M:S"""
        if (seconds is None or 
            math.isinf(seconds) or 
            math.isnan(seconds) or seconds < 0):
            return "--:--:--"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    def _log_progress(self, is_final=False):
        """Logs the current progress status if conditions are met."""

        #ensure logging doesn't happen before start() is called
        if self._closed and not is_final: return
        if self.start_time is None: return
        if not self.logger: return

        current_time = time.perf_counter()

        #determine if logging is needed
        progress_milestone_reached = (self.current_progress >= self.last_log_progress_milestone + self.update_log_every)
        time_interval_reached = (current_time - self.last_log_time >= self.min_log_interval)

        #log if it's the final update OR if enough progress OR enough time has passed
        # Also log if progress is exactly 0 and it hasn't been logged yet (initial state)
        should_log = (
            is_final 
            or progress_milestone_reached 
            or time_interval_reached 
            or (
                self.current_progress == 0 
                and self.last_log_progress_milestone < 0
                )
            )

        if should_log:
            elapsed_time = current_time - self.start_time
            percentage = self.current_progress * 100

            # --- Calculate Rate (Steps / Real Time) ---
            rate = 0.0
            if elapsed_time > 1e-6 and self.stats["total_steps"] > 0:
                rate = self.stats["total_steps"] / elapsed_time
            rate_str = f"{rate:.1f} steps/s"

            # --- Calculate ETA (based on percentage/time) ---
            eta_seconds = None
            # Avoid division by zero or large estimates at the very beginning
            if self.current_progress > 1e-6 and elapsed_time > 0.1:
                eta_seconds = elapsed_time * (1.0 - self.current_progress) / self.current_progress
            eta_str = self._format_time(eta_seconds)
            elapsed_str = self._format_time(elapsed_time)

            # --- Build Postfix ---
            postfix_str = ", ".join([f"{k}={v}" for k, v in self.postfix_dict.items() if v is not None])
            if postfix_str:
                postfix_str = " [" + postfix_str + "]"

            # --- Assemble Log Message ---
            log_message = (f"PROGRESS {percentage:3.0f}% | "
                           f"elapsed: {elapsed_str} (eta: {eta_str}) | "
                           f"{self.stats['total_steps']} steps ({rate_str})"
                           f"{postfix_str}")

            # --- Log ---
            self.logger.log(self.log_level, log_message)

            self.last_log_time = current_time

            # Update threshold for next progress log if a milestone was reached
            if progress_milestone_reached and not is_final:
                self.last_log_progress_milestone = math.floor(self.current_progress / self.update_log_every) * self.update_log_every
            elif self.current_progress == 0 and self.last_log_progress_milestone < 0:
                 self.last_log_progress_milestone = 0.0 # Mark 0% as logged


    def update(self, progress, success=True, **kwargs):
        """
        Update the tracker's progress and optional postfix info, logging if necessary.
        Should be called within the loop iterating over the tracker.

        Parameters
        ----------
        progress : float
            Current progress fraction (0.0 to 1.0).
        success : bool, optional
            Indicates if the step contributing to this progress was successful.
        **kwargs : dict, optional
            Key-value pairs to display as postfix information (e.g., dt=0.01).
            These overwrite previous postfix values.
        """
        if self._closed:
             warnings.warn("ProgressTracker updated after being closed.")
             return
        if self.start_time is None:
             # This shouldn't happen if used correctly with 'with' and 'for'
             warnings.warn("ProgressTracker updated before start() or outside 'with' block.")
             self.start()

        # Update stats first
        self.stats["total_steps"] += 1
        if success:
            self.stats["successful_steps"] += 1

        # Update postfix with latest kwargs
        self.postfix_dict = kwargs

        # Set current progress (triggers setter for clamping)
        self.current_progress = progress

        # Log progress (throttled inside _log_progress)
        self._log_progress()


    def close(self):
        """Logs the final status and marks the tracker as closed."""
        if not self._closed:
            if self.start_time is not None:
                 runtime = time.perf_counter() - self.start_time
                 self.stats["runtime_ms"] = runtime * 1000
                 # Ensure final progress is logged at 100%
                 self.current_progress = 1.0
                 self._log_progress(is_final=True)
                 # Log final stats summary
                 if self.logger:
                      final_stats_str = (f"total steps: {self.stats['total_steps']}, "
                                         f"successful: {self.stats['successful_steps']}, "
                                         f"runtime: {self.stats['runtime_ms']:.2f} ms")
                      self.logger.log(self.log_level, f"FINISHED -> {self.description} ({final_stats_str})")
            self._closed = True
