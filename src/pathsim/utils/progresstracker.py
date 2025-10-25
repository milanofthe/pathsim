########################################################################################
##
##                      PROGRESS TRACKER CLASS DEFINITION  
##                          (utils/progresstracker.py)
##
#                               Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import logging
import time
import math
import warnings

from .._constants import LOG_MIN_INTERVAL, LOG_UPDATE_EVERY
from .logger import LoggerManager


# HELPER CLASS =========================================================================

class ProgressTracker:
    """A progress tracker suitable for simulations with variable timesteps,
    updated by progress fraction and integrated with standard logging.

    Logs progress updates periodically based on time and progress intervals.
    Calculates an estimated step rate based on the interval since the last log.
    Can be used as both an iterator and a context manager for convenient
    integration into simulation loops.

    Examples
    --------

    .. code-block::python

        tracker = ProgressTracker(total_duration=10.0)

        with tracker: # Handles start() and close()

            for _ in tracker: # Iterates until progress >= 1.0

                # ... perform simulation step ...

                current_time = get_current_sim_time()
                progress = current_time / 10.0

                tracker.update(progress, dt=current_dt)
                

    Parameters
    ----------
    total_duration : float
        The total simulation duration (e.g., in seconds) to track against.
        Must be positive.
    description : str, optional
        A short description or name for the process being tracked, used in
        log messages. Defaults to "Progress".
    logger : logging.Logger, optional
        The logger instance to use for outputting progress messages. If None,
        a logger is obtained from the LoggerManager singleton under the
        'progress.<description>' hierarchy. For testing, you can pass a custom
        logger instance.
    log : bool
        Flag to indicate if logging messages should be printed or not.
    log_level : int, optional
        The logging level (e.g., `logging.INFO`, `logging.DEBUG`) to use for
        progress messages. Defaults to `logging.INFO`.
    min_log_interval : float, optional
        The minimum real time interval in seconds that must pass between
        consecutive log updates, regardless of progress change. Helps to
        throttle output. Default from '_constants.py'.
    update_log_every : float, optional
        Log a message every time the progress increases by at least this
        fraction (e.g., 0.1 for 10%). Must be between 0 (exclusive) and
        1 (inclusive). Default from '_constants.py'.

    """

    def __init__(
        self,
        total_duration,
        description="",
        logger=None,
        log=True,
        log_level=logging.INFO,
        min_log_interval=LOG_MIN_INTERVAL,
        update_log_every=LOG_UPDATE_EVERY
        ):

        #input validation
        if total_duration <= 0:
            raise ValueError("total_duration must be positive")
        if not (0 < update_log_every <= 1):
            raise ValueError("update_log_every must be between 0 (exclusive) and 1 (inclusive)")
        if min_log_interval < 0:
            raise ValueError("min_log_interval cannot be negative")

        self.total_duration = float(total_duration)
        self.description = description if description else "Progress"
        self.log_level = log_level
        self.min_log_interval = min_log_interval
        self.update_log_every = update_log_every
        self.log = log

        #flag for interrupts
        self._interrupted = False

        #setup logger
        if logger is None:
            #use LoggerManager to get a logger under the progress hierarchy
            logger_mgr = LoggerManager()
            self.logger = logger_mgr.get_logger(f"progress.{self.description}")
        else:
            #use provided logger (for testing or custom setups)
            self.logger = logger

        #time tracking starts
        self.start_time = None

        #time of the last log message
        self.last_log_time = 0

        #store step count at last log for interval rate calculation
        self.last_log_total_steps = 0

        #progress milestone for percentage-based logging trigger
        #initialized below zero to ensure the first log (0%) happens
        self.last_log_progress_milestone = -self.update_log_every

        #internal progress state (0.0 to 1.0)
        self._current_progress = 0.0

        self.stats = {
            "total_steps": 0,
            "successful_steps": 0,
            "runtime_ms": 0.0
        }

        #latest dynamic info from update() kwargs
        self.postfix_dict = {}

        #flag to indicate if close() has been called
        self._closed = False


    # properties -----------------------------------------------------------------------

    @property
    def current_progress(self):
        """float: The current progress fraction (0.0 to 1.0)."""
        return self._current_progress


    @current_progress.setter
    def current_progress(self, value):
        """Sets and clamps the current progress between 0.0 and 1.0."""
        self._current_progress = max(0.0, min(1.0, float(value)))


    # context manager protocol ---------------------------------------------------------

    def __enter__(self):
        """Starts the tracker when entering a 'with' block.

        Returns
        -------
        iterator
            The tracker's iterator object (`self.__iter__()`).
        """
        self.start()
        return self.__iter__()


    def __exit__(self, exc_type, exc_value, traceback):
        """Closes the tracker when exiting a 'with' block, ensuring final log.

        Parameters
        ----------
        exc_type : type or None
            The type of exception raised (if any).
        exc_value : Exception or None
            The exception instance raised (if any).
        traceback : traceback or None
            The traceback object (if any).

        Returns
        -------
        bool
            False to indicate exceptions (if any) should be propagated.
        """
        self.close()
        return False


    # iterator protocol ----------------------------------------------------------------

    def __iter__(self):
        """Iterator protocol allowing use in 'for' loops. Yields control
         until `self.current_progress` reaches 1.0.

        Yields
        ------
        ProgressTracker
            Yields self to allow calling update() within the loop.
        """
        if self.start_time is None:
            warnings.warn("Tracker iterator started before entering 'with' block or calling start().")
            self.start() # Attempt to start if not already

        # Loop continues as long as progress is less than 100%
        while self.current_progress < 1.0:
            yield self # Yield control back to the calling loop


    # core methods ---------------------------------------------------------------------

    def interrupt(self):
        """Interrupts the progress tracking and marks for special logging.
        """
        self._interrupted = True


    def start(self):
        """Starts the progress timer and logs the initial message."""

        self.start_time = time.perf_counter()

        #last log time and step count reset
        self.last_log_time = self.start_time
        self.last_log_total_steps = 0 # Reset step count for rate calculation
        self.stats["total_steps"] = 0 # Ensure stats also start fresh if restart occurs
        self.stats["successful_steps"] = 0

        if self.log:
            self.logger.log(
                self.log_level,
                f"STARTING -> {self.description} (Duration: {self.total_duration:.2f}s)"
                )

        #log initial 0% state
        self._log_progress()


    def update(self, progress, success=True, **kwargs):
        """Updates the tracker's progress and optional postfix info,
        logging if necessary. Should be called within the loop iterating
        over the tracker.

        Parameters
        ----------
        progress : float
            Current progress fraction (0.0 to 1.0).
        success : bool, optional
            Indicates if the step contributing to this progress was successful.
            Defaults to True.
        **kwargs : dict, optional
            Key-value pairs to display as postfix information (e.g., dt=0.01).
            These overwrite previous postfix values each time update is called.
        """

        if self._closed:
            warnings.warn("ProgressTracker updated after being closed.")
            return

        if self.start_time is None:
            # This shouldn't happen if used correctly with 'with' and 'for'
            warnings.warn("ProgressTracker updated before start() or outside 'with' block.")
            self.start()

        #update stats first
        self.stats["total_steps"] += 1
        if success:
            self.stats["successful_steps"] += 1

        #update postfix dict
        self.postfix_dict = kwargs

        #set current progress
        self.current_progress = progress

        #trigger logging check
        self._log_progress()


    def close(self):
        """Modified to distinguish between normal finish and interrupt"""
        
        if not self._closed:
            
            if self.start_time is not None:
                
                # Calculate final runtime
                runtime = time.perf_counter() - self.start_time
                self.stats["runtime_ms"] = runtime * 1000
                
                # Log final progress
                self._log_progress(is_final=True)
                
                # Choose log message based on interrupt state
                if self.logger and self.log:
                    final_stats_str = (
                        f"total steps: {self.stats['total_steps']}, "
                        f"successful: {self.stats['successful_steps']}, "
                        f"runtime: {self.stats['runtime_ms']:.2f} ms"
                    )
                    
                    if self._interrupted:
                        log_msg = f"INTERRUPTED -> {self.description} ({final_stats_str})"
                    else:
                        log_msg = f"FINISHED -> {self.description} ({final_stats_str})"
                    
                    self.logger.log(self.log_level, log_msg)
            
            self._closed = True


    # helper methods -------------------------------------------------------------------

    def _format_time(self, seconds):
        """Formats a duration in seconds into HH:MM:SS string format.

        Parameters
        ----------
        seconds : float or None
            the duration in seconds

        Returns
        -------
        str
            Formatted time string (e.g., "00:01:35") or "--:--:--" if input is invalid.
        """

        #handle invalid inputs
        if seconds is None or math.isinf(seconds) or math.isnan(seconds) or seconds < 0:
            return "--:--:--"

        #conversion
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


    def _log_progress(self, is_final=False):
        """Internal method to format and log the current progress status if
        throttling conditions (time interval or progress milestone) are met,
        or if it's the final call. Calculates rate based on interval since last log.
        """

        #don't log if closed unless final
        if self._closed and not is_final: return

        #don't log if not started
        if self.start_time is None: return

        #don't log if no logger
        if not self.logger: return

        #don't log if logging not wanted
        if not self.log: return                

        current_time = time.perf_counter()

        #logging needed?
        progress_milestone_reached = (self.current_progress >= self.last_log_progress_milestone + self.update_log_every)
        time_interval_reached = (current_time - self.last_log_time >= self.min_log_interval)

        #first log after start()
        is_initial_log = (self.current_progress == 0 and self.last_log_progress_milestone < 0) 

        #log if final, or milestone reached, or time interval reached, or initial 0% log
        should_log = is_final or progress_milestone_reached or time_interval_reached or is_initial_log

        if should_log:

            elapsed_time = current_time - self.start_time
            percentage = self.current_progress * 100

            #calculate interval rate
            time_since_last_log = current_time - self.last_log_time
            steps_since_last_log = self.stats["total_steps"] - self.last_log_total_steps

            rate = 0.0
            rate_label = "steps/s"

            if is_initial_log:

                # For the very first log, interval rate is meaningless, show N/A or 0
                rate_str = "N/A steps/s"

            elif is_final:

                #for the final log -> overall average rate
                if elapsed_time > 1e-6 and self.stats["total_steps"] > 0:
                    rate = self.stats["total_steps"] / elapsed_time
                    rate_str = f"{rate:.1f} avg steps/s"

            else:
                #for intermediate logs, calculate interval rate
                if time_since_last_log > 1e-6 and steps_since_last_log > 0:
                    rate = steps_since_last_log / time_since_last_log
                    rate_str = f"{rate:.1f} {rate_label}"

                elif steps_since_last_log == 0 and time_since_last_log > 1e-6:
                    #0 rate if time passed but no steps
                    rate_str = f"0.0 {rate_label}"

                else:
                    #avoid division by zero or near-zero if log triggered rapidly
                    rate_str = "calc steps/s..." 

            #calculate ETA (based on percentage/time) - unchanged
            eta_seconds = None
            if self.current_progress > 1e-6 and elapsed_time > 0.1:
                eta_seconds = elapsed_time * (1.0 - self.current_progress) / self.current_progress
            
            eta_str = self._format_time(eta_seconds)
            elapsed_str = self._format_time(elapsed_time)

            #postfix
            postfix_str = ", ".join([f"{k}={v}" for k, v in self.postfix_dict.items() if v is not None])
            if postfix_str:
                postfix_str = " [" + postfix_str + "]"

            #assemble log message
            _msg = (
                f"{self.description}: {percentage:3.0f}% | "
                f"elapsed: {elapsed_str} (eta: {eta_str}) | "
                f"{self.stats['total_steps']} steps ({rate_str})"
                f"{postfix_str}"
                )

            self.logger.log(self.log_level, _msg)

            #update logging state
            self.last_log_time = current_time

            #update step count snapshot
            self.last_log_total_steps = self.stats["total_steps"]

            #update progress milestone marker if it was reached
            if progress_milestone_reached and not is_final:

                #ensure milestone advances correctly even if progress jumps multiple thresholds
                self.last_log_progress_milestone = math.floor(self.current_progress / self.update_log_every) * self.update_log_every

            elif is_initial_log:
                #mark 0% as logged if it was the trigger
                self.last_log_progress_milestone = 0.0