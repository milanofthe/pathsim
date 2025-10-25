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
import warnings

from .._constants import LOG_MIN_INTERVAL, LOG_UPDATE_EVERY
from .logger import LoggerManager


# HELPER CLASS =========================================================================

class ProgressTracker:
    """A progress tracker for simulations with adaptive ETA and step rate display.

    Uses exponential moving average for stable rate estimates and smart ETA calculation.
    Can be used as both an iterator and a context manager.

    Parameters
    ----------
    total_duration : float
        The total simulation duration to track against. Must be positive.
    description : str, optional
        Description for log messages. Defaults to "Progress".
    logger : logging.Logger, optional
        Logger instance. If None, uses LoggerManager. Defaults to None.
    log : bool, optional
        Enable logging. Defaults to True.
    log_level : int, optional
        Logging level. Defaults to logging.INFO.
    min_log_interval : float, optional
        Minimum seconds between logs. Defaults to LOG_MIN_INTERVAL.
    update_log_every : float, optional
        Log every N progress fraction (e.g., 0.2 = 20%). Defaults to LOG_UPDATE_EVERY.
    bar_width : int, optional
        Progress bar width in characters. Defaults to 20.
    ema_alpha : float, optional
        EMA smoothing factor (0-1), lower = more smoothing. Defaults to 0.3.
    """

    def __init__(
        self,
        total_duration,
        description="Progress",
        logger=None,
        log=True,
        log_level=logging.INFO,
        min_log_interval=LOG_MIN_INTERVAL,
        update_log_every=LOG_UPDATE_EVERY,
        bar_width=20,
        ema_alpha=0.3
        ):

        if total_duration <= 0:
            raise ValueError("total_duration must be positive")
        if not (0 < update_log_every <= 1):
            raise ValueError("update_log_every must be in (0, 1]")
        if min_log_interval < 0:
            raise ValueError("min_log_interval cannot be negative")

        self.total_duration = float(total_duration)
        self.description = description
        self.log = log
        self.log_level = log_level
        self.min_log_interval = min_log_interval
        self.update_log_every = update_log_every
        self.bar_width = bar_width
        self.ema_alpha = max(0.01, min(1.0, ema_alpha))

        #setup logger
        if logger is None:
            self.logger = LoggerManager().get_logger(f"progress.{self.description}")
        else:
            self.logger = logger

        #state tracking
        self.start_time = None
        self._progress = 0.0
        self._interrupted = False
        self._closed = False

        #stats
        self.stats = {"total_steps": 0, "successful_steps": 0, "runtime_ms": 0.0}

        #logging state
        self._last_log_time = 0.0
        self._last_log_progress = -self.update_log_every
        self._last_log_steps = 0
        self._last_logged_percentage = None

        #EMA tracking
        self._ema_progress_rate = None  #progress per second
        self._ema_step_rate = None      #steps per second
        self._last_update_time = None


    @property
    def current_progress(self):
        """Current progress fraction (0.0 to 1.0)"""
        return self._progress


    @current_progress.setter
    def current_progress(self, value):
        """Set progress, clamped to [0.0, 1.0]"""
        self._progress = max(0.0, min(1.0, float(value)))


    # context manager ------------------------------------------------------------------

    def __enter__(self):
        """Start tracker on context entry"""
        self.start()
        return self.__iter__()


    def __exit__(self, exc_type, exc_value, traceback):
        """Close tracker on context exit"""
        self.close()
        return False


    # iterator -------------------------------------------------------------------------

    def __iter__(self):
        """Iterate while progress < 1.0"""
        if self.start_time is None:
            warnings.warn("ProgressTracker iterator started before calling start()")
            self.start()
        while self.current_progress < 1.0:
            yield self


    # core methods ---------------------------------------------------------------------

    def start(self):
        """Start the progress tracker"""
        self.start_time = time.perf_counter()
        self._last_log_time = self.start_time
        self._last_update_time = self.start_time

        if self.log:
            self.logger.log(self.log_level,
                f"STARTING -> {self.description} (Duration: {self.total_duration:.2f}s)")


    def update(self, progress, success=True, **kwargs):
        """Update progress and optionally log

        Parameters
        ----------
        progress : float
            Progress fraction (0.0 to 1.0)
        success : bool, optional
            Whether this step was successful. Defaults to True.
        **kwargs
            Additional data (first key-value shown in logs if provided)
        """
        if self._closed:
            warnings.warn("ProgressTracker updated after being closed")
            return

        if self.start_time is None:
            warnings.warn("ProgressTracker updated before start()")
            self.start()

        current_time = time.perf_counter()

        #update stats
        self.stats["total_steps"] += 1
        if success:
            self.stats["successful_steps"] += 1

        #update progress
        old_progress = self._progress
        self.current_progress = progress

        #update EMA rates
        if self._last_update_time is not None:
            dt = current_time - self._last_update_time
            if dt > 1e-6:
                #calculate instantaneous rates
                progress_rate = (self._progress - old_progress) / dt
                step_rate = 1.0 / dt

                #apply EMA
                if self._ema_progress_rate is None:
                    self._ema_progress_rate = progress_rate
                    self._ema_step_rate = step_rate
                else:
                    self._ema_progress_rate = (self.ema_alpha * progress_rate +
                                               (1 - self.ema_alpha) * self._ema_progress_rate)
                    self._ema_step_rate = (self.ema_alpha * step_rate +
                                          (1 - self.ema_alpha) * self._ema_step_rate)

        self._last_update_time = current_time

        #log if needed
        self._log_progress()


    def interrupt(self):
        """Mark tracker as interrupted"""
        self._interrupted = True


    def close(self):
        """Close tracker and log final stats"""
        if self._closed:
            return

        if self.start_time is not None:
            runtime = time.perf_counter() - self.start_time
            self.stats["runtime_ms"] = runtime * 1000

            if self.log:
                status = "INTERRUPTED" if self._interrupted else "FINISHED"
                self.logger.log(self.log_level,
                    f"{status} -> {self.description} "
                    f"(total steps: {self.stats['total_steps']}, "
                    f"successful: {self.stats['successful_steps']}, "
                    f"runtime: {self.stats['runtime_ms']:.2f} ms)")

        self._closed = True


    # logging --------------------------------------------------------------------------

    def _log_progress(self):
        """Log progress if conditions met"""
        if not self.log or self.start_time is None:
            return

        current_time = time.perf_counter()

        #check if should log (skip initial 0% log)
        time_passed = (current_time - self._last_log_time) >= self.min_log_interval
        progress_milestone = self._progress >= (self._last_log_progress + self.update_log_every)

        if not (time_passed or progress_milestone):
            return

        #calculate display values
        elapsed = current_time - self.start_time
        percentage = int(self._progress * 100)

        #skip 0% and duplicate percentages
        if percentage == 0 or percentage == self._last_logged_percentage:
            return

        #ETA from EMA progress rate
        if self._ema_progress_rate and self._ema_progress_rate > 1e-6 and self._progress < 1.0:
            eta = (1.0 - self._progress) / self._ema_progress_rate
        else:
            eta = None

        #step rate from EMA
        step_rate = self._ema_step_rate if self._ema_step_rate else None

        #format and log
        bar = self._render_bar(self._progress)
        time_str = f"{self._format_time(elapsed)}<{self._format_time(eta)}"
        rate_str = self._format_rate(step_rate) if step_rate else "N/A"

        msg = f"{bar} {percentage:3d}% | {time_str} | {rate_str}"
        self.logger.log(self.log_level, msg)

        #update logging state
        self._last_log_time = current_time
        self._last_log_progress = (self._progress // self.update_log_every) * self.update_log_every
        self._last_logged_percentage = percentage


    def _render_bar(self, progress):
        """Render ASCII progress bar"""
        filled = int(progress * self.bar_width)
        empty = self.bar_width - filled
        return '#' * filled + '-' * empty


    def _format_time(self, seconds):
        """Format time adaptively: 5.2s, 05:23, or 01:23:45"""
        if seconds is None or seconds < 0 or not (0 <= seconds < float('inf')):
            return "--:--"

        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m:02d}:{s:02d}"
        else:
            h, m = divmod(int(seconds // 60), 60)
            s = int(seconds % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"


    def _format_rate(self, rate):
        """Format rate adaptively"""
        if rate is None or rate <= 0:
            return "N/A"

        if rate < 0.1:
            return f"{rate * 60:.1f} it/min"
        elif rate < 1:
            return f"{rate:.2f} it/s"
        else:
            return f"{rate:.1f} it/s"
