########################################################################################
##
##                             PROGRESS TRACKER CLASS DEFINITION 
##                                (utils/progresstracker.py)
##
##                                   Milan Rother 2023/24
##
########################################################################################

# IMPORTS ==============================================================================

from time import perf_counter
import logging


# HELPER CLASS =========================================================================

class ProgressTracker:
    """Class that manages progress tracking by providing a generator
    interface that runs until an external condition is satisfied.
    
    Parameters
    ----------
    logger : logging.Logger
        logger instance
    log_interval : int
        interval to log percentages, from 0 to 100    
    """

    def __init__(self, logger=None, log_interval=10, **kwargs):

        #set logger
        self.logger = logger or logging.getLogger(__name__)
        
        #generation condition
        self.condition = True
        
        #step counter
        self.steps = 0
        self.successful_steps = 0

        #tracker stats
        self.stats = {
            "total_steps"          : 0,
            "successful_steps"     : 0,
            "function_evaluations" : 0,
            "solver_iterations"    : 0,
            "runtime"              : 0
        }

        #update initial stats 
        for k, v in kwargs.items():
            if k in self.stats:
                self.stats[k] += v

        #for progress display in percent
        self.display_percentages = list(range(0, 101, log_interval))


    def __iter__(self):
        """Iterator interface for ProgressTracker"""

        #starting progress tracker
        if self.logger:
            self.logger.info("STARTING progress tracker")

        #computer time for performance estimate
        starting_time = perf_counter()

        #generate as long as 'self.condition' is 'True'
        while self.condition: 

            #count total steps
            self.stats["total_steps"] += 1
            yield 

        #compute tracker runtime and save it to stats
        runtime = perf_counter() - starting_time
        self.stats["runtime"] = runtime*1e3 #in ms

        #log the runtime
        if self.logger:
            self.logger.info(
                "FINISHED, steps(total)={}({}), runtime={}ms".format(
                    self.stats["successful_steps"],
                    self.stats["total_steps"],
                    round(self.stats["runtime"], 2)
                    )
                )


    def check(self, progress, success=False, **kwargs):
        """Update the progress of the generator. 

        This method needs to be called within the iteration loop 
        to update the looping condition and the internal tracking.
    
        Parameters
        ----------
        progress : float
            progress number between 0 and 1
        success : bool
            True if the update step was successful, False otherwise
        """

        #compute progress in percent (round to integer)
        percentage = int(100 * progress)

        #count successful steps
        self.stats["successful_steps"] += int(success)         

        #update stats (track evaluations, etc.)
        for k, v in kwargs.items():
            if k in self.stats:
                self.stats[k] += v

        #generation condition is progress less then 1
        self.condition = progress < 1.0

        #check if percentage can be displayed
        if percentage >= self.display_percentages[0]:
            self.display_percentages.pop(0)
            if self.logger:
                self.logger.info(f"progress={percentage:.0f}%")
