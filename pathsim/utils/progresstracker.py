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
    """
    Class that manages progress tracking by providing a generator
    interface that runs until an external condition is satisfied.
    """

    def __init__(self, logger=None, log_interval=10):

        #set logger
        self.logger = logger or logging.getLogger(__name__)
        
        #generation condition
        self.condition = True
        
        #step counter
        self.steps = 0
        self.successful_steps = 0

        #for progress display in percent
        self.display_percentages = list(range(0, 101, log_interval))


    def __iter__(self):

        #starting progress tracker
        if self.logger:
            self.logger.info("STARTING progress tracker")

        #computer time for performance estimate
        starting_time = perf_counter()

        #generate as long as 'self.condition' is 'True'
        while self.condition: 
            yield 

        #compute tracker runtime
        runtime = perf_counter() - starting_time

        #log the runtime
        if self.logger:
            self.logger.info(f"FINISHED steps(total)={self.successful_steps}({self.steps}) runtime={runtime*1e3:.2f}ms")


    def check(self, progress, success=False, msg=""):
        """
        Update the progress of the generator. 

        This method needs to be called within the iteration loop 
        to update the looping condition and the internal tracking.

        INPUTS :
            progress : (float) progress number between 0 and 1
            success  : (bool) was the update step successful?
            msg      : (string) additional logging message
        """

        #compute progress in percent (round to integer)
        percentage = int(100 * progress)

        #count successful steps
        self.successful_steps += int(success)
        
        #count total steps
        self.steps += 1

        #generation condition is progress less then 1
        self.condition = progress < 1.0

        #check if percentage can be displayed
        if percentage >= self.display_percentages[0]:
            self.display_percentages.pop(0)
            if self.logger:
                self.logger.info(f"progress={percentage:.0f}%"+msg)
