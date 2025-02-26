
########################################################################################
##
##                         ADAPTIV BUFFER CLASS DEFINITION 
##                            (utils/adaptivebuffer.py)
##
##                                Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from collections import deque
from bisect import bisect_left


# HELPER CLASS =========================================================================

class AdaptiveBuffer:
    """A class that manages an adaptive buffer for delay modeling which is primarily 
    used in the pathsim 'Delay' block. 

    It implements a linear interpolation for arbitraty time lookup.
    
    Parameters
    ----------
    delay : float
        time delay in seconds
    
    Attributes
    ----------
    buffer : deque
        deque that collects the data for buffering
    counter : int
        count the number of lookups
    clean_every : int
        interval for buffer cleanup
    """

    def __init__(self, delay):

        #the buffer uses a double ended queue
        self.buffer = deque()
        self.delay = delay

        #for buffer cleanup every 100 lookups
        self.clean_every = 100
        self.counter = 0


    def __len__(self):
        return len(self.buffer)


    def add(self, t, value):
        """adding a new datapoint to the buffer

        Parameters
        ----------
        t : float
            time to add
        value : float, int, complex
            numerical value to add
        """

        #add the time-value tuple
        self.buffer.append((t, value))
        
        #clean up the buffer
        if self.counter > self.clean_every:
            self.counter = 0
            while len(self.buffer) > 1 and t > self.delay + self.buffer[0][0] :
                self.buffer.popleft()
        else:
            self.counter += 1

    
    def get(self, t):
        """lookup datapoint from buffer
    
        Parameters
        ----------
        t : float
            time for lookup
        """

        #default 0
        if not self.buffer:
            return 0.0
        
        #interpolation
        target_time = t - self.delay

        #requested time too small -> return first value
        if target_time <= self.buffer[0][0]:
            return self.buffer[0][1]
        
        #requested time too large -> return last value
        if target_time >= self.buffer[-1][0]:
            return self.buffer[-1][1]

        #find buffer index for requested time
        i = bisect_left(self.buffer, (target_time,))
        t0, y0 = self.buffer[i-1]
        t1, y1 = self.buffer[i]
        
        #linear interpolation
        return y0 + (y1 - y0) * (target_time - t0) / (t1 - t0)


    def clear(self):
        """clear the buffer, reset everything"""
        self.buffer = deque()
        self.counter = 0
