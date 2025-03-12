
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
    used in the pathsim 'Delay' block but might have future applications aswell.

    It implements a linear interpolation for arbitrary time lookup.
    
    Parameters
    ----------
    delay : float
        time delay in seconds
    
    Attributes
    ----------
    buffer_t : deque
        deque that collects the time data for buffering
    buffer_v : deque
        deque that collects the value data for buffering
    """

    def __init__(self, delay):

        #the buffer uses a double ended queue
        self.buffer_t = deque()
        self.buffer_v = deque()
        self.delay = delay


    def __len__(self):
        return len(self.buffer_t)


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
        self.buffer_t.append(t)
        self.buffer_v.append(value)
        
        #remove second to last value from buffer -> enable interpolation
        if len(self.buffer_t) > 1:
            while t - self.buffer_t[1] > self.delay:
                self.buffer_t.popleft()
                self.buffer_v.popleft()


    def interp(self, t):
        """interpolate buffer at defined lookup time
    
        Parameters
        ----------
        t : float
            time for interpolation

        Returns
        -------
        out : float, array
            interpolated value
        """

        #default 0
        if not self.buffer_t:
            return 0.0

        #requested time too small -> return first value
        if t <= self.buffer_t[0]:
            return self.buffer_v[0]
        
        #requested time too large -> return last value
        if t >= self.buffer_t[-1]:
            return self.buffer_v[-1]

        #find buffer index for requested time
        i = bisect_left(self.buffer_t, t)
        t0, t1 = self.buffer_t[i], self.buffer_t[i-1]
        y0, y1 = self.buffer_v[i], self.buffer_v[i-1]
    
        #linear interpolation
        return y0 + (y1 - y0) * (t - t0) / (t1 - t0)


    def get(self, t):
        """lookup datapoint from buffer with 
        delay at `t_lookup = t - delay`
    
        Parameters
        ----------
        t : float
            time for lookup with delay
        """
        return self.interp(t - self.delay)


    def clear(self):
        """clear the buffer, reset everything"""
        self.buffer_t.clear()
        self.buffer_v.clear()