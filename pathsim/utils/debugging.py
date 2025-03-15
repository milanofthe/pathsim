########################################################################################
##
##                           DEBUGGING AND EVALUATION TOOLS  
##                                (utils/debugging.py)
##
##                                  Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from collections import deque
from contextlib import ContextDecorator
from functools import wraps
from time import perf_counter

try:
    import cProfile, pstats
    PROFILE_AVAILABLE = True
except ImportError:
    PROFILE_AVAILABLE = False


# CLASSES ==============================================================================

class Timer:
    """Context manager that times the execution time 
    of the code inside of the context in 'ms' for 
    debugging purposes.

    Example
    -------
    
    .. code-block:: python
        
        #time the code within the context
        with Timer() as T:
            complicated_function()
            
        #print the runtime
        print(T.time)
    
    Parameters
    ----------
    verbose : bool
        flag for verbose output
    """
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f"{self.time*1e3:.3f}ms"
        if self.verbose:
            print("runtime:", self.readout)


def timer(func):
    """Shows the execution time in milliseconds of the 
    function object passed for debugging purposes
    
    Parameters
    ----------
    func : callable
        function to track execution time of
    """

    @wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        print(f"Function '{func.__name__!r}' executed in {(t2 - t1)*1e3:.2f}ms")
        return result
    return wrap_func

    
class Profiler(ContextDecorator):

    """Context manager for easy code profiling
    
    Example
    -------

    .. code-block:: python 
    
        #profile the code within the context
        with Profiler():
            complicated_function()

    Parameters
    ----------
    top_n : int
        track top n function calls
    sort_by : str
        method to sort function cally by
    """
    
    def __init__(self, top_n=50, sort_by="cumulative"):
        self.top_n = top_n
        self.sort_by = sort_by
        self.profiler = cProfile.Profile()
    
    def __enter__(self):
        self.profiler.enable()
        return self
    
    def __exit__(self, *exc):
        self.profiler.disable()
        stats = pstats.Stats(self.profiler)
        stats.strip_dirs()
        stats.sort_stats(self.sort_by)
        stats.print_stats(self.top_n)
        return False        
        