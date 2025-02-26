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

import cProfile
import pstats

from functools import wraps

from collections import deque

from time import perf_counter

from contextlib import ContextDecorator


# CLASSES ==============================================================================

class Timer:
    """context manager that times the execution time 
    of the code inside of the context in 'ms' for 
    debugging purposes
    
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



def track_block_runtime(cls):
    """
    Class decorator that adds runtime tracking to all public methods.
    Also adds method to get runtime estimates based on rolling average.
    """
    
    def wrap_method(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):

            #initialize history dict if not exists
            if not hasattr(self, "_runtime_history"):
                self._runtime_history = {}
            
            #initialize deque for this method if not exists    
            if method.__name__ not in self._runtime_history:
                self._runtime_history[method.__name__] = deque(maxlen=100)
                
            start = perf_counter()
            result = method(self, *args, **kwargs)
            self._runtime_history[method.__name__].append(perf_counter() - start)
            
            return result
        return wrapped

    #add runtime estimate method
    def get_runtime_estimate(self, method_name):
        if not hasattr(self, "_runtime_history"):
            return 0.0
        
        history = self._runtime_history.get(method_name, [])
        
        return np.mean(history) if history else 0.0
    
    cls.get_runtime_estimate = get_runtime_estimate
    
    #wrap all public methods
    for name, method in vars(cls).items():
        if callable(method) and not name.startswith("_"):
            setattr(cls, name, wrap_method(method))
    
    return cls

    
class Profiler(ContextDecorator):

    """Context manager for easy code profiling

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
        