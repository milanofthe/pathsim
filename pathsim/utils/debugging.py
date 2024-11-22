########################################################################################
##
##                           DEBUGGING AND EVALUATION TOOLS  
##                                (utils/debugging.py)
##
##                                  Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import cProfile
import pstats

from time import perf_counter
from contextlib import ContextDecorator


# CLASSES ==============================================================================

class Timer:
    """
    context manager that times the execution time 
    of the code inside of the context in 'ms' for 
    debugging purposes
    """

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f"runtime: {self.time*1e3:.3f}ms"
        print(self.readout)


def timer(func):
    """
    shows the execution time in milliseconds of the 
    function object passed for debugging purposes
    """

    def wrap_func(*args, **kwargs):
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        print(f"Function '{func.__name__!r}' executed in {(t2 - t1)*1e3:.2f}ms")
        return result
    return wrap_func

    
class Profiler(ContextDecorator):
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
        








