########################################################################################
##
##                            DEBUGGING AND EVALUATION TOOLS  
##                                 (utils/analysis.py)
##
##                                  Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from time import perf_counter
from functools import wraps
from contextlib import ContextDecorator

try:
    import cProfile, pstats
    PROFILE_AVAILABLE = True
except ImportError:
    PROFILE_AVAILABLE = False


# CLASSES ==============================================================================

class Timer(ContextDecorator):
    """Context manager that times the execution time 
    of the code inside of the context in 'ms' for 
    debugging purposes.

    Example
    -------
    
    .. code-block:: python
        
        #time the code within the context
        with Timer() as T:
            complicated_function()
            
        #print the runtime in ms
        print(T)
    
    Parameters
    ----------
    verbose : bool
        flag for verbose output
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.time = None


    def __float__(self):
        return self.time


    def __repr__(self):
        if self.time is None: return None
        return f"{self.time*1e3:.3f}ms" 
        

    def __enter__(self):
        self._start = perf_counter()
        return self


    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self._start
        if self.verbose: print(self)


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
        with Timer(verbose=False) as T:
            result = func(*args, **kwargs)
        print(f"Function '{func.__name__!r}' executed in {T}")
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
        if not PROFILE_AVAILABLE:
            _msg = "'Profiler' not available, make sure 'cProfile' and 'pstats' is installed!"
            raise ImportError(_msg)
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
        