#########################################################################################
##
##                                   Register Class
##                            (pathsim/utils/register.py)
##
##                                 Milan Rother 2025
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from bisect import insort


# CLASSES ===============================================================================

class Register:
    """This class is a intended to be used for the inputs and outputs of blocks. 

    Its basic functionality is similar to a `dict` but with some additional methods. 
    The core functionality is that values can be added dynamically and the size of the 
    register doesnt have to be specified. It also implements some methods to interact 
    with numpy arrays and to streamline convergence checks.
    
    Parameters
    ----------
    size : int, optional
        initial size of the register 

    Attributes
    ----------
    _values : dict[int, float]
        internal dict that stores the values of the register
    _sorted_keys : list[int]
        internal sorted list of port keys for fast ordered iterations of `_values`
    """

    __slots__ = ["_values", "_sorted_keys"]


    def __init__(self, size=1):
        self._values = {k:0.0 for k in range(size)}
        self._sorted_keys = list(range(size))


    def __len__(self):
        """Returns the number of register entries / ports."""
        return len(self._values)


    def __iter__(self):
        for k in self._sorted_keys:
            yield self._values[k]


    def reset(self):
        """Set all stored values of the register to zero."""
        for k in self._values.keys():
            self._values[k] = 0.0


    def clear(self):
        """Fully clear the register of all values to length is zero."""
        self._values.clear()
        self._sorted_keys.clear()
        

    def to_array(self):
        """Convert the register to a numpy array with entries 
        sorted by ports.

        Note
        ----
        This method is performance critical, since it gets called **A LOT** 
        and makes up a siginificant portion of all function calls during the 
        main simulation loop! Its already profiled and optimized, so be 
        careful with premature *improvements*.

        Returns
        -------
        arr : numpy.ndarray
            converted register as array
        """
        return np.array([self._values[k] for k in self._sorted_keys])


    def update_from_array(self, arr):
        """Update the register values from an array in place.

        Note
        ----
        This method is performance critical, since it gets called **A LOT** 
        and makes up a siginificant portion of all function calls during the 
        main simulation loop! Its already profiled and optimized, so be 
        careful with premature *improvements*.

        Parameters
        ----------
        arr : numpy.ndarray, float
            array or scalar that is used to update internal register values
        """
        if np.isscalar(arr):
            self._values[0] = arr
            return
            
        for k, a in enumerate(arr):
            if k not in self._values:
                insort(self._sorted_keys, k)
            self._values[k] = a
            

    def __setitem__(self, key, val):
        """Set the value of `_values`, wraps its setter method. 
        For direct access to the register values.

        Parameters
        ----------
        key : int
            port key, where to set value
        val : float, obj
            value to set at port
        """
        if key not in self._values:
            insort(self._sorted_keys, key) 
        self._values[key] = val


    def __getitem__(self, key):
        """Get the value of `_values`, wraps its getter method.
        For direct access to the register values.
        
        Parameters
        ----------
        key : int
            port key, where to get value from

        Returns
        -------
        out : float, obj
            value from port at `key` position
        """
        return self._values.get(key, 0.0)