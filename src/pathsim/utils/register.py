#########################################################################################
##
##                                   Register Class
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from collections import defaultdict


# CLASSES ===============================================================================

class Register:
    """This class is a intended to be used for the inputs and outputs of blocks. 

    Its basic functionality is similar to a `dict` but with some additional methods. 
    The core functionality is that values can be added dynamically and the size of the 
    register doesnt have to be specified. It also implements some methods to interact 
    with numpy arrays and to streamline convergence checks.
    
    Parameters
    ----------
    size : `int`, optional
        initial size of the register 

    Attributes
    ----------
    _values : `collections.defaultdict`
        internal dict that stores the values of the register
    """

    def __init__(self, size=1):
        self._values = defaultdict(
            float, 
            {k:0.0 for k in range(size)}
            )


    def __len__(self):
        return len(self._values)


    def __iter__(self):
        for k in sorted(self._values.keys()):
            yield self._values[k]


    def reset(self):
        """Set all stored values of the register to zero."""
        for k in self._values.keys():
            self._values[k] = 0.0


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
        arr : `numpy.ndarray`
            converted register as array
        """
        return np.array([
            self._values[k] for k in sorted(self._values.keys())
            ]).flatten()


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
        arr : `numpy.ndarray`, `float`
            array or scalar that is used to update internal register values
        """
        if np.isscalar(arr):
            self._values[0] = arr
        else:
            for k, a in enumerate(arr):
                self._values[k] = a


    def update_from_array_max_err(self, arr):
        """Update the register values from an array in place and compute 
        the maximum absolute deviation, which is used in the main simulation 
        loop for convergencechecks of the fixed point iterations. 
    
        Note
        ----
        This method is performance critical, since it gets called **A LOT** 
        and makes up a siginificant portion of all function calls during the 
        main simulation loop! Its already profiled and optimized, so be 
        careful with premature *improvements*.

        Parameters
        ----------
        arr : `numpy.ndarray`, `float`
            array or scalar that is used to update internal register values
        
        Returns
        -------
        err : `float`
            maximum absolute deviation from previous register values
        """
        if np.isscalar(arr):
            _err = abs(self._values[0] - arr)
            self._values[0] = arr
            return _err
        else:
            _max_err = 0.0
            for k, a in enumerate(arr):
                _err = abs(self._values[k] - a)
                if _err > _max_err: 
                    _max_err = _err
                self._values[k] = a
            return _max_err


    def __setitem__(self, key, val):
        """Set the value of `_values`, wraps its setter method. 
        For direct access to the register values.
        """
        self._values[key] = val


    def __getitem__(self, key):
        """Get the value of `_values`, wraps its getter method.
        For direct access to the register values.
        """
        return self._values[key]