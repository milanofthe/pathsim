#########################################################################################
##
##                                   Register Class
##                            (pathsim/utils/register.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from bisect import insort


# CLASSES ===============================================================================

class Register:
    """This class is a intended to be used for the inputs and outputs of blocks. 

    Its basic functionality is similar to a `dict` but with some additional methods 
    and implemented as a numpy array for fast data transfer. 
    
    The core functionality is that values can be added dynamically and the size of the 
    register doesnt have to be specified. It also implements some methods to interact 
    with numpy arrays and to streamline convergence checks.
    
    Parameters
    ----------
    size : int, optional
        initial size of the register 
    mapping : dict[str: int]
        string aliases for integer ports

    Attributes
    ----------
    _data : np.ndarray
        internal numpy array that holds the values
    _mapping : dict[str: int]
        internal mapping for port aliases from string to int (index)
    """
    
    __slots__ = ["_data", "_mapping"]
    
    def __init__(self, size=None, mapping=None, dtype=np.float64):
        self._data = np.zeros(1 if size is None else size, dtype=dtype)
        self._mapping = {} if mapping is None else mapping.copy()
    
    
    def _map(self, key):
        """Map string keys to integers defined in '_mapping'

        Parameters
        ----------
        key : int, str
            port key, to map to index

        Returns
        -------
        _key : int
            port index 
        """
        return self._mapping.get(key, key)
    

    def _expand_to(self, index):
        """Expands the internal numpy array up to an index"""
        new_size = max(index + 1, len(self._data) * 2)
        new_data = np.zeros(new_size, dtype=self._data.dtype)
        new_data[:len(self._data)] = self._data
        self._data = new_data
    

    def _get_max_index(self, key):
        """Identify max index from different key types."""
        if isinstance(key, int):
            return key
        elif isinstance(key, slice):
            return key.stop - 1 if key.stop is not None else -1
        elif isinstance(key, (list, tuple, np.ndarray)):
            indices = np.asarray(key, dtype=np.intp)
            return indices.max() if len(indices) > 0 else -1
        return -1
    

    def __len__(self):
        return len(self._data)
    

    def __iter__(self):
        """Iteration and unpacking into tuples or lists"""
        return iter(self._data)
    

    def __getitem__(self, key):
        """Get the value for direct access to the 
        register values.
        
        Parameters
        ----------
        key : int, str
            port key, where to get value from

        Returns
        -------
        out : float, obj
            value from port at `key` position
        """
        if isinstance(key, str):
            key = self._map(key)
            if not isinstance(key, int):
                return 0.0
        
        if isinstance(key, int):
            if key < 0 or key >= len(self._data):
                return 0.0
            return self._data[key]
        
        return self._data[key]
    

    def __setitem__(self, key, value):
        """Set the value at key index for direct access 
        to the register values.

        Parameters
        ----------
        key : int, str
            port key, where to set value
        val : float, obj
            value to set at port
        """
        if isinstance(key, str):
            if key not in self._mapping:
                self._mapping[key] = len(self._mapping)
            key = self._mapping[key]
        
        max_idx = self._get_max_index(key)
        if max_idx >= len(self._data):
            self._expand_to(max_idx)
        
        self._data[key] = value
    

    def reset(self):
        """Set all stored values to zero."""
        self._data[:] = 0.0
    

    def clear(self):
        """Clear the array to length zero."""
        self._data = np.zeros(0, dtype=self._data.dtype)
        self._mapping.clear()
    

    def to_array(self):
        """Returns a copy of the internal array.

        Returns
        -------
        arr : np.ndarray
            converted register as array
        """
        return self._data.copy()
    

    def update_from_array(self, arr):
        """Update the register values from an array in place.

        Parameters
        ----------
        arr : np.ndarray, float
            array or scalar that is used to update internal register values
        """
        if np.isscalar(arr):
            if len(self._data) == 0:
                self._data = np.array([arr], dtype=self._data.dtype)
            else:
                self._data[0] = arr
            return
        
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        
        if len(arr) > len(self._data):
            self._expand_to(len(arr) - 1)
        
        np.copyto(self._data[:len(arr)], arr)

    
    def __contains__(self, key):
        """Check if a key is in mapping or is valid integer index."""
        return key in self._mapping or isinstance(key, int)