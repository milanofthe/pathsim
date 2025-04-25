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
        for k in sorted(self._values.keys()):
            self._values[k] = 0.0


    def to_array(self):
        return np.array([
            self._values[k] for k in sorted(self._values.keys())
            ]).flatten()


    def update_from_array(self, arr):
        if np.isscalar(arr):
            self._values[0] = arr
        else:
            for k, a in enumerate(arr):
                self._values[k] = a


    def update_from_array_max_err(self, arr):
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
        self._values[key] = val


    def __getitem__(self, key):
        return self._values[key]