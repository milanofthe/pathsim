from pathsim.blocks import Function
from scipy.interpolate import LinearNDInterpolator, interp1d
import numpy as np

class LUT(Function):
    """
    Build internal n-d interpolant and treats it as a function. 
    Inherits from the `Function` block

    Parameters
    ----------
    points : ndarray of floats
        2-D array of data point coordinates, or a precomputed Delaunay triangulation 
    values : ndarray of float 
        N-D array of data values at points. 
    """

    def __init__(self, points, values):
        self.points = np.asarray(points)
        self.values = np.asarray(values)

        self.inter = LinearNDInterpolator(self.points, self.values)

        super().__init__(func=lambda *x :self.inter(x))

class LUT1D(Function):
    """
    Build internal 1-D interpolant and treats it as a function. 
    Inherits from the `Function` block

    Parameters
    ----------
    points : ndarray of floats
        1-D array of data point coordinates
    values : ndarray of float
        1-D or 2-D array of data values at points. If 2-D, each column represents
        a different output dimension.
    """

    def __init__(self, points, values):
        self.points = np.asarray(points).flatten()
        self.values = np.asarray(values)
        
        # Handle both 1D and 2D values
        if self.values.ndim == 1:
            # Single output dimension
            self.inter = interp1d(self.points, self.values, fill_value="extrapolate")
            def func(*x):
                return self.inter(x[0])
        else:
            # Multiple output dimensions - interpolate each column separately
            self.inter = interp1d(self.points, self.values, axis=0, fill_value="extrapolate")
            def func(*x):
                return self.inter(x[0])

        super().__init__(func=func)