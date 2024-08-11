#########################################################################################
##
##                              RF FILTERS (filters.py)
##
##                                Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from scipy.signal import butter, tf2ss

from math import factorial

from ..lti import StateSpace



# FILTER BLOCKS =========================================================================

class ButterworthLowpassFilter(StateSpace):

    """
    Direct implementation of a low pass butterworth filter block.

    Follows the same structure as the 'StateSpace' block in the 
    'pathsim.blocks' module. The numerator and denominator of the 
    filter transfer function are generated and then the transfer 
    function is realized as a state space model. 
    
    INPUTS : 
        Fc : (float) corner frequency of the filter in [Hz]
        n  : (int) filter order
    """

    def __init__(self, Fc, n):

        #filter parameters
        self.Fc = Fc
        self.n = n

        #use scipy.signal for filter design
        num, den = butter(n, 2*np.pi*Fc, btype="low", analog=True, output="ba")

        #initialize parent block
        super().__init__(*tf2ss(num, den))


class ButterworthHighpassFilter(StateSpace):

    """
    Direct implementation of a high pass butterworth filter block.

    Follows the same structure as the 'StateSpace' block in the 
    'pathsim.blocks' module. The numerator and denominator of the 
    filter transfer function are generated and then the transfer 
    function is realized as a state space model. 
    
    INPUTS : 
        Fc : (float) corner frequency of the filter in [Hz]
        n  : (int) filter order
    """

    def __init__(self, Fc, n):

        #filter parameters
        self.Fc = Fc
        self.n = n

        #use scipy.signal for filter design
        num, den = butter(n, 2*np.pi*Fc, btype="high", analog=True, output="ba")

        #initialize parent block
        super().__init__(*tf2ss(num, den))


class ButterworthBandpassFilter(StateSpace):

    """
    Direct implementation of a bandpass butterworth filter block.

    Follows the same structure as the 'StateSpace' block in the 
    'pathsim.blocks' module. The numerator and denominator of the 
    filter transfer function are generated and then the transfer 
    function is realized as a state space model. 
    
    INPUTS : 
        Fc : (list, tuple) corner frequencies of the filter in [Hz]
        n  : (int) filter order
    """

    def __init__(self, Fc, n):

        #filter parameters
        self.Fc = np.asarray(Fc)
        self.n = n

        if len(Fc) != 2:
            raise ValueError("'ButterworthBandpassFilter' requires two corner frequencies!")

        #use scipy.signal for filter design
        num, den = butter(n, 2*np.pi*self.Fc, btype="bandpass", analog=True, output="ba")

        #initialize parent block
        super().__init__(*tf2ss(num, den))


class ButterworthBandstopFilter(StateSpace):

    """
    Direct implementation of a bandstop butterworth filter block.

    Follows the same structure as the 'StateSpace' block in the 
    'pathsim.blocks' module. The numerator and denominator of the 
    filter transfer function are generated and then the transfer 
    function is realized as a state space model. 
    
    INPUTS : 
        Fc : (list, tuple) corner frequencies of the filter in [Hz]
        n  : (int) filter order
    """

    def __init__(self, Fc, n):

        #filter parameters
        self.Fc = np.asarray(Fc)
        self.n = n

        if len(Fc) != 2:
            raise ValueError("'ButterworthBandstopFilter' requires two corner frequencies!")

        #use scipy.signal for filter design
        num, den = butter(n, 2*np.pi*self.Fc, btype="bandstop", analog=True, output="ba")

        #initialize parent block
        super().__init__(*tf2ss(num, den))


class AllpassFilter(StateSpace):

    """
    Direct implementation of an Allpass filter using Pade approximants. 
    The transfer function of the ideal allpass is
    
        H(s) = exp(-sT) = exp(-sT/2) / exp(sT/2)

    where T is the time delay. This implementation uses Pade approximation 
    of the exponential to create a n-th order LTI statespace model that is 
    used for the numerical integration internally.
    
    INPUTS : 
        T : (float) time delay of the allpass in [s]
        n : (int) order of the pade approximation
    """

    def __init__(self, T, n=1):

        #filter parameters
        self.T = T
        self.n = n

        #taylor approximations for numerator and denominator
        num = [(-T/2)**i/factorial(i) for i in range(n+1)]
        den = [ (T/2)**i/factorial(i) for i in range(n+1)]

        #initialize parent block
        super().__init__(*tf2ss(num, den))