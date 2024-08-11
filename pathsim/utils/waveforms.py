########################################################################################
##
##                              WAVEFORM FUNCTIONS (utils/waveforms.py)
##
##                                   Milan Rother 2023/24
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np


# WAVEFORMS ============================================================================

def gaussian(t, f_max):
    """
    gaussian pulse with its maximum at t=0
    """
    tau = 0.5 / f_max
    return np.exp(-(t/tau)**2)


def triangle_wave(t, f):
    """
    triangle wave with amplitude '1' and frequency 'f'
    """
    return 2 * abs(t*f - np.floor(t*f + 0.5)) - 1


def square_wave(t, f):
    """
    square wave with amplitude '1' and frequency 'f'
    """
    return np.sign(np.sin(2*np.pi*f*t))

