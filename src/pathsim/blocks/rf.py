"""
RF block
"""
import numpy as np
import skrf as rf
from inspect import signature

from .lti import TransferFunctionPRC

# TODO LIST
# class RFAmplifier Model amplifier in RF systems
# class Resistor/Capacitor/Inductor
# class RFMixer for mixer in RF systems?

class RFNetwork(TransferFunctionPRC):
    """
    N-port RF network linear time invariant (LTI) multi input multi output (MIMO) state-space model.

    Uses Vector Fitting for rational approximation of the frequency response using poles and residues.
    The resulting approximation has guaranteed stable poles that are real or come in complex conjugate pairs.

    Assumes N inputs and N outputs, where N is the number of ports of the RF network.

    Parameters
    ----------
    ntwk : :py:class:`~skrf.network.Network`
        scikit-rf [skrf]_ RF Network object


    References
    ----------
    .. [skrf] scikit-rf webpage https://scikit-rf.org/

    """
    def __init__(self, ntwk: rf.Network, auto_fit: bool = True,  **kwargs):
        # Select the vector fitting function from scikit-rf
        vf_fun_name = 'auto_fit' if auto_fit else 'vector_fit'
        vf_fun = getattr(rf.VectorFitting, vf_fun_name)
        # Filter kwargs for the selected vf function
        vf_fun_keys = signature(vf_fun).parameters
        vf_kwargs = {k: v for k, v in kwargs.items() if k in vf_fun_keys}
        # Apply vector fitting
        vf = rf.VectorFitting(ntwk)
        getattr(vf, vf_fun_name)(**vf_kwargs)
        # keep a copy of the network and VF
        self.network = ntwk
        self.vf = vf

        super().__init__(Poles=vf.poles.real,
                         Residues=vf.residues.reshape(len(vf.poles), ntwk.nports, ntwk.nports),
                         Const=vf.constant_coeff.reshape(ntwk.nports, ntwk.nports))
