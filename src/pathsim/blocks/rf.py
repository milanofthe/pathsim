"""
RF block
"""
import numpy as np
import skrf as rf
from inspect import signature
from pathlib import Path

from .lti import StateSpace


# TODO LIST
# class RFAmplifier Model amplifier in RF systems
# class Resistor/Capacitor/Inductor
# class RFMixer for mixer in RF systems?

class RFNetwork(StateSpace):
    """
    N-port RF network linear time invariant (LTI) multi input multi output (MIMO) state-space model.

    Uses Vector Fitting for rational approximation of the frequency response using poles and residues.
    The resulting approximation has guaranteed stable poles that are real or come in complex conjugate pairs.

    Assumes N inputs and N outputs, where N is the number of ports of the RF network.

    Parameters
    ----------
    ntwk : can be :py:class:`~skrf.network.Network`, str, Path, or file-object.
        scikit-rf [skrf]_ RF Network object, or file to load information from.
        Supported formats are touchstone file V1 (.s?p) or V2 (.ts).

    References
    ----------
    .. [skrf] scikit-rf webpage https://scikit-rf.org/

    """

    def __init__(self, ntwk: rf.Network | str | Path, auto_fit: bool = True, **kwargs):
        if isinstance(ntwk, Path) or isinstance(ntwk, str):
            ntwk = rf.Network(ntwk)

        # Select the vector fitting function from scikit-rf
        vf_fun_name = 'auto_fit' if auto_fit else 'vector_fit'
        vf_fun = getattr(rf.VectorFitting, vf_fun_name)
        # Filter kwargs for the selected vf function
        vf_fun_keys = signature(vf_fun).parameters
        vf_kwargs = {k: v for k, v in kwargs.items() if k in vf_fun_keys}
        # Apply vector fitting
        vf = rf.VectorFitting(ntwk)
        getattr(vf, vf_fun_name)(**vf_kwargs)
        A, B, C, D, _ = vf._get_ABCDE()
        # keep a copy of the network and VF
        self.network = ntwk
        self.vf = vf

        super().__init__(A, B, C, D)

    def s(self, freqs: np.ndarray) -> np.ndarray:
        """
        S-matrix of the vector fitted N-port model calculated from its state-space representation.

        Parameters
        ----------
        freqs : :py:class:`~numpy.ndarray`
            Frequencies (in Hz) at which to calculate the S-matrices.

        Returns
        -------
        s : :py:class:`~numpy.ndarray`
            Complex-valued S-matrices (fxNxN) calculated at frequencies `freqs`.
        """
        return rf.VectorFitting._get_s_from_ABCDE(freqs, self.A, self.B, self.C, self.D, 0)
