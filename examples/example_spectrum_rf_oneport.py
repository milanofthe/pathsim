#########################################################################################
##
##                     PathSim Example for RF block
##
#########################################################################################

# IMPORTS ===============================================================================
import numpy as np
import matplotlib.pyplot as plt
try:
    import skrf as rf  # requires the scikit-rf package
except ImportError as e:
    raise ImportError("This example requires the scikit-rf package to be installed.")
# the standard blocks are imported like this
from pathsim import Simulation, Connection
from pathsim.blocks import Spectrum, GaussianPulseSource, RFNetwork
# from pathsim.blocks.rf import RFNetwork
from pathsim.solvers import RKBS32

# RF Network block is created from a scikit-rf Network object
# (here an example of a 1-port RF network included in scikit-rf)
# When creating a RFNetwork block, scikit-rf performs a vector-fitting
# of the N-port frequency response and deduces the state-space response.
rfntwk = RFNetwork(rf.data.ring_slot_meas)

# Gaussian pulse simulating an impulse response
# The scikit-rf Network object is passed as the 'network' parameter.
src = GaussianPulseSource(f_max=rfntwk.network.frequency.stop)

# Spectrum analyser setup with the start and stop frequencies of the RF network
spc = Spectrum(
    freq=rfntwk.network.f,
    labels=["pulse", "response"]
)

# create the system connections and simulation setup
sim = Simulation(
    blocks=[src, rfntwk, spc],
    connections=[
        Connection(src, rfntwk, spc[0]),
        Connection(rfntwk, spc[1])
    ],
    tolerance_lte_abs=1e-16, # this is due to the super tiny states
    tolerance_lte_rel=1e-5,  # so error control is dominated by the relative truncation error
    Solver=RKBS32,
)

sim.run(1e-9)

# model frequency response H(f) recovered from the spectrum block
freq, (G_pulse, G_filt) = spc.read()
H_filt_sim = G_filt / G_pulse

# plot the original S11 data, the vector-fitted model and the recovered frequency response
fig, ax = plt.subplots()
ax.plot(rfntwk.network.f/1e9, abs(rfntwk.network.s[:, 0, 0]), '.', label="S11 measurements", alpha=0.5)
ax.plot(freq/1e9, abs(rfntwk.vf.get_model_response(0, 0, freqs=freq)), lw=2, label="scikit-rf vector-fitting model")
ax.plot(freq/1e9, abs(H_filt_sim), '--', lw=2, label="pathsim impulse response")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("S11 magnitude")
ax.legend()
plt.show()
