#########################################################################################
##
##                     PathSim Example for the Spectrum block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection

#the standard blocks are imported like this
from pathsim.blocks import (
    Scope, 
    Spectrum
    )

#special blocks (for example from 'rf' module) are imported like this
from pathsim.blocks.rf import (
    ButterworthLowpassFilter,
    GaussianPulseSource
    )

from pathsim.solvers import SSPRK33


# FREQUENCY DOMAIN RESPONSE OF A FILTER =================================================

#simulation timestep
dt = 0.001

#corner frequency of the filter
f = 5

#blocks that define the system
Src = GaussianPulseSource(f_max=5*f, tau=0.3)
FLT = ButterworthLowpassFilter(f, 8)
Sco = Scope(labels=["pulse", "filtered"])
Spc = Spectrum(freq=np.linspace(0, 2*f, 1000), 
               labels=["pulse", "filtered"])


blocks = [Src, FLT, Sco, Spc]

#the connections between the blocks
connections = [
    Connection(Src, FLT, Sco, Spc),
    Connection(FLT, Sco[1], Spc[1])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, dt=dt, log=True, Solver=SSPRK33)

#run the simulation for some time
Sim.run(2)

#plot the simulation results
Sco.plot()
Spc.plot()

#recover frequency response from spectrum block
freq, (G_pulse, G_filt) = Spc.read()
H_filt_sim = G_filt / G_pulse

#ideal frequency response from filter
def H(s):
    return np.dot(FLT.C, np.linalg.solve((s*np.eye(FLT.n) - FLT.A), FLT.B)) + FLT.D

H_filt_ideal = np.array([H(2j*np.pi*f) for f in freq]).flatten()

#plot the frequency domain responses (ideal and recovered)
fig, ax = plt.subplots(nrows=2, tight_layout=True, dpi=120)
ax[0].plot(freq, abs(H_filt_sim), label="recovered")
ax[0].plot(freq, abs(H_filt_ideal), "--", label="ideal")
ax[0].set_xlabel("freq")
ax[0].set_ylabel("mag")
ax[0].legend()

ax[1].plot(freq, np.angle(H_filt_sim), label="recovered")
ax[1].plot(freq, np.angle(H_filt_ideal), "--", label="ideal")
ax[1].set_xlabel("freq")
ax[1].set_ylabel("phase")
ax[1].legend()


plt.show()