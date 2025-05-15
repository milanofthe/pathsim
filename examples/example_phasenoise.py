#########################################################################################
##
##                PathSim Example for Sinusoidal Sources with Phasenoise
##
#########################################################################################

# IMPORTS ===============================================================================

import matplotlib.pyplot as plt
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, Spectrum, SinusoidalPhaseNoiseSource


# NOISY SOURCES =========================================================================

f = 2
dt = 0.01

sr1 = SinusoidalPhaseNoiseSource(frequency=f, sig_cum=0, sig_white=0, sampling_rate=20)
sr2 = SinusoidalPhaseNoiseSource(frequency=f, sig_cum=0, sig_white=0.1, sampling_rate=20)
sr3 = SinusoidalPhaseNoiseSource(frequency=f, sig_cum=0.5, sig_white=0, sampling_rate=20)
sr4 = SinusoidalPhaseNoiseSource(frequency=f, sig_cum=0.5, sig_white=0.1, sampling_rate=20)

sco = Scope(labels=["signal", "white", "integral", "both"])
spc = Spectrum(freq=np.linspace(0.5*f,1.5*f, 501), 
               labels=["signal", "white", "integral", "both"])

blocks = [sr1, sr2, sr3, sr4, sco, spc]

connections = [
    Connection(sr1, sco, spc),
    Connection(sr2, sco[1], spc[1]),
    Connection(sr3, sco[2], spc[2]),
    Connection(sr4, sco[3], spc[3])
    ]

Sim = Simulation(blocks, connections, dt=dt, log=True)


# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(100/f)

    sco.plot()
    spc.plot()

    plt.show()