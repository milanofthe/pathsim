#########################################################################################
##
##                   PathSim Example for Simple FMCW Radar System 
##
#########################################################################################

# IMPORTS ===============================================================================

import matplotlib.pyplot as plt
import numpy as np

#the core functionalities can now be imported directly
from pathsim import Simulation, Connection

#the standard blocks are imported like this
from pathsim.blocks import (
    Multiplier,
    Scope, 
    Spectrum,
    Delay
    )

#special blocks (for example from 'rf' module) are imported like this
from pathsim.blocks.rf import ChirpSource


# FMCW RADAR SYSTEM =====================================================================

#natural constants
c0 = 3e8

#chirp parameters
B = 5e9
T = 0.5e-6
f_min = 1e9

#simulation timestep
dt = 5e-12

#delay for target emulation
tau = 4e-9

#target distances
R = c0 * tau / 2

#frequencies for targets
f_trg = 4 * R * B / (T * c0)

#initialize blocks
Src  = ChirpSource(f0=f_min, BW=B, T=T)
Dly  = Delay(tau)
Mul  = Multiplier()
Spc  = Spectrum(freq=np.linspace(0, f_trg*2, 500), labels=["chirp", "delay", "mixer"])
Sco  = Scope(labels=["chirp", "delay", "mixer"])

blocks = [Src, Dly, Mul, Spc, Sco]

#initialize connections
connections = [
    Connection(Src, Dly, Mul, Sco, Spc),
    Connection(Dly, Mul[1], Sco[1], Spc[1]),
    Connection(Mul, Sco[2], Spc[2])
]

#initialize simulation
Sim = Simulation(blocks, connections, dt, log=True)

#run simulation for one up chirp period
Sim.run(T/2)

#plot the recording directly
Sco.plot()
Spc.plot()

plt.show()