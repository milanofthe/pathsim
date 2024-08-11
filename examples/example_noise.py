#########################################################################################
##
##               PathSim Example for Noise Sources from the RF toolbox
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection

#the standard blocks are imported like this
from pathsim.blocks import (
    Amplifier, 
    Adder, 
    Scope, 
    Spectrum
    )

#special blocks (for example from 'rf' module) are imported like this
from pathsim.blocks.rf import (
    ButterworthBandpassFilter,
    ButterworthLowpassFilter,
    SquareWaveSource, 
    WhiteNoise,
    OneOverFNoise
    )

from pathsim.solvers import SSPRK33, RKCK54



# FILTERING A SQUAREWAVE ================================================================

#simulation timestep
dt = 0.02

#fundamental frequency of square wave
f = 0.5

#blocks that define the system
Src = SquareWaveSource(frequency=f)
Ns1 = OneOverFNoise(spectral_density=0.1, sampling_rate=5)
Ns2 = WhiteNoise(spectral_density=0.05, sampling_rate=5)
FLT = ButterworthBandpassFilter((f-f/4, f+f/4), 4)
# FLT = ButterworthLowpassFilter(1.3*f+f/4, 2)
# FLT = Amplifier()
Add = Adder()
Sco = Scope(labels=["squarewave", 
                    "filter", 
                    "adder", 
                    "1/f noise", 
                    "white noise"])
Spc = Spectrum(freq=np.linspace(0, 5, 500), 
               labels=["squarewave", 
                       "filter", 
                       "adder", 
                       "1/f noise", 
                       "white noise"])


blocks = [Src, Ns1, Ns2, Add, FLT, Sco, Spc]

#the connections between the blocks
connections = [
    Connection(Src, Add, Sco, Spc),
    Connection(Ns1, Add[1], Sco[3]),
    Connection(Ns2, Add[2], Sco[4]),
    Connection(Add, FLT, Sco[2], Spc[2]),
    Connection(FLT, Sco[1], Spc[1])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, dt=dt, log=True, Solver=SSPRK33)

#run the simulation for some time
Sim.run(30/f)

Sco.plot()

Spc.plot()

plt.show()