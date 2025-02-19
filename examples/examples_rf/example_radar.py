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
    Adder,
    Spectrum,
    Delay
    )

#special blocks (for example from 'rf' module) are imported like this
from pathsim.blocks.rf import (
    ChirpSource, 
    ButterworthLowpassFilter,
    WhiteNoise
    )


# FMCW RADAR SYSTEM =====================================================================

#natural constants
c0 = 3e8

#chirp parameters
B = 5e9
T = 5e-7
f_min = 1e9

#simulation timestep
dt = 1e-11

#delay for target emulation
tau = 2e-9

#target distances
R = c0 * tau / 2

#frequencies for targets
f_trg = 2 * R * B / (T * c0)

#initialize blocks
Src = ChirpSource(f0=f_min, BW=B, T=T)
Add = Adder()
Wns = WhiteNoise(5e-3)
Dly = Delay(tau)
Mul = Multiplier()
Lpf = ButterworthLowpassFilter(f_trg*3, 2)
Spc = Spectrum(
    freq=np.logspace(6, 10, 500), 
    labels=["noisy chirp", "delay", "mixer", "lpf"]
    )
Sco = Scope(
    labels=["noisy chirp", "delay", "mixer", "lpf"]
    )

blocks = [Src, Add, Wns, Dly, Mul, Lpf, Spc, Sco]

#initialize connections
connections = [
    Connection(Src, Add[0]),
    Connection(Wns, Add[1]),
    Connection(Add, Dly, Mul, Sco, Spc),
    Connection(Dly, Mul[1], Sco[1], Spc[1]),
    Connection(Mul, Lpf, Sco[2], Spc[2]),
    Connection(Lpf, Sco[3], Spc[3])
]

#initialize simulation
Sim = Simulation(blocks, connections, dt=dt, log=True)

#run simulation for one chirp period
Sim.run(T)

#plot the recording of the scope
Sco.plot()

#plot the spectrum
Spc.plot()
Spc.ax.set_xscale("log")
Spc.ax.set_yscale("log")

plt.show()