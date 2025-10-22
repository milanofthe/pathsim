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
    Delay,
    RealtimeScope,
    ChirpPhaseNoiseSource, 
    ButterworthLowpassFilter
    )


# FMCW RADAR SYSTEM =====================================================================

#natural constants
c0 = 3e8

#chirp parameters
B = 4e9
T = 5e-7
f_min = 1e9

#simulation timestep
dt = 1e-11

#delay for target emulation
tau = 2e-9

#target distances
R = c0 * tau / 2

#frequencies for targets
f_trg = 4 * R * B / (T * c0)

#initialize blocks
Src = ChirpPhaseNoiseSource(f0=f_min, BW=B, T=T)
Add = Adder()
Dly = Delay(tau)
Mul = Multiplier()
Lpf = ButterworthLowpassFilter(f_trg*3, 2)
Spc = Spectrum(
    freq=np.logspace(6, 10.5, 500), 
    labels=["chirp", "delay", "mixer", "lpf"]
    )
Sco = Scope(
    labels=["chirp", "delay", "mixer", "lpf"]
    )

blocks = [Src, Add,  Dly, Mul, Lpf, Spc, Sco]

#initialize connections
connections = [
    Connection(Src, Add[0]),
    Connection(Add, Dly, Mul, Sco, Spc),
    Connection(Dly, Mul[1], Sco[1], Spc[1]),
    Connection(Mul, Lpf, Sco[2], Spc[2]),
    Connection(Lpf, Sco[3], Spc[3])
]

#initialize simulation
Sim = Simulation(blocks, connections, dt=dt, log=True)


# Run Example ===========================================================================

if __name__ == "__main__":

    #run simulation for one chirp period
    Sim.run(T)

    #plot the recording of the scope
    Sco.plot()

    #plot the spectrum
    Spc.plot()
    Spc.ax.set_xscale("log")
    Spc.ax.set_yscale("log")

    Spc.ax.axvline(f_trg, ls="--", c="k")

    plt.show()