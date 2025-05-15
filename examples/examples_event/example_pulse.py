#########################################################################################
##
##        PathSim example for small pulse with and without event mechanism
##
#########################################################################################

# IMPORTS ===============================================================================

import matplotlib.pyplot as plt
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, Source
from pathsim.blocks.rf import ButterworthLowpassFilter
from pathsim.solvers import SSPRK33, RKCK54
from pathsim.events import Schedule


# FAST PULSE ============================================================================

dt = 0.1

#filter bandwidth, order and signal frequency
B, n = 5, 4

#pulse parameters
tau1 = 1.0
tau2 = 1.05

#blocks that define the system
Src = Source(lambda t: int(t-tau1>=0) - int(t-tau2>=0))
Lpf = ButterworthLowpassFilter(B, n)
Sco = Scope(labels=["source", "output"])

blocks = [Src, Lpf, Sco]

#the connections between the blocks
connections = [
    Connection(Src, Lpf, Sco),
    Connection(Lpf, Sco[1])
    ]

#event to catch the pulse
E = Schedule(
    t_start=tau1, 
    t_end=tau2, 
    t_period=tau2-tau1, 
    # tolerance=1e-6
    )

events = [E]
# E.off() #switch event tracking

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, events, dt=dt, log=True, Solver=RKCK54)


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation 
    Sim.run(2)

    #plot the results from the scope directly
    Sco.plot(".-")

    plt.show()