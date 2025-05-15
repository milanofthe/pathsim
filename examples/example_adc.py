#########################################################################################
##
##                                      Example with ADC
##
#########################################################################################

# IMPORTS ===============================================================================

import matplotlib.pyplot as plt
import numpy as np

from pathsim import Simulation, Connection

from pathsim.blocks import Scope, Source, ButterworthLowpassFilter, ADC, DAC

from pathsim.solvers import RKBS32


# EXAMPLE ===============================================================================

n = 8
fs = 50

omega = 2.0*np.pi

#blocks that define the system
src = Source(lambda t: np.sin(omega*t))
adc = ADC(n_bits=n, T=1/fs, span=[-1, 1])
dac = DAC(n_bits=n, T=1/fs, tau=0.1/fs, span=[-1, 1]) #dac has slight delay
lpf = ButterworthLowpassFilter(Fc=fs/5, n=1)

sco = Scope(labels=["src", "dac", "lpf"])

blocks = [src, dac, adc, lpf, sco]

#the connections between the blocks
connections = [
    Connection(src, adc, sco[0]),
    Connection(dac, lpf, sco[1]),
    Connection(lpf, sco[2]),
    ]

#digital connections
for i in range(n):
    connections.append(
        Connection(adc[i], dac[i])
        )

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    Solver=RKBS32,
    log=True
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation 
    Sim.run(2)
    Sim.plot()

    plt.show()


