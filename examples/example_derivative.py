#########################################################################################
##
##                    PathSim testing of the 'Differentiator' block
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source, 
    Integrator, 
    Differentiator, 
    Adder, 
    Scope, 
    )

from pathsim.solvers import (
    SSPRK33,
    RKCK54
    )


# TESTING INTEGRATOR AND DERIVATIVE =====================================================

#simulation timestep
dt = 0.02

#signal frequency
f = 1
omega = 2*np.pi*f

#blocks that define the system
Src = Source(lambda t: np.sin(omega*t))
Sri = Source(lambda t: -1/omega*np.cos(omega*t))
Srd = Source(lambda t: omega*np.cos(omega*t))
Int = Integrator(-1/omega)
Dif = Differentiator(f_max=100)
Sco = Scope(
    labels=[
        "sin", 
        "integrator", 
        "differentiator", 
        "reference integral", 
        "reference derivative"
        ]
    )

blocks = [Src, Sri, Srd, Int, Dif, Sco]

#the connections between the blocks
connections = [
    Connection(Src, Int, Dif, Sco[0]),
    Connection(Int, Sco[1]),
    Connection(Dif, Sco[2]),
    Connection(Sri, Sco[3]),
    Connection(Srd, Sco[4]),
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, dt=dt, log=True, Solver=RKCK54)


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(20/f)

    Sco.plot()

    plt.show()