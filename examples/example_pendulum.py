#########################################################################################
##
##                           PathSim example of a pendulum
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection

from pathsim.blocks import (
    Integrator, 
    Amplifier, 
    Function,
    Adder, 
    Scope
    )

from pathsim.solvers import RKCK54


# MATHEMATICAL PENDULUM =================================================================

#simulation timestep
dt = 0.1

#initial angle and angular velocity
phi0, omega0 = 0.99*np.pi, 0

#parameters (gravity, length)
g, l = 9.81, 1 

#blocks that define the system
In1 = Integrator(omega0) 
In2 = Integrator(phi0) 
Amp = Amplifier(-g/l) 
Fnc = Function(np.sin) 
Sco = Scope(labels=["angle", "angular velocity"])

blocks = [In1, In2, Amp, Fnc, Sco]

#connections between the blocks
connections = [
    Connection(In1, In2, Sco[1]), 
    Connection(In2, Fnc, Sco),
    Connection(Fnc, Amp), 
    Connection(Amp, In1)
    ]

# Create a simulation instance from the blocks and connections
Sim = Simulation(blocks, connections, dt=dt, log=True, Solver=RKCK54)

# Run the simulation for 25 seconds
Sim.run(duration=20)

# Plot the results directly from the scope
Sco.plot()

plt.show()