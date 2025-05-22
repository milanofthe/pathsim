#########################################################################################
##
##                   PathSim Example of a spring-mass-damper system 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection

from pathsim.blocks import (
    Integrator, 
    Amplifier, 
    Adder, 
    Scope
    )

from pathsim.solvers import SSPRK33


# HARMONIC OSCILLATOR INITIAL VALUE PROBLEM =============================================

#initial position and velocity
x0, v0 = 2, 5

#parameters (mass, damping, spring constant)
m, c, k = 0.8, 0.2, 1.5

#blocks that define the system
I1 = Integrator(v0)   # integrator for velocity
I2 = Integrator(x0)   # integrator for position
A1 = Amplifier(c)
A2 = Amplifier(k)
A3 = Amplifier(-1/m)
P1 = Adder()
Sc = Scope(labels=["velocity", "position"])

blocks = [I1, I2, A1, A2, A3, P1, Sc]

#connections between the blocks
connections = [
    Connection(I1, I2, A1, Sc), 
    Connection(I2, A2, Sc[1]),
    Connection(A1, P1), 
    Connection(A2, P1[1]), 
    Connection(P1, A3),
    Connection(A3, I1)
    ]

#create a simulation instance from the blocks and connections
Sim = Simulation(
    blocks, 
    connections, 
    dt=0.1,
    Solver=SSPRK33
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    # Run the simulation for 30 seconds
    Sim.run(duration=30)

    # Plot the results directly from the scope
    Sc.plot(lw=2)
    Sc.plot2D(lw=2)

    plt.show()