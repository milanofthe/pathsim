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
    Source,
    Function,
    Adder, 
    Scope
    )

from pathsim.solvers import (
    SSPRK33,
    RKCK54
    )


# DUFFING OSCILLATOR ====================================================================

#initial position and velocity
x0, v0 = 0.0, 0.0

#driving angular frequency and amplitude
a, omega = 5.0, 2.0

#parameters (mass, damping, linear stiffness, nonlienar stiffness)
m, c, k, d = 1.0, 0.5, 1.0, 1.4

#blocks that define the system
I1 = Integrator(v0)                      # integrator for velocity
I2 = Integrator(x0)                      # integrator for position
A1 = Amplifier(c)
A2 = Amplifier(k)
A3 = Amplifier(-1/m)
F1 = Function(lambda x: d*x**3)          # nonlinear stiffness
Sr = Source(lambda t: a*np.sin(omega*t))
P1 = Adder()
Sc = Scope(labels=["velocity", "position"])

blocks = [I1, I2, A1, A2, A3, P1, F1, Sr, Sc]

#connections between the blocks
connections = [
    Connection(I1, I2, A1, Sc), 
    Connection(I2, F1), 
    Connection(F1, P1[2]), 
    Connection(Sr, P1[3]), 
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
    Solver=RKCK54,
    tolerance_lte_rel=1e-6,
    tolerance_lte_abs=1e-9
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(100)

    #plot the results directly from the scope
    Sc.plot()
    Sc.plot2D()

    plt.show()