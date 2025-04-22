#########################################################################################
##
##                     PathSim example of FitzHugh-Nagumo model
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, Integrator, Adder, Amplifier, Constant, Function
from pathsim.solvers import RKCK54, RKBS32


# FitzHugh-Nagumo System ================================================================

#parameters
a, b, tau, R, I_ext = 0.7, 0.8, 12.5, 1.0, 0.5

#system definition
Iv = Integrator()
Iw = Integrator()
F3 = Function(lambda x: 1/3 * x**3)
Ca = Constant(a)
CR = Constant(R*I_ext)
Gb = Amplifier(b)
Gt = Amplifier(1/tau)
Av = Adder("+--+")
Aw = Adder("++-")

Sc = Scope(labels=["v", "w"])

blocks = [Iv, Iw, F3, Ca, CR, Gb, Gt, Av, Aw, Sc]

#the connections between the blocks
connections = [
    Connection(Av, Iv),
    Connection(Gt, Iw),
    Connection(Aw, Gt),
    Connection(Iv, Av[0], Aw[0], F3, Sc[0]),
    Connection(Iw, Av[2], Gb, Sc[1]),
    Connection(F3, Av[1]),
    Connection(CR, Av[3]),
    Connection(Ca, Aw[1]),
    Connection(Gb, Aw[2])
    ]

Sim = Simulation(
    blocks, 
    connections, 
    dt=0.01, 
    Solver=RKCK54, 
    tolerance_lte_abs=1e-8, 
    tolerance_lte_rel=1e-6
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(200)

    #plotting
    Sc.plot(lw=2)
    Sc.plot2D(lw=2)

    plt.show()