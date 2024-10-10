#########################################################################################
##
##                             PathSim Subsystem Example
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection, Interface, Subsystem
from pathsim.blocks import Integrator, Scope, Function
from pathsim.solvers import ESDIRK54, ESDIRK85


# VAN DER POL OSCILLATOR INITIAL VALUE PROBLEM ==========================================

#initial condition
x1_0 = 2 
x2_0 = 0

#van der Pol parameter
mu = 1000

#simulation timestep
dt = 0.05

#blocks that define the system
Sco = Scope()

#subsystem with two separate integrators to emulate ODE block
If = Interface()
I1 = Integrator(x1_0)
I2 = Integrator(x2_0)
Fn = Function(lambda x1, x2: mu*(1 - x1**2)*x2 - x1)

sub_blocks = [If, I1, I2, Fn]
sub_connections = [
    Connection(I2, I1, Fn[1], If[1]), 
    Connection(I1, Fn, If), 
    Connection(Fn, I2)
    ]

#this is subsystem acts just like a normal block
VDP = Subsystem(sub_blocks, sub_connections)

#blocks of the main system
blocks = [VDP, Sco]

#the connections between the blocks in the main system
connections = [
    Connection(VDP, Sco),
    Connection(VDP[1], Sco[1])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, dt=dt, log=True, Solver=ESDIRK54, tolerance_lte_abs=1e-6, tolerance_lte_rel=1e-4)

#run simulation for some number of seconds
Sim.run(3*mu)

Sco.plot(".-", lw=1.5)

plt.show()