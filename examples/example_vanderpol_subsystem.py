#########################################################################################
##
##                             PathSim Subsystem Example
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection, Interface, Subsystem
from pathsim.blocks import Integrator, Scope, Adder, Multiplier, Amplifier, Function
from pathsim.solvers import ESDIRK32, ESDIRK43, GEAR52A


# VAN DER POL OSCILLATOR INITIAL VALUE PROBLEM ==========================================

#initial condition
x1_0 = 2
x2_0 = 0

#van der Pol parameter
mu = 1000

#blocks that define the system
Sco = Scope(labels=["$x_1$", "$x_2$"])

#subsystem with two separate integrators to emulate ODE block
If = Interface()

I1 = Integrator(x1_0)
I2 = Integrator(x2_0)
Fn = Function(lambda a: 1 - a**2)
Pr = Multiplier()
Ad = Adder("-+")
Am = Amplifier(mu)

sub_blocks = [If, I1, I2, Fn, Pr, Ad, Am]
sub_connections = [
    Connection(I2, I1, Pr[0], If[1]), 
    Connection(I1, Fn, Ad[0], If[0]), 
    Connection(Fn, Pr[1]),
    Connection(Pr, Am),
    Connection(Am, Ad[1]),
    Connection(Ad, I2)
    ]

#the subsystem acts just like a normal block
VDP = Subsystem(sub_blocks, sub_connections)

#blocks of the main system
blocks = [VDP, Sco]

#the connections between the blocks in the main system
connections = [
    Connection(VDP, Sco),
    # Connection(VDP[1], Sco[1])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    Solver=GEAR52A, 
    tolerance_lte_abs=1e-5, 
    tolerance_lte_rel=1e-3, 
    tolerance_fpi=1e-8 
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    #run simulation for some number of seconds
    Sim.run(3*mu)

    Sco.plot(".-", lw=1.5)

    plt.show()