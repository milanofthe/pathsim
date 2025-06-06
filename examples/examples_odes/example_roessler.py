#########################################################################################
##
##                     PathSim Example for Rössler System
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, Integrator, Constant, Adder, Amplifier, Multiplier
from pathsim.solvers import RKCK54, RKBS32, RKV65


# RÖSSLER SYSTEM ========================================================================

# parameters a, b, c
a, b, c = 0.2, 0.2, 5.7

#initial conditions
x0, y0, z0 = 1.0, 1.0, 1.0

#integrators store the state variables x, y, z
itg_x = Integrator(x0) # dx/dt = -y - z
itg_y = Integrator(y0) # dy/dt = x + a*y
itg_z = Integrator(z0) # dz/dt = b + z*(x - c)

#components for dx/dt
add_neg_yz = Adder("--") # Computes -y - z

#components for dy/dt
amp_a = Amplifier(a)     # Computes a*y
add_x_ay = Adder("++")    # Computes x + (a*y)

#components for dz/dt
cns_b = Constant(b)
cns_c = Constant(c)
add_x_c = Adder("+-")     # Computes x - c
mul_z_xc = Multiplier()   # Computes z * (x - c)
add_b_zxc = Adder("++")   # Computes b + [z * (x - c)]

#scope for plotting
sco = Scope(labels=["x", "y", "z"])

#list of all blocks
blocks = [
    itg_x, itg_y, itg_z,
    add_neg_yz,
    amp_a, add_x_ay,
    cns_b, cns_c, add_x_c, 
    mul_z_xc, add_b_zxc,
    sco
    ]

# Connections
connections = [
    # Output signals (from integrators)
    Connection(itg_x, add_x_ay[0], add_x_c[0], sco[0]),    # x connects to: (x + ay), (x - c), scope
    Connection(itg_y, add_neg_yz[0], amp_a, sco[1]),       # y connects to: (-y - z), a*y, scope
    Connection(itg_z, add_neg_yz[1], mul_z_xc[0], sco[2]), # z connects to: (-y - z), z*(x - c), scope

    # dx/dt path: -y - z -> itg_x
    Connection(add_neg_yz, itg_x),       # -y - z -> integrator x

    # dy/dt path: x + a*y -> itg_y
    Connection(amp_a, add_x_ay[1]),      # a*y -> x + (a*y) input 1
    Connection(add_x_ay, itg_y),         # x + a*y -> integrator y

    # dz/dt path: b + z*(x - c) -> itg_z
    Connection(cns_b, add_b_zxc[0]),     # b -> b + [...] input 0
    Connection(cns_c, add_x_c[1]),       # c -> x - c input 1
    Connection(add_x_c, mul_z_xc[1]),    # (x - c) -> z * (x - c) input 1
    Connection(mul_z_xc, add_b_zxc[1]),  # z * (x - c) -> b + [z * (x - c)] input 1
    Connection(add_b_zxc, itg_z)         # b + z*(x - c) -> integrator z
    ]


Sim = Simulation(
    blocks,
    connections,
    Solver=RKV65,
    tolerance_lte_rel=1e-4,
    tolerance_lte_abs=1e-6
    )

# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(100)
    sco.plot()
    sco.plot3D()

    plt.show()