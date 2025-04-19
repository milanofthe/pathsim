#########################################################################################
##
##                         PathSim Example for Lorenz System 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, Integrator, Constant, Adder, Amplifier, Multiplier
from pathsim.solvers import RKCK54, RKBS32, GEAR52A, RKF21, ESDIRK43


# LORENZ SYSTEM =========================================================================

# parameters 
sigma, rho, beta = 10, 28, 8/3

# Initial conditions
x0, y0, z0 = 1.0, 1.0, 1.0

# Integrators store the state variables x, y, z
itg_x = Integrator(x0) # dx/dt = sigma * (y - x)
itg_y = Integrator(y0) # dy/dt = x * (rho - z) - y
itg_z = Integrator(z0) # dz/dt = x * y - beta * z

# Components for dx/dt
amp_sigma = Amplifier(sigma)
add_x = Adder("+-") # Computes y - x

# Components for dy/dt
cns_rho = Constant(rho)
add_rho_z = Adder("+-") # Computes rho - z
mul_x_rho_z = Multiplier() # Computes x * (rho - z)
add_y = Adder("-+") # Computes -y + [x * (rho - z)]

# Components for dz/dt
mul_xy = Multiplier() # Computes x * y
amp_beta = Amplifier(beta) # Computes beta * z
add_z = Adder("+-") # Computes (x * y) - (beta * z)

# Scope for plotting
sco = Scope(labels=["x", "y", "z"])

# List of all blocks
blocks = [
    itg_x, itg_y, itg_z,
    amp_sigma, add_x,
    cns_rho, add_rho_z, mul_x_rho_z, add_y,
    mul_xy, amp_beta, add_z,
    sco
    ]

# Connections
connections = [
    # Output signals (from integrators)
    Connection(itg_x, add_x[1], mul_x_rho_z[0], mul_xy[0], sco[0]), # x -> (y-x), x*(rho-z), x*y, scope
    Connection(itg_y, add_x[0], add_y[0], mul_xy[1], sco[1]),       # y -> (y-x), -y + [...], x*y, scope
    Connection(itg_z, add_rho_z[1], amp_beta, sco[2]),              # z -> (rho-z), beta*z, scope

    # dx/dt path: sigma * (y - x) -> itg_x
    Connection(add_x, amp_sigma),  # (y - x) -> sigma * (y - x)
    Connection(amp_sigma, itg_x),  # sigma * (y - x) -> integrator x

    # dy/dt path: x * (rho - z) - y -> itg_y
    Connection(cns_rho, add_rho_z[0]),     # rho -> (rho - z) input 0
    Connection(add_rho_z, mul_x_rho_z[1]), # (rho - z) -> x * (rho - z) input 1
    Connection(mul_x_rho_z, add_y[1]),     # x * (rho - z) -> -y + [x * (rho - z)] input 1
    Connection(add_y, itg_y),              # x * (rho - z) - y -> integrator y

    # dz/dt path: x * y - beta * z -> itg_z
    Connection(mul_xy, add_z[0]),    # x * y -> (x * y) - (beta * z) input 0
    Connection(amp_beta, add_z[1]),  # beta * z -> (x * y) - (beta * z) input 1
    Connection(add_z, itg_z)         # (x * y) - (beta * z) -> integrator z
    ]


Sim = Simulation(
    blocks,
    connections,
    Solver=RKBS32,
    tolerance_lte_rel=1e-4,
    tolerance_lte_abs=1e-6,
    tolerance_fpi=1e-6
    )


# Run Example ===========================================================================    

if __name__ == "__main__":

    #run the simulation
    Sim.run(50)
    sco.plot()
    sco.plot3D(lw=1)

    plt.show()
