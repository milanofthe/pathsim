#########################################################################################
##
##                      PathSim Example of Differentiable Simulation
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope

#optimization module
from pathsim.optim import Value

from pathsim.solvers import RKBS32


# 1st ORDER FEEDBACK SYSTEM =============================================================

#parameters
x0, a, s = Value.array([2, -1, 1])

#step delay
tau = 3

#blocks that define the system
Src = Source(lambda t: s*int(t>tau))
Int = Integrator(x0)
Amp = Amplifier(a)
Add = Adder()
Sco = Scope(labels=["s(t)", "x(t)"])

blocks = [Src, Int, Amp, Add, Sco]

#the connections between the blocks
connections = [
    Connection(Src, Add[0], Sco[0]),
    Connection(Amp, Add[1]),
    Connection(Add, Int),
    Connection(Int, Amp, Sco[1])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    dt=0.01, 
    Solver=RKBS32
    )
    

# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(4*tau)
    Sco.plot(lw=2)


    #plot derivatives -------------------------------------------------------------------

    #get the simulation results
    time, [stp, res] = Sco.read()

    fig, ax = plt.subplots(nrows=1, figsize=(8, 4), tight_layout=True, dpi=120)

    ax.plot(time, Value.der(res, a), lw=2, c="tab:red", label=r"$\partial x(t)/ \partial a$")
    ax.plot(time, Value.der(res, s), lw=2, c="tab:green", label=r"$\partial x(t)/\partial s$")
    ax.plot(time, Value.der(res, x0), lw=2, c="tab:blue", label=r"$\partial x(t)/\partial x_0$")

    ax.set_xlabel("time [s]")

    ax.grid(True)
    ax.legend()


    #extract the sensitivities
    dx_da = Value.der(res, a)
    dx_ds = Value.der(res, s)
    dx_dx0 = Value.der(res, x0)

    #parameter uncertainties
    sigma_a = 0.1    # Standard deviation in feedback gain
    sigma_s = 0.05   # Standard deviation in input amplitude
    sigma_x0 = 0.5   # Standard deviation in initial condition

    #output variance contribution at each time point
    var_x = (dx_da**2 * sigma_a**2 + 
             dx_ds**2 * sigma_s**2 + 
             dx_dx0**2 * sigma_x0**2)

    #standard deviation bounds
    upper_bound = Value.numeric(res) + np.sqrt(var_x)
    lower_bound = Value.numeric(res) - np.sqrt(var_x)


    #plot with uncertainty bounds
    
    fig, ax = plt.subplots(nrows=1, figsize=(8, 4), tight_layout=True, dpi=120)

    ax.plot(time, stp, lw=2, c="tab:red", label="s(t)")
    ax.plot(time, res, lw=2, c="tab:blue", label="x(t)")
    ax.fill_between(time, lower_bound, upper_bound, color="tab:blue", alpha=0.25, label='±1σ', ec=None)
    ax.set_xlabel('time [s]')
    ax.legend()
    ax.grid(True)

    plt.show()