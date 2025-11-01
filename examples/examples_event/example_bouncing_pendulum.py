#########################################################################################
##
##                       PathSim example of a bouncing pendulum
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

from pathsim.solvers import RKCK54, RKBS32, SSPRK33

from pathsim.events import ZeroCrossing


# MATHEMATICAL PENDULUM =================================================================

#initial angle and angular velocity
phi0, omega0 = 0.99*np.pi, 0.0

#parameters (gravity, length)
g, l = 9.81, 1

#bounceback for sensitivity
b = 0.9

#blocks that define the system
In1 = Integrator(omega0) 
In2 = Integrator(phi0) 
Amp = Amplifier(-g/l) 
Fnc = Function(np.sin) 
Sco = Scope(labels=[r"$\omega$", r"$\phi$"])

blocks = [In1, In2, Amp, Fnc, Sco]

#connections between the blocks
connections = [
    Connection(In1, In2, Sco[0]), 
    Connection(In2, Fnc, Sco[1]),
    Connection(Fnc, Amp), 
    Connection(Amp, In1)
    ]

#event function for zero crossing detection
def func_evt(t):
    *_, ph = In2()
    return ph 

#action function for state transformation
def func_act(t):
    *_, om = In1()
    *_, ph = In2()
    In1.engine.set(-om*b) #bounceback
    In2.engine.set(abs(ph)) 

#events (zero crossing)
E1 = ZeroCrossing(
    func_evt=func_evt,                 
    func_act=func_act, 
    tolerance=1e-6
    )

events = [E1]

#simulation instance from the blocks and connections
Sim = Simulation(
    blocks, 
    connections, 
    events,
    dt=0.1, 
    log=True, 
    Solver=RKCK54, 
    tolerance_lte_abs=1e-8, 
    tolerance_lte_rel=1e-6
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(duration=15)

    #plot the results directly from the scope
    fig, ax = Sco.plot(lw=2)

    #add the events to scope plot
    for t in E1: ax.axvline(t, c="k", ls="--")


    plt.show()