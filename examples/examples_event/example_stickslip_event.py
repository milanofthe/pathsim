#########################################################################################
##
##            PathSim example of slip-stick system (box on conveyor belt) 
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
    Source,
    Switch,
    Adder, 
    Scope
    )

from pathsim.events import ZeroCrossing

from pathsim.solvers import RKBS32


# SYSTEM DEFINITION =====================================================================

#initial position and velocity
x0, v0 = 0, 0

#system parameters
m = 20.0    # mass
k = 70.0    # spring constant
d = 10.0    # spring damping
mu = 1.5    # friction coefficient
g = 9.81    # gravity
v = 2.0     # belt velocity magnitude
T = 50.0    # excitation period

F_c = mu * m * g # friction force 

#function for belt velocity
def v_belt(t):
    return v * np.sin(2*np.pi*t/T)
    # return v * t / T
    # return v * (1 - np.exp(-t/T))

#function for coulomb friction force
def f_coulomb(v, vb):
    return F_c * np.sign(vb - v)


# system topology -----------------------------------------------------------------------

#blocks that define the system dynamics
Sr = Source(v_belt)      # velocity of the belt
I1 = Integrator(v0)      # integrator for velocity
I2 = Integrator(x0)      # integrator for position
A1 = Amplifier(-d)
A2 = Amplifier(-k)
A3 = Amplifier(1/m)
Fc = Function(f_coulomb) # coulomb friction (kinetic)
P1 = Adder()
Sw = Switch(0)           # selecting port '0' initially

#blocks for visualization
Sc1 = Scope(
    labels=[
        "belt velocity", 
        "box velocity", 
        "box position"
        ]
    )
Sc2 = Scope(
    labels=[
        "box force",
        "coulomb force"
        ]
    )

blocks = [Sr, I1, I2, A1, A2, A3, Fc, P1, Sw, Sc1, Sc2]

#connections between the blocks
connections = [
    Connection(I1, Sw[1], Fc[0]), 
    Connection(Sr, Sw[0], Fc[1], Sc1[0]), 
    Connection(Sw, I2, A1, Sc1[1]), 
    Connection(I2, A2, Sc1[2]), 
    Connection(A1, P1[0]), 
    Connection(A2, P1[1]), 
    Connection(Fc, P1[2], Sc2[1]),
    Connection(P1, A3, Sc2[0]), 
    Connection(A3, I1)
    ]


# event management ----------------------------------------------------------------------

def slip_to_stick_evt(t):
    _1, v_box , _2 = Sw() 
    _1, v_belt, _2 = Sr()
    dv = v_box - v_belt 

    return dv

def slip_to_stick_act(t):

    #change switch state
    Sw.select(0)

    I1.off()
    Fc.off()

    E_slip_to_stick.off()
    E_stick_to_slip.on()
    
E_slip_to_stick = ZeroCrossing(
    func_evt=slip_to_stick_evt,                 
    func_act=slip_to_stick_act, 
    tolerance=1e-3
    )


def stick_to_slip_evt(t):
    _1, F, _2 = P1()
    return F_c - abs(F)

def stick_to_slip_act(t):

    #change switch state
    Sw.select(1)

    I1.on()
    Fc.on()

    #set integrator state
    _1, v_box , _2 = Sw() 
    I1.engine.set(v_box)

    E_slip_to_stick.on()
    E_stick_to_slip.off()

E_stick_to_slip = ZeroCrossing(
    func_evt=stick_to_slip_evt,                 
    func_act=stick_to_slip_act, 
    tolerance=1e-3
    )


events = [E_slip_to_stick, E_stick_to_slip]


# simulation setup ----------------------------------------------------------------------

#create a simulation instance from the blocks and connections
Sim = Simulation(
    blocks, 
    connections, 
    events,
    dt=0.01, 
    dt_max=0.1, 
    log=True, 
    Solver=RKBS32, 
    tolerance_lte_abs=1e-6, 
    tolerance_lte_rel=1e-4
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    #run the simulation for some time
    Sim.run(2*T)


    # visualization ---------------------------------------------------------------------

    #plot the results directly from the two scopes
    fig, ax = Sc1.plot("-", lw=2)

    for t in E_slip_to_stick:
        ax.axvline( t , ls="--", c="k")

    for t in E_stick_to_slip:
        ax.axvline( t , ls=":", c="k")

    Sc2.plot("-", lw=2)


    plt.show()