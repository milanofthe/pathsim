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
    Constant,
    Function,
    Source,
    Switch,
    Adder, 
    Scope
    )

from pathsim.events import ZeroCrossing, ZeroCrossingUp

from pathsim.solvers import RKBS32


# SYSTEM DEFINITION =====================================================================

#initial position and velocity
x0, v0 = 0, 0

#system parameters
m = 20.0    # mass
k = 70.0    # spring constant
d = 10.0    # spring damping
mu_s = 1.5  # stick friction coefficient
mu_k = 1.4  # kinetic friction coefficient
g = 9.81    # gravity
v = 3.0     # belt velocity magnitude
T = 50.0    # excitation period

F_s = mu_s * m * g # sticking friction force 
F_k = mu_k * m * g #kinetic friction force

#function for belt velocity
def v_belt(t):
    return v * np.sin(2*np.pi*t/T)
    # return v * t/T

#function for coulomb friction (acceleration)
def f_coulomb(v, vb):
    return F_k / m * np.sign(vb - v)


# system topology -----------------------------------------------------------------------

#blocks that define the system
Sr = Source(v_belt)      # velocity of the belt
I1 = Integrator(v0)      # integrator for velocity
I2 = Integrator(x0)      # integrator for position
A1 = Amplifier(-d/m)
A2 = Amplifier(-k/m)
Fc = Function(f_coulomb) # coulomb friction (kinetic)
P1 = Adder()
Sw = Switch(1)           # selecting port '1' initially
Sc = Scope(
    labels=[
        "belt velocity", 
        "box velocity", 
        "box position", 
        "box acceleration",
        "coulomb acceleration"
        ]
    )

blocks = [Sr, I1, I2, A1, A2, Fc, P1, Sw, Sc]


#connections between the blocks
connections = [
    Connection(I1, Sw[0]), 
    Connection(Sr, Sw[1], Sc[0]), 
    Connection(Sw, I2, A1, Sc[1]), 
    Connection(I2, A2, Sc[2]), 
    Connection(Sw, Fc[0]), 
    Connection(Sr, Fc[1]),
    Connection(A1, P1[0]), 
    Connection(A2, P1[1]), 
    Connection(Fc, P1[2], Sc[4]), 
    Connection(P1, I1, Sc[3])
    ]



# event management ----------------------------------------------------------------------

def slip_to_stick_evt(t):
    _1, v_box , _2 = Sw() 
    _1, v_belt, _2 = Sr()
    dv = v_box - v_belt 

    return dv

def slip_to_stick_act(t):

    #change switch state
    Sw.state = 1

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
    _1, a, _2 = P1()
    return F_s - abs(m * a)

def stick_to_slip_act(t):

    #change switch state
    Sw.state = 0

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
    tolerance_lte_rel=1e-3
    )

#run the simulation for some time
Sim.run(2*T)


# visualization -------------------------------------------------------------------------

#plot the results directly from the scope
Sc.plot()

for t in E_slip_to_stick:
    Sc.ax.axvline(t, ls=":", c="k")

for t in E_stick_to_slip:
    Sc.ax.axvline(t, ls="--", c="k")


plt.show()