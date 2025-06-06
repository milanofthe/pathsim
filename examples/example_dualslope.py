#########################################################################################
##
##                             Dual-Slope (integrating) ADC
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection

from pathsim.blocks import (
    Integrator, Adder, Amplifier, Scope, Source, 
    Constant, Switch, SampleHold, Comparator
    )

from pathsim.events import Schedule, ZeroCrossingDown

from pathsim.solvers import RKBS32


# SYSTEM SETUP AND SIMULATION ===========================================================

v_ref = -0.5           #dac reference
f_clk = 20             #sampling frequency
T_clk = 1.0 / f_clk    #sampling period

K = 0.1

#blocks that define the system
src = Source(lambda t: 0.3*np.sin(2*np.pi*t)+0.5) 
sah = SampleHold(T=T_clk) 
ref = Constant(v_ref) 
swt = Switch(None) 

sub = Adder("+-") 
itg = Integrator() 
fbk = Amplifier(K) 

sco = Scope(labels=["src", "sah", "swt", "itg"]) 

blocks = [src, sub, ref, swt, fbk, itg, sah, sco]

#connections between the blocks
connections = [
    Connection(src, sah, sco[0]),  
    Connection(sah, swt[0], sco[1]),     
    Connection(ref, swt[1]), 
    Connection(swt, sub[0], sco[2]), 
    Connection(fbk, sub[1]),    
    Connection(sub, itg),    
    Connection(itg, fbk, sco[3])
]


#timing logic through events
events = [
    ZeroCrossingDown(
        func_evt=lambda *_: itg.engine.get(),
        func_act=lambda *_: [itg.reset(), swt.select(None)]
        ),
    Schedule(
        t_start=0, 
        t_period=T_clk, 
        func_act=lambda *_: swt.select(0),
        ),
    Schedule(
        t_start=T_clk*0.2, 
        t_period=T_clk, 
        func_act=lambda *_: swt.select(1),
        )
]


#simulation with adaptive solver
Sim = Simulation(
    blocks,
    connections,
    events,
    dt_max=T_clk*0.01,
    Solver=RKBS32
)


# Run Example ===========================================================================

if __name__ == "__main__":

    Sim.run(1)
    Sim.plot()

    plt.show()