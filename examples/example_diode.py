#########################################################################################
##
##                        PathSim example of a diode circuit
##                           (Nonlinear algebraic loop)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection

from pathsim.blocks import (
    Source,
    Amplifier, 
    Function,
    Adder, 
    Scope
    )

from pathsim.solvers import RKBS32


# DIODE CIRCUIT =========================================================================

# Circuit parameters
R = 1000.0          # Resistor (Ohms)
I_s = 1e-12         # Diode saturation current (A)
V_T = 0.026         # Thermal voltage at room temperature (V)

# Define diode current function: i = I_s * (exp(v_diode/(n*V_T)) - 1)
def diode_current(v_diode):
    """Diode current as function of diode voltage"""
    clipped = np.clip(v_diode/V_T, None, 25)
    return I_s * (np.exp(clipped) - 1)

# Define voltage source function
def voltage_source(t):
    """Sinusoidal voltage source"""
    return 5.0 * np.sin(2 * np.pi * t)

# Blocks that define the system
Src = Source(voltage_source)                    # Voltage source
DiodeFn = Function(diode_current)               # Diode i-v characteristic  
ResAmp = Amplifier(-R)                          # -R (negative resistance)
Add = Adder()                                   # Adder for KVL
Sc1 = Scope(labels=["v_source", "v_diode"])
Sc2 = Scope(labels=["i_diode"])

blocks = [Src, DiodeFn, ResAmp, Add, Sc1, Sc2]

connections = [
    Connection(Src, Add[0], Sc1[0]),            # Source to adder and scope
    Connection(Add, DiodeFn, Sc1[1]),           # Diode voltage to function and scope
    Connection(DiodeFn, ResAmp, Sc2),           # Diode current to resistor and scope
    Connection(ResAmp, Add[1]),                  # Voltage drop back to adder (algebraic loop)
]

# Simulation instance
Sim = Simulation(
    blocks, 
    connections, 
    dt=0.001, 
    Solver=RKBS32,
    tolerance_fpi=1e-12
    )


# Run Example ===========================================================================

if __name__ == "__main__":
    
    # Run the simulation for 2 seconds
    Sim.run(duration=2.0)

    # Sim.graph.save_graphviz("example")
        
    # Plot the results
    Sim.plot()
        
    plt.show()