#########################################################################################
##
##                             PathSim Chemical Reactor Example
##
#########################################################################################

# IMPORTS ===============================================================================
import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Source, Scope
from pathsim.solvers import ESDIRK32, ESDIRK43, GEAR52A


# CSTR WITH CONSECUTIVE REACTIONS INITIAL VALUE PROBLEM =================================

# Initial conditions
Ca_0 = 1.0    # Initial concentration of A
Cb_0 = 0.0    # Initial concentration of B
T_0 = 300.0   # Initial temperature


# System parameters
Tc = 280.0    # Coolant temperature
tau = 1.0     # Residence time
k1_0 = 1e4    # Rate constant 1
k2_0 = 1e3    # Rate constant 2
E1 = 5e4      # Activation energy 1
E2 = 5.5e4    # Activation energy 2
dH1 = -5e4    # Reaction enthalpy 1
dH2 = -5.2e4  # Reaction enthalpy 2
rho = 1000.0  # Density
Cp = 4.184    # Heat capacity
U = 1000.0    # Heat transfer coefficient
V = 0.1       # Reactor volume
A = 0.1       # Heat transfer area
R = 8.314     # Gas constant


# Define system blocks
Sco    = Scope(labels=['Ca', 'Cb', 'T'])
Src_Ca = Source(lambda t: 2.0 + np.sin(0.5*t))
Src_T  = Source(lambda t: 280.0 * (1 - 0.8 * np.exp(-0.6*t)))

def reaction_rates(x, u, t):

    #unpack states
    Ca, Cb, T = x

    #unpack inputs
    Ca_in, T_in = u
    
    # Concentration dynamics
    dCa_dt = (Ca_in - Ca)/tau - k1_0*np.exp(-E1/(R*T))*Ca
    dCb_dt = -Cb/tau + k1_0*np.exp(-E1/(R*T))*Ca - k2_0*np.exp(-E2/(R*T))*Cb

    # Temperature dynamics
    dT_dt = (T_in - T)/tau + \
            (-dH1/(rho*Cp))*k1_0*np.exp(-E1/(R*T))*Ca + \
            (-dH2/(rho*Cp))*k2_0*np.exp(-E2/(R*T))*Cb - \
            U*A*(T-Tc)/(V*rho*Cp)

    return np.array([dCa_dt, dCb_dt, dT_dt])

CSTR = ODE(reaction_rates, np.array([Ca_0, Cb_0, T_0]))

# Main system blocks and connections
blocks = [CSTR, Src_Ca, Src_T, Sco]
connections = [
    Connection(CSTR, Sco),         # Ca output
    Connection(CSTR[1], Sco[1]),   # Cb output
    Connection(CSTR[2], Sco[2]),   # T output
    Connection(Src_Ca, CSTR[0]),
    Connection(Src_T, CSTR[1])  
]

# Initialize simulation
Sim = Simulation(
    blocks,
    connections,
    dt=0.001,
    log=True,
    Solver=GEAR52A,
    tolerance_lte_abs=1e-6,
    tolerance_lte_rel=1e-4
)


# Run Example ===========================================================================

if __name__ == "__main__":

    # Run simulation for 10 seconds
    Sim.run(20)  

    # Plot results
    Sco.plot(".-", lw=1.5)

    plt.show()