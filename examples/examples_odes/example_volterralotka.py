#########################################################################################
##
##                     PathSim Example for Volterra-Lotka System 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, Integrator, Adder, Amplifier, Multiplier
from pathsim.solvers import RKBS32


# VOLTERRA-LOTKA SYSTEM =================================================================

#parameters 
alpha = 1.0  # growth rate of prey
beta = 0.1   # predator sucess rate
delta = 0.5  # predator efficiency
gamma = 1.2  # death rate of predators


#blocks that define the system
i_pred = Integrator(10)
i_prey = Integrator(5)

a_alp = Amplifier(alpha)
a_gma = Amplifier(gamma)
a_bet = Amplifier(beta)
a_del = Amplifier(delta)

p_pred = Adder("-+")
p_prey = Adder("+-")

m_pp = Multiplier()

sco = Scope(labels=["predator population", "prey population"])

blocks = [
    i_pred, i_prey, a_alp, a_gma, 
    a_bet, a_del, p_pred, p_prey, 
    m_pp, sco
    ]

#the connections between the blocks
connections = [
    Connection(i_pred, m_pp[0], a_alp, sco[0]),
    Connection(i_prey, m_pp[1], a_gma, sco[1]),
    Connection(a_del, p_prey[0]),
    Connection(a_gma, p_prey[1]),
    Connection(a_bet, p_pred[0]),
    Connection(a_alp, p_pred[1]),
    Connection(m_pp, a_del, a_bet),
    Connection(p_pred, i_pred),
    Connection(p_prey, i_prey)
    ]


#initialize the simulation with everything
Sim = Simulation(
    blocks, 
    connections, 
    Solver=RKBS32
    )


# Run Example ===========================================================================    

if __name__ == "__main__":

    #run the simulation
    Sim.run(20)

    sco.plot()
    sco.plot2D()

    plt.show()
