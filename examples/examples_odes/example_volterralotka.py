#########################################################################################
##
##                     PathSim Example for Volterra-Lotka System 
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, Integrator, Constant, Adder, Amplifier, Multiplier
from pathsim.solvers import RKBS32


# VOLTERRA-LOTKA SYSTEM =================================================================

#parameters 
alpha = 1.0  # growth rate of prey
beta = 0.1   # predator sucess rate
delta = 0.5  # predator efficiency
gamma = 1.2  # death rate of predators


i_pred = Integrator(10)
i_prey = Integrator(5)

c_alp = Constant(alpha)
c_gma = Constant(gamma)

a_bet = Amplifier(beta)
a_del = Amplifier(delta)

p_pred = Adder("-+")
p_prey = Adder("+-")

m_pred = Multiplier()
m_prey = Multiplier()

sco = Scope(labels=["predator population", "prey population"])

blocks = [
    i_pred, i_prey, c_alp, c_gma, 
    a_bet, a_del, p_pred, p_prey, 
    m_pred, m_prey, sco
    ]

connections = [
    Connection(i_pred, m_pred[0], a_del, sco[0]),
    Connection(i_prey, m_prey[0], a_bet, sco[1]),
    Connection(a_del, p_prey[0]),
    Connection(c_gma, p_prey[1]),
    Connection(a_bet, p_pred[0]),
    Connection(c_alp, p_pred[1]),
    Connection(p_pred, m_pred[1]),
    Connection(p_prey, m_prey[1]),
    Connection(m_pred, i_pred),
    Connection(m_prey, i_prey)
    ]


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
