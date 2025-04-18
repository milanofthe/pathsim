#########################################################################################
##
##                  PathSim toy example for solar system dynamics 
##
##          this example shows that individual blocks can be quite complex,
##           here a multi-body system runs inside of a single 'ODE' block
##
#########################################################################################


# IMPORTS ===============================================================================

import numpy as np
import matplotlib.pyplot as plt

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Scope
from pathsim.solvers import RKCK54, RKF78


# SIMULATION PARAMETERS AND SETUP =======================================================

# Gravitational constant (in AU^3 / (solar mass * day^2))
G = 4 * np.pi**2 / 365**2


# Solar system body
class Body:

    def __init__(self, name, mass, pos, vel):
        self.name = name
        self.mass = mass
        self.pos = pos
        self.vel = vel

        self.scope = Scope(labels=[f"{name}_x", f"{name}_y", f"{name}_z"])

    def acceleration(self, others):
        acc = 0.0
        for other in others:
            if self == other: continue
            r = other.pos - self.pos
            acc += G * other.mass * r / np.linalg.norm(r)**3
        return acc

    def set_state(self, x):
        self.pos, self.vel = np.split(x, 2)


# list of bodies for solar system with initial conditions for 01.01.2024
bodies = [
    Body("Sun", 1.0, 
        np.array([-7.953295388406003E-03, -2.927365330787842E-03, 2.101676755805344E-04]), 
        np.array([4.895790326265684E-06, -7.031130993046038E-06, -4.608199080172032E-08])),
    Body("Mercury", 0.16601e-6, 
        np.array([-3.402449827894693E-01, 1.270778296493449E-01, 4.131347894005659E-02]), 
        np.array([-1.605040287977755E-02, -2.501076003142073E-02, -5.707201750753626E-04])),
    Body("Venus", 2.4478383e-6, 
        np.array([-7.145681732605230E-01, -1.397795793060785E-01, 3.910297216503490E-02]), 
        np.array([3.722046271827600E-03, -1.995491285165146E-02, -4.884816665833027E-04])),
    Body("Earth", 3.00348959632e-6, 
        np.array([-2.252729645653915E-01, 9.560667628227153E-01, 1.577009263320669E-04]), 
        np.array([-1.705539602709480E-02, -3.867848205935779E-03, 8.659435473035585E-07])),
    Body("Moon", 1.23000383e-8, 
        np.array([-2.279054882662570E-01, 9.555472591607231E-01, 1.966883909702569E-04]), 
        np.array([-1.692842706672744E-02, -4.413215938363101E-03, -4.814510536207644E-05])),
    Body("Mars", 0.3227151e-6, 
        np.array([-2.584550024643773E-01, -1.458170543308088E+00, -2.414302778676946E-02]), 
        np.array([1.432301013565291E-02, -1.180131804284778E-03, -3.758404643770721E-04])),
    Body("Jupiter", 954.79194e-6, 
        np.array([3.468750430010888E+00, 3.569029893989959E+00, -9.241237573403982E-02]), 
        np.array([-5.494699932030809E-03, 5.617278671368608E-03, 9.965559179501148E-05])),
    Body("Saturn", 285.8860e-6, 
        np.array([8.993559415482203E+00, -3.703623931688067E+00, -2.936792504543212E-01]), 
        np.array([1.813595119365610E-03, 5.147794035913646E-03, -1.618510392848537E-04])),
    Body("Uranus", 43.66244e-6, 
        np.array([1.225372655349175E+01, 1.530421738450486E+01, -1.019092411813474E-01]), 
        np.array([-3.099119844130931E-03, 2.274987396511312E-03, 4.859903652849407E-05])),
    Body("Neptune", 51.51389e-6, 
        np.array([2.983551610844023E+01, -1.784353951224969E+00, -6.508462844445427E-01]), 
        np.array([1.665857900509985E-04, 3.152219637988330E-03, -6.863577717527224E-05]))
]


# right hand side of the solar system ODE
def solar_system_ode(x, u, t):
    for s, b in zip(np.split(x, len(bodies)), bodies): b.set_state(s)
    return np.hstack([np.hstack([b.vel, b.acceleration(bodies)]) for b in bodies])

# Initial conditions
x0 = np.hstack([np.hstack([b.pos, b.vel]) for b in bodies])

# Create ODE block for the entire solar system
solar_system = ODE(func=solar_system_ode, initial_value=x0)

# Block list
blocks = [solar_system, *[b.scope for b in bodies]]

# Connections
connections = []
for i, b in enumerate(bodies):
    connections.extend([
        Connection(solar_system[6*i], b.scope[0]),
        Connection(solar_system[6*i+1], b.scope[1]),
        Connection(solar_system[6*i+2], b.scope[2])
    ])

# Create simulation
Sim = Simulation(
    blocks, 
    connections, 
    dt=0.1, 
    Solver=RKF78, 
    tolerance_lte_rel=1e-6, 
    tolerance_lte_abs=1e-8
    )


# Run Example ===========================================================================

if __name__ == "__main__":

    # Run simulation for some number of days
    Sim.run(365*5)


    # PLOT THE RESULTS ==================================================================

    # Plot solar system
    fig = plt.figure(figsize=(12, 10), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    for b in bodies:
        t, data = b.scope.read()
        line, = ax.plot(*data, alpha=0.5)
        s = (10 + np.log10(b.mass**1/3))*1.2
        ax.plot(*b.pos, "o", markersize=s, color=line.get_color(), label=b.name)

    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_zlabel("Z (AU)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"Solar System Orbits - {Sim.time} days")


    # Plot Earth-Moon system
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120, tight_layout=True)

    earth = next(b for b in bodies if b.name=="Earth")
    moon = next(b for b in bodies if b.name=="Moon")

    _, data_earth = earth.scope.read()
    _, data_moon = moon.scope.read()

    #earth
    s = (10 + np.log10(earth.mass**1/3))*3
    ax.plot(0.0, 0.0, "o", markersize=s, label="Earth")

    #moon and orbit
    line, = ax.plot(*(data_moon-data_earth)[:2], alpha=0.5)
    s = (10 + np.log10(moon.mass**1/3))*4
    ax.plot(*(moon.pos-earth.pos)[:2], "o", markersize=s, color=line.get_color(), label="Moon")

    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"Earth-Moon Orbit - {Sim.time} days")


    plt.show()