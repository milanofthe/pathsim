# PathSim: A Time-Domain System Simulation Framework


## Overview

PathSim is a minimalistic and flexible block-based time-domain system simulation framework in Python with basic automatic differentiation capabilities. It provides a modular and intuitive approach to modeling and simulating complex interconnected dynamical systems. It is similar to Matlab Simulink in spirit but works very differently under the hood.

Key features of PathSim include:

- Decentralized architecture where each dynamical block has their own numerical integration engine.
- The system is solved directly on the computational graph instead of compiling a unified differential algebraic system.
- This has some advantages such as hot-swappable blocks during simulation and reading simulation results directly from the scopes.
- The block execution is decoupled from the data transfer, which enables parallelization (future) and linear computational complexity scaling for sparsely connected systems.
- Support for MIMO (Multiple Input, Multiple Output) blocks, enabling the creation of complex interconnected system topologies.
- Fixed-point iteration approach with path length estimation to efficiently resolve algebraic loops.
- Wide range of numerical solvers, including implicit and explicit multi-stage, and adaptive Runge-Kutta methods such as `RKDP54` or `ESDIRK54`.
- Modular and hierarchical modeling with (nested) subsystems.
- Automatic differentiation for differentiable system simulations.
- Library of pre-defined blocks, including mathematical operations, integrators, delays, transfer functions, and more.
- Easy extensibility, allowing users to define custom blocks by subclassing the base `Block` class and implementing just a handful of methods.

## Installation

The latest release version of pathsim available on [PyPi](https://pypi.org/project/pathsim/)  and installable via pip:

```console
$ pip install pathsim
```

## Example - Harmonic Oscillator

Here's an example that demonstrates how to create a basic simulation. The main components of the package are:

- `Simulation`: The main class that handles the blocks, connections, and the simulation loop.
- `Connection`: The class that defines the connections between blocks.
- Various block classes from the `blocks` module, such as `Integrator`, `Amplifier`, `Adder`, `Scope`, etc.

In this example, we create a simulation of the harmonic oscillator (a spring mass damper 2nd order system) initial value problem. The ODE that defines it is give by

$$
\ddot{x} + \frac{c}{m} \dot{x} + \frac{k}{m} x = 0
$$

where $c$ is the damping, $k$ the spring constant and $m$ the mass. And initial conditions $x_0$ and $v_0$ for position and velocity.

The ODE above can be translated to a block diagram using integrators, amplifiers and adders in the following way:

![png](README_files/harmonic_oscillator_blockdiagram.png)

The topology of the block diagram above can be directly defined as blocks and connections in the `PathSim` framework. First we initialize the blocks needed to represent the dynamical systems with their respective arguments such as initial conditions and gain values, then the blocks are connected using `Connection` objects, forming two feedback loops. The `Simulation` instance manages the blocks and connections and advances the system in time with the timestep (`dt`). The `log` flag for logging the simulation progress is also set. Finally, we run the simulation for some number of seconds and plot the results using the `plot()` method of the scope block.



```python
from pathsim import Simulation
from pathsim import Connection
from pathsim.blocks import Integrator, Amplifier, Adder, Scope
from pathsim.solvers import SSPRK22  # 2nd order fixed timestep, this is also the default

#initial position and velocity
x0, v0 = 2, 5

#parameters (mass, damping, spring constant)
m, c, k = 0.8, 0.2, 1.5

# Create blocks 
I1 = Integrator(v0)   # integrator for velocity
I2 = Integrator(x0)   # integrator for position
A1 = Amplifier(-c/m)
A2 = Amplifier(-k/m)
P1 = Adder()
Sc = Scope(labels=["v(t)", "x(t)"])

blocks = [I1, I2, A1, A2, P1, Sc]

# Create connections
connections = [
    Connection(I1, I2, A1, Sc),   # one to many connection
    Connection(I2, A2, Sc[1]),
    Connection(A1, P1),           # default connection to port 0
    Connection(A2, P1[1]),        # specific connection to port 1
    Connection(P1, I1)
    ]

# Create a simulation instance from the blocks and connections
Sim = Simulation(blocks, connections, dt=0.05, log=True, Solver=SSPRK22)

# Run the simulation for 50 seconds
Sim.run(duration=50.0)

# Plot the results directly from the scope
Sc.plot()

# Read the results from the scope for further processing
time, data = Sc.read()
```

    2024-11-16 11:08:37,339 - INFO - LOGGING enabled
    2024-11-16 11:08:37,341 - INFO - SOLVER SSPRK22 adaptive=False implicit=False
    2024-11-16 11:08:37,343 - INFO - PATH LENGTH ESTIMATE 2, 'iterations_min' set to 2
    2024-11-16 11:08:37,346 - INFO - RESET
    2024-11-16 11:08:37,352 - INFO - RUN duration=50.0
    2024-11-16 11:08:37,354 - INFO - STARTING progress tracker
    2024-11-16 11:08:37,355 - INFO - progress=0%
    2024-11-16 11:08:37,378 - INFO - progress=10%
    2024-11-16 11:08:37,396 - INFO - progress=20%
    2024-11-16 11:08:37,414 - INFO - progress=30%
    2024-11-16 11:08:37,432 - INFO - progress=40%
    2024-11-16 11:08:37,452 - INFO - progress=50%
    2024-11-16 11:08:37,471 - INFO - progress=60%
    2024-11-16 11:08:37,489 - INFO - progress=70%
    2024-11-16 11:08:37,507 - INFO - progress=80%
    2024-11-16 11:08:37,524 - INFO - progress=90%
    2024-11-16 11:08:37,543 - INFO - progress=100%
    2024-11-16 11:08:37,543 - INFO - FINISHED steps(total)=1001(1001) runtime=187.95ms
    


    
![png](README_files/README_4_1.png)
    


## Differentiable Simulation

PathSim also includes a rudimentary automatic differentiation framework based on a dual number system with overloaded operators. This makes the system simulation fully differentiable with respect to a predefined set of parameters. For now it only works with the explicit integrators. To demonstrate this lets consider the following linear feedback system.

![png](README_files/linear_feedback_blockdiagram.png)


The source term is a scaled unit step function (scaled by $A$). The parameters we want to differentiate the time domain response by are the feedback term $a$, the initial condition $x_0$ and the amplitude of the source term $A$.


```python
from pathsim import Simulation, Connection
from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope

#AD module
from pathsim.diff import Value, der

#values for derivative propagation
A  = Value(1)
a  = Value(-1)
x0 = Value(2)

#simulation timestep
dt = 0.01

#step function
tau = 3
def s(t):
    return A*int(t>tau)

#blocks that define the system
Src = Source(s)
Int = Integrator(x0)
Amp = Amplifier(a)
Add = Adder()
Sco = Scope(labels=["step", "response"])

blocks = [Src, Int, Amp, Add, Sco]

#the connections between the blocks
connections = [
    Connection(Src, Add[0], Sco[0]),
    Connection(Amp, Add[1]),
    Connection(Add, Int),
    Connection(Int, Amp, Sco[1])
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, dt=dt, log=True)
    
#run the simulation for some time
Sim.run(4*tau)

Sco.plot()
```

    2024-11-16 11:08:37,697 - INFO - LOGGING enabled
    2024-11-16 11:08:37,698 - INFO - SOLVER SSPRK22 adaptive=False implicit=False
    2024-11-16 11:08:37,698 - INFO - PATH LENGTH ESTIMATE 2, 'iterations_min' set to 2
    2024-11-16 11:08:37,699 - INFO - RESET
    2024-11-16 11:08:37,700 - INFO - RUN duration=12
    2024-11-16 11:08:37,700 - INFO - STARTING progress tracker
    2024-11-16 11:08:37,701 - INFO - progress=0%
    2024-11-16 11:08:37,752 - INFO - progress=10%
    2024-11-16 11:08:37,800 - INFO - progress=20%
    2024-11-16 11:08:37,852 - INFO - progress=30%
    2024-11-16 11:08:37,904 - INFO - progress=40%
    2024-11-16 11:08:37,956 - INFO - progress=50%
    2024-11-16 11:08:38,009 - INFO - progress=60%
    2024-11-16 11:08:38,062 - INFO - progress=70%
    2024-11-16 11:08:38,115 - INFO - progress=80%
    2024-11-16 11:08:38,168 - INFO - progress=90%
    2024-11-16 11:08:38,220 - INFO - progress=100%
    2024-11-16 11:08:38,221 - INFO - FINISHED steps(total)=1201(1201) runtime=520.23ms
    


    
![png](README_files/README_6_1.png)
    


Now the recorded data is of type `Value` and we can evaluate the automatically computed partial derivatives at each timestep. For example 
$\partial x(t) / \partial a$ the response with respect to the linear feedback parameter.


```python
import matplotlib.pyplot as plt

#read data from the scope
time, [step, data] = Sco.read()

fig, ax = plt.subplots(nrows=1, tight_layout=True, figsize=(8, 4), dpi=120)

#evaluate and plot partial derivatives
ax.plot(time, der(data, a), label="$dx/da$")
ax.plot(time, der(data, x0), label="$dx/dx_0$")
ax.plot(time, der(data, A), label="$dx/dA$")

ax.set_xlabel("time [s]")
ax.grid(True)
ax.legend(fancybox=False);
```


    
![png](README_files/README_8_0.png)
    


## Event Detection

Recently I added an event handling system to PathSim that implements zero-crossing detection. This enables the simulation of hybrid continuous time systems with discrete events. Probably the most popular example for this is the bouncing ball where discrete events occur whenever the ball touches the floor.


```python
from pathsim import Simulation, Connection, Event
from pathsim.blocks import Integrator, Constant, Scope
from pathsim.solvers import RKBS32 #using an adaptive timestep solver

#initial values
x0, v0 = 1, 10

#blocks that define the system
Ix = Integrator(x0)     # v -> x
Iv = Integrator(v0)     # a -> v 
Cn = Constant(-9.81)    # gravitational acceleration
Sc = Scope(labels=["x", "v"])

blocks = [Ix, Iv, Cn, Sc]

#the connections between the blocks
connections = [
    Connection(Cn, Iv),
    Connection(Iv, Ix),
    Connection(Ix, Sc)
    ]

#events (zero-crossings) -> ball makes contact
E1 = Event(
    blocks=[Ix, Iv],                  # blocks to watch states of
    g=lambda x, y: x,                 # event function for zero crossing detection
    f=lambda x, y: [abs(x), -0.9*y],  # action function for state transformation
    tolerance=1e-4
    )

events = [E1]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, events, dt=0.1, log=True, Solver=RKBS32, dt_max=0.1)

#run the simulation
Sim.run(15)

#plot the recordings from the scope
Sc.plot();
```

    2024-11-16 11:08:38,716 - INFO - LOGGING enabled
    2024-11-16 11:08:38,717 - INFO - SOLVER RKBS32 adaptive=True implicit=False
    2024-11-16 11:08:38,717 - INFO - PATH LENGTH ESTIMATE 1, 'iterations_min' set to 1
    2024-11-16 11:08:38,717 - INFO - RESET
    2024-11-16 11:08:38,718 - INFO - RUN duration=15
    2024-11-16 11:08:38,718 - INFO - STARTING progress tracker
    2024-11-16 11:08:38,719 - INFO - progress=0%
    2024-11-16 11:08:38,723 - INFO - progress=10%
    2024-11-16 11:08:38,729 - INFO - progress=20%
    2024-11-16 11:08:38,735 - INFO - progress=30%
    2024-11-16 11:08:38,741 - INFO - progress=40%
    2024-11-16 11:08:38,744 - INFO - progress=50%
    2024-11-16 11:08:38,751 - INFO - progress=60%
    2024-11-16 11:08:38,759 - INFO - progress=70%
    2024-11-16 11:08:38,766 - INFO - progress=80%
    2024-11-16 11:08:38,772 - INFO - progress=90%
    2024-11-16 11:08:38,781 - INFO - progress=100%
    2024-11-16 11:08:38,781 - INFO - FINISHED steps(total)=243(286) runtime=63.02ms
    


    
![png](README_files/README_10_1.png)
    


During the event handling, the simulator approaches the event until the event tolerance is met. You can see this by analyzing the timesteps taken by `RKBS32`.


```python
import numpy as np

fig, ax = plt.subplots(figsize=(8,4), tight_layout=True, dpi=120)

time, _ = Sc.read()

#add detected events
for t in E1: ax.axvline(t, ls="--", c="k")

#plot the timesteps
ax.plot(time[:-1], np.diff(time))

ax.set_yscale("log")
ax.set_ylabel("dt [s]")
ax.set_xlabel("time [s]")
ax.grid(True)
```


    
![png](README_files/README_12_0.png)
    



```python

```

## Contributing and Future

There are some things I want to explore with PathSim eventually, and your help is highly appreciated! If you want to contribute, send me a message and we can discuss how!

Some of the possible directions for future features are:
- better `__repr__` for the blocks maybe in json format OR just add a `json` method to the blocks and to the connections that builds a netlist representation to save to and load from an interpretable file (compatibility with other system description languages)
- include discrete time blocks and integrate them into the event handling mechanism
- more extensive testing and validation (as always)



```python

```
