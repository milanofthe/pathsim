# PathSim: A Time-Domain System Simulation Framework


## Overview

PathSim is a minimalistic and flexible block-based time-domain system simulation framework in Python. It provides a modular and intuitive approach to modeling and simulating complex dynamical systems using a directed computational graph. It is similar to Matlab Simulink in spirit but works very differently under the hood.

Key features of PathSim include:

- Decentralized architecture where each dynamical block has their own numerical integration engine. 
- The system is solved directly on the computational graph instead of compiling a unified differetial algebraic system.
- This has some advantages such as hot-swappable blocks during simulation and reading simulation results directly from the scopes.
- The block execution is decoupled from the data transfer, which enables parallelization (future) and linear computational complexity scaling for sparsely connected systems.
- Support for MIMO (Multiple Input, Multiple Output) blocks, enabling the creation of complex interconnected system topologies.
- Fixed-point iteration approach with path length estimation to efficiently resolve algebraic loops.
- Wide range of numerical solvers, including implicit and explicit multi-stage, and adaptive Runge-Kutta methods such as `RKDP54` or `ESDIRK54`.
- Modular and hierarchical modeling with (nested) subsystems.
- Library of pre-defined blocks, including mathematical operations, integrators, delays, transfer functions, and more.
- Easy extensibility, allowing users to define custom blocks by subclassing the base `Block` class and implementing just a handfull of methods. 


## Installation

Version `0.1.0` of pathsim is pip installable

```console
$ pip install pathsim
```

## Getting Started

To get started with PathSim, you need to import the necessary modules and classes. The main components of the package are:

- `Simulation`: The main class that handles the blocks, connections, and the simulation loop.
- `Connection`: The class that defines the connections between blocks.
- Various block classes from the `blocks` module, such as `Integrator`, `Amplifier`, `Adder`, `Scope`, etc.

Here's an example that demonstrates how to create a basic simulation. 
In this example, we create a simulation of the harmonic oscillator (a spring mass damper 2nd order system) initial value problem. The ODE that defines it is give by

$$
\ddot{x} + \frac{c}{m} \dot{x} + \frac{k}{m} x = 0
$$

where $c$ is the damping, $k$ the spring constant and $m$ the mass.

It can be translated to a block diagram using integrators, amplifiers and adders in the following way:

![png](README_files/harmonic_oscillator_blockdiagram.png)

The topology of the block diagram above can be directly defined as blocks and connections in the pathsim framework. First we initialize the blocks needed to represent the dynamical systems with their respective arguments such as initial conditions and gain values, then the blocks are connected using `Connection` objects, forming two feedback loops. The `Simulation` instance manages the blocks and connections and advances the system in time with the timestep (`dt`). The `log` flag for logging the simulation progress is also set. Finally, we run the simulation for some number of seconds and plot the results using the `plot()` method of the scope block.



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
A1 = Amplifier(c)
A2 = Amplifier(k)
A3 = Amplifier(-1/m)
P1 = Adder()
Sc = Scope(labels=["velocity", "position"])

blocks = [I1, I2, A1, A2, A3, P1, Sc]

# Create connections
connections = [
    Connection(I1, I2, A1, Sc),   # one to many connection
    Connection(I2, A2, Sc[1]),
    Connection(A1, P1),           # default connection to port 0
    Connection(A2, P1[1]),        # specific connection to port 1
    Connection(P1, A3),
    Connection(A3, I1)
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

    2024-08-12 10:38:03,838 - INFO - LOGGING enabled
    2024-08-12 10:38:03,839 - INFO - SOLVER SSPRK22 adaptive=False implicit=False
    2024-08-12 10:38:03,840 - INFO - PATH LENGTH ESTIMATE 3, 'iterations_min' set to 3
    2024-08-12 10:38:03,840 - INFO - RESET
    2024-08-12 10:38:03,841 - INFO - RUN duration=50.0
    2024-08-12 10:38:03,842 - INFO - STARTING progress tracker
    2024-08-12 10:38:03,843 - INFO - progress=0%
    2024-08-12 10:38:03,881 - INFO - progress=10%
    2024-08-12 10:38:03,921 - INFO - progress=20%
    2024-08-12 10:38:03,957 - INFO - progress=30%
    2024-08-12 10:38:03,990 - INFO - progress=40%
    2024-08-12 10:38:04,028 - INFO - progress=50%
    2024-08-12 10:38:04,065 - INFO - progress=60%
    2024-08-12 10:38:04,103 - INFO - progress=70%
    2024-08-12 10:38:04,138 - INFO - progress=80%
    2024-08-12 10:38:04,175 - INFO - progress=90%
    2024-08-12 10:38:04,208 - INFO - progress=100%
    2024-08-12 10:38:04,209 - INFO - FINISHED steps(total)=1001(1001) runtime=366.36ms
    


    
![png](README_files/README_4_1.png)
    


## Examples
There are many examples of dynamical system simulations in the `examples` directory. They cover almost all the blocks currently available in PathSim as well as different solvers.
