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

Version `0.2.0` of pathsim is pip installable

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

    2024-10-23 14:06:09,237 - INFO - LOGGING enabled
    2024-10-23 14:06:09,238 - INFO - SOLVER SSPRK22 adaptive=False implicit=False
    2024-10-23 14:06:09,238 - INFO - PATH LENGTH ESTIMATE 2, 'iterations_min' set to 2
    2024-10-23 14:06:09,239 - INFO - RESET
    2024-10-23 14:06:09,240 - INFO - RUN duration=50.0
    2024-10-23 14:06:09,241 - INFO - STARTING progress tracker
    2024-10-23 14:06:09,242 - INFO - progress=0%
    2024-10-23 14:06:09,270 - INFO - progress=10%
    2024-10-23 14:06:09,295 - INFO - progress=20%
    2024-10-23 14:06:09,324 - INFO - progress=30%
    2024-10-23 14:06:09,350 - INFO - progress=40%
    2024-10-23 14:06:09,377 - INFO - progress=50%
    2024-10-23 14:06:09,405 - INFO - progress=60%
    2024-10-23 14:06:09,433 - INFO - progress=70%
    2024-10-23 14:06:09,465 - INFO - progress=80%
    2024-10-23 14:06:09,503 - INFO - progress=90%
    2024-10-23 14:06:09,538 - INFO - progress=100%
    2024-10-23 14:06:09,540 - INFO - FINISHED steps(total)=1001(1001) runtime=299.13ms
    


    
![png](README_files/README_4_1.png)
    


## Example - Differentiable Simulation

PathSim also includes a rudimentary automatic differentiation framework based on a dual number system with overloaded operators. This makes the system simulation fully differentiable with respect to a predefined set of parameters. For now it only works with the explicit integrators. To demonstrate this lets consider the following linear feedback system.

![png](README_files/linear_feedback_blockdiagram.png)


The source term is a scaled unit step function (scaled by $A$). The parameters we want to differentiate the time domain response by are the feedback term $a$, the initial condition $x_0$ and the amplitude of the source term $A$.


```python
from pathsim import Simulation, Connection
from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope

#AD module
from pathsim.diff import Parameter

#parameters
A  = Parameter(1)
a  = Parameter(-1)
x0 = Parameter(2)

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

    2024-10-23 14:06:12,213 - INFO - LOGGING enabled
    2024-10-23 14:06:12,215 - INFO - SOLVER SSPRK22 adaptive=False implicit=False
    2024-10-23 14:06:12,215 - INFO - PATH LENGTH ESTIMATE 2, 'iterations_min' set to 2
    2024-10-23 14:06:12,216 - INFO - RESET
    2024-10-23 14:06:12,217 - INFO - RUN duration=12
    2024-10-23 14:06:12,218 - INFO - STARTING progress tracker
    2024-10-23 14:06:12,220 - INFO - progress=0%
    2024-10-23 14:06:12,303 - INFO - progress=10%
    2024-10-23 14:06:12,377 - INFO - progress=20%
    2024-10-23 14:06:12,452 - INFO - progress=30%
    2024-10-23 14:06:12,527 - INFO - progress=40%
    2024-10-23 14:06:12,602 - INFO - progress=50%
    2024-10-23 14:06:12,676 - INFO - progress=60%
    2024-10-23 14:06:12,752 - INFO - progress=70%
    2024-10-23 14:06:12,827 - INFO - progress=80%
    2024-10-23 14:06:12,902 - INFO - progress=90%
    2024-10-23 14:06:12,976 - INFO - progress=100%
    2024-10-23 14:06:12,977 - INFO - FINISHED steps(total)=1201(1201) runtime=757.99ms
    


    
![png](README_files/README_6_1.png)
    


Now the recorded data is of type `Parameter` and we can evaluate the automatically computed partial derivatives at each timestep. For example 
$\partial x(t) / \partial a$ the response with respect to the linear feedback parameter.


```python
import matplotlib.pyplot as plt

#read data from the scope
time, [step, data] = Sco.read()

#evaluate partial derivatives
dxda = list(map(lambda x: x.d(a), data))    # w.r.t. feedback
dxdx0 = list(map(lambda x: x.d(x0), data))  # w.r.t. initial condition
dxdA = list(map(lambda x: x.d(A), data))    # w.r.t. source amplitude

fig, ax = plt.subplots(nrows=1, tight_layout=True, figsize=(8, 4), dpi=120)

ax.plot(time, dxda, label="$dx/da$")
ax.plot(time, dxdx0, label="$dx/dx_0$")
ax.plot(time, dxdA, label="$dx/dA$")

ax.set_xlabel("time [s]")
ax.grid(True)
ax.legend(fancybox=False);
```


    
![png](README_files/README_8_0.png)
    


## More Examples
There are many examples of dynamical system simulations in the `examples` directory. They cover almost all the blocks currently available in `PathSim` as well as different numerical integrators / solvers to experiment with.


```python

```
