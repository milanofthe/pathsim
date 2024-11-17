# PathSim: A Time-Domain System Simulation Framework


## Overview

PathSim is a flexible block-based time-domain system simulation framework in Python with automatic differentiation capabilities and an event handling mechanism. It provides a variety of classes that enable modeling and simulating complex interconnected dynamical systems similar to Matlab Simulink but in Python!

Key features of PathSim include:

- Natural handling of algebraic loops
- Hot-swappable blocks and solvers during simulation
- Blocks are inherently MIMO (Multiple Input, Multiple Output) capable
- Blocks are "physicalized" and manage their own state, i.e. reading from the scope is just scope.read()
- Scales linearly with the number of blocks and connections
- Wide range of numerical solvers, including implicit and explicit very high order Runge-Kutta and multistep methods
- Modular and hierarchical modeling with (nested) subsystems
- Event handling system that can detect and resolve discrete events (zero-crossing detection)
- Automatic differentiation for fully differentiable system simulations (even through events) for sensitivity analysis and optimization
- Library of pre-defined blocks, including mathematical operations, integrators, delays, transfer functions, etc.
- Easy extensibility, subclassing the base `Block` class with just a handful of methods

All features are demonstrated for benchmark problems in the `example` directory.

## Installation

The latest release version of pathsim is available on [PyPi](https://pypi.org/project/pathsim/)  and installable via pip:

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
time, data = Sc.read();
```

    2024-11-17 11:26:20,457 - INFO - LOGGING enabled
    2024-11-17 11:26:20,458 - INFO - SOLVER SSPRK22 adaptive=False implicit=False
    2024-11-17 11:26:20,459 - INFO - PATH LENGTH ESTIMATE 2, 'iterations_min' set to 2
    2024-11-17 11:26:20,459 - INFO - RESET
    2024-11-17 11:26:20,459 - INFO - RUN duration=50.0
    2024-11-17 11:26:20,460 - INFO - STARTING progress tracker
    2024-11-17 11:26:20,460 - INFO - progress=0%
    2024-11-17 11:26:20,478 - INFO - progress=10%
    2024-11-17 11:26:20,495 - INFO - progress=20%
    2024-11-17 11:26:20,513 - INFO - progress=30%
    2024-11-17 11:26:20,530 - INFO - progress=40%
    2024-11-17 11:26:20,548 - INFO - progress=50%
    2024-11-17 11:26:20,564 - INFO - progress=60%
    2024-11-17 11:26:20,583 - INFO - progress=70%
    2024-11-17 11:26:20,600 - INFO - progress=80%
    2024-11-17 11:26:20,617 - INFO - progress=90%
    2024-11-17 11:26:20,634 - INFO - progress=100%
    2024-11-17 11:26:20,635 - INFO - FINISHED steps(total)=1001(1001) runtime=175.09ms
    


    
![png](README_files/README_4_1.png)
    


## Stiff Systems

PathSim implements a large variety of implicit integrators such as diagonally implicit runge-kutta (`DIRK2`, `ESDIRK43`, etc.) and multistep (`BDF2`, `GEAR52A`, etc.) methods. This enables the simulation of very stiff systems where the timestep is limited by stability and not accuracy of the method.

A common example for a stiff system is the Van der Pol oscillator where the parameter $\mu$ "controls" the severity of the stiffness. It is defined by the following second order ODE:

$$
\ddot{x} + \mu (1 - x^2) \dot{x} + x = 0
$$

Below, the Van der Pol system is built with two discrete `Integrator` blocks and a `Function` block. The parameter is set to $\mu = 1000$ which means severe stiffness. 


```python
from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Scope, Function

#implicit adaptive timestep adaptive order solver 
from pathsim.solvers import GEAR52A

#initial conditions
x1, x2 = 2, 0

#van der Pol parameter (1000 is very stiff)
mu = 1000

#blocks that define the system
Sc = Scope(labels=["$x_1(t)$"])
I1 = Integrator(x1)
I2 = Integrator(x2)
Fn = Function(lambda x1, x2: mu*(1 - x1**2)*x2 - x1)

blocks = [I1, I2, Fn, Sc]

#the connections between the blocks
connections = [
    Connection(I2, I1, Fn[1]), 
    Connection(I1, Fn, Sc), 
    Connection(Fn, I2)
    ]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, dt=0.05, log=True, Solver=GEAR52A, tolerance_lte_abs=1e-6, tolerance_lte_rel=1e-4)

#run simulation for some number of seconds
Sim.run(3*mu)

#plot the results directly (steps highlighted)
Sc.plot(".-");
```

    2024-11-17 11:26:31,260 - INFO - LOGGING enabled
    2024-11-17 11:26:31,260 - INFO - SOLVER GEAR52A adaptive=True implicit=True
    2024-11-17 11:26:31,261 - INFO - PATH LENGTH ESTIMATE 1, 'iterations_min' set to 1
    2024-11-17 11:26:31,261 - INFO - RESET
    2024-11-17 11:26:31,262 - INFO - RUN duration=3000
    2024-11-17 11:26:31,262 - INFO - STARTING progress tracker
    2024-11-17 11:26:31,284 - INFO - progress=0%
    2024-11-17 11:26:31,677 - INFO - progress=13%
    2024-11-17 11:26:31,720 - INFO - progress=21%
    2024-11-17 11:26:32,291 - INFO - progress=30%
    2024-11-17 11:26:32,317 - INFO - progress=43%
    2024-11-17 11:26:32,371 - INFO - progress=50%
    2024-11-17 11:26:32,940 - INFO - progress=61%
    2024-11-17 11:26:32,962 - INFO - progress=70%
    2024-11-17 11:26:33,080 - INFO - progress=80%
    2024-11-17 11:26:33,591 - INFO - progress=92%
    2024-11-17 11:26:33,619 - INFO - progress=100%
    2024-11-17 11:26:33,619 - INFO - FINISHED steps(total)=682(880) runtime=2357.86ms
    


    
![png](README_files/README_6_1.png)
    


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

    2024-11-17 11:26:41,033 - INFO - LOGGING enabled
    2024-11-17 11:26:41,034 - INFO - SOLVER SSPRK22 adaptive=False implicit=False
    2024-11-17 11:26:41,035 - INFO - PATH LENGTH ESTIMATE 2, 'iterations_min' set to 2
    2024-11-17 11:26:41,035 - INFO - RESET
    2024-11-17 11:26:41,036 - INFO - RUN duration=12
    2024-11-17 11:26:41,036 - INFO - STARTING progress tracker
    2024-11-17 11:26:41,037 - INFO - progress=0%
    2024-11-17 11:26:41,088 - INFO - progress=10%
    2024-11-17 11:26:41,138 - INFO - progress=20%
    2024-11-17 11:26:41,188 - INFO - progress=30%
    2024-11-17 11:26:41,237 - INFO - progress=40%
    2024-11-17 11:26:41,287 - INFO - progress=50%
    2024-11-17 11:26:41,339 - INFO - progress=60%
    2024-11-17 11:26:41,389 - INFO - progress=70%
    2024-11-17 11:26:41,439 - INFO - progress=80%
    2024-11-17 11:26:41,489 - INFO - progress=90%
    2024-11-17 11:26:41,539 - INFO - progress=100%
    2024-11-17 11:26:41,540 - INFO - FINISHED steps(total)=1201(1201) runtime=503.43ms
    


    
![png](README_files/README_8_1.png)
    


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


    
![png](README_files/README_10_0.png)
    


## Event Detection

PathSim has an event handling system that watches states of dynamic blocks or outputs of static blocks and can trigger callbacks or state teansformations. etection. This enables the simulation of hybrid continuous time systems with discrete events. Probably the most popular example for this is the bouncing ball where discrete events occur whenever the ball touches the floor.


```python
from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Constant, Scope
from pathsim.solvers import RKBS32

#event library
from pathsim.events import ZeroCrossing

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
E1 = ZeroCrossing(
    blocks=[Ix, Iv],                  # blocks to watch states of
    g=lambda x, y: x,                 # event function for zero crossing detection
    f=lambda x, y: [abs(x), -0.9*y],  # action function for state transformation
    tolerance=1e-4
    )

events = [E1]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(blocks, connections, events, dt=0.1, log=True, Solver=RKBS32, dt_max=0.1)

#run the simulation
Sim.run(20)

#plot the recordings from the scope
Sc.plot();
```

    2024-11-17 11:17:53,524 - INFO - LOGGING enabled
    2024-11-17 11:17:53,524 - INFO - SOLVER RKBS32 adaptive=True implicit=False
    2024-11-17 11:17:53,525 - INFO - PATH LENGTH ESTIMATE 1, 'iterations_min' set to 1
    2024-11-17 11:17:53,525 - INFO - RESET
    2024-11-17 11:17:53,526 - INFO - RUN duration=20
    2024-11-17 11:17:53,526 - INFO - STARTING progress tracker
    2024-11-17 11:17:53,527 - INFO - progress=0%
    2024-11-17 11:17:53,532 - INFO - progress=10%
    2024-11-17 11:17:53,539 - INFO - progress=20%
    2024-11-17 11:17:53,548 - INFO - progress=30%
    2024-11-17 11:17:53,558 - INFO - progress=40%
    2024-11-17 11:17:53,565 - INFO - progress=50%
    2024-11-17 11:17:53,575 - INFO - progress=60%
    2024-11-17 11:17:53,585 - INFO - progress=70%
    2024-11-17 11:17:53,598 - INFO - progress=80%
    2024-11-17 11:17:53,610 - INFO - progress=90%
    2024-11-17 11:17:53,632 - INFO - progress=100%
    2024-11-17 11:17:53,633 - INFO - FINISHED steps(total)=395(496) runtime=106.76ms
    


    
![png](README_files/README_12_1.png)
    


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


    
![png](README_files/README_14_0.png)
    



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
