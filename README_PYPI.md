# PathSim - A System Simulation Framework

## Overview

**PathSim** is a flexible block-based time-domain system simulation framework in Python with automatic differentiation capabilities and an event handling mechanism! It provides a variety of classes that enable modeling and simulating complex interconnected dynamical systems through Python scripting.

All of that with minimal dependencies, only `numpy`, `scipy` and `matplotlib`!

Key Features:

- **Hot-swappable** blocks and solvers during simulation
- Blocks are inherently **MIMO** (Multiple Input, Multiple Output) capable
- Wide range of **numerical integrators** (implicit, explicit, high order, adaptive)
- **Modular and hierarchical** modeling with (nested) subsystems
- **Event handling** system to detect and resolve discrete events (zero-crossing detection)
- Automatic differentiation for **fully differentiable** system simulations
- **Extensibility** by subclassing the base `Block` class and implementing just a handful of methods

For the full **documentation**, tutorials and API-reference visit [readthedocs](https://pathsim.readthedocs.io/en/latest/)!

The source code can be found in the [GitHub repository](https://github.com/milanofthe/pathsim) and is fully open source under **MIT license**. Consider starring PathSim to support its development.


## Installation

The latest release version of PathSim is available on [PyPi](https://pypi.org/project/pathsim/) and installable via pip:

```console
pip install pathsim
```

## Example - Harmonic Oscillator

There are lots of [examples](https://github.com/milanofthe/pathsim/tree/master/examples) of dynamical system simulations in the GitHub repository. 

But first, lets have a look at how we can simulate the harmonic oscillator (a spring mass damper 2nd order system) using PathSim. The system and its corresponding equivalent block diagram are shown in the figure below:

![png](https://raw.githubusercontent.com/milanofthe/pathsim/master/README_files/harmonic_oscillator.png)

The equation of motion that defines the harmonic oscillator it is give by

$$
\ddot{x} + \frac{c}{m} \dot{x} + \frac{k}{m} x = 0
$$

where $c$ is the damping, $k$ the spring constant and $m$ the mass together with the initial conditions  $x_0$ and $v_0$ for position and velocity.

The topology of the block diagram above can be directly defined as blocks and connections in the PathSim framework. First we initialize the blocks needed to represent the dynamical systems with their respective arguments such as initial conditions and gain values, then the blocks are connected using `Connection` objects, forming two feedback loops.

The `Simulation` instance manages the blocks and connections and advances the system in time with the timestep (`dt`). The `log` flag for logging the simulation progress is also set. Finally, we run the simulation for some number of seconds and plot the results using the `plot()` method of the scope block.


```python
from pathsim import Simulation, Connection

#import the blocks we need for the harmonic oscillator
from pathsim.blocks import Integrator, Amplifier, Adder, Scope

#initial position and velocity
x0, v0 = 2, 5

#parameters (mass, damping, spring constant)
m, c, k = 0.8, 0.2, 1.5

#define the blocks 
I1 = Integrator(v0)   # integrator for velocity
I2 = Integrator(x0)   # integrator for position
A1 = Amplifier(-c/m)
A2 = Amplifier(-k/m)
P1 = Adder()
Sc = Scope(labels=["v(t)", "x(t)"])

blocks = [I1, I2, A1, A2, P1, Sc]

#define the connections between the blocks
connections = [
    Connection(I1, I2, A1, Sc),   # one to many connection
    Connection(I2, A2, Sc[1]),
    Connection(A1, P1),           # default connection to port 0
    Connection(A2, P1[1]),        # specific connection to port 1
    Connection(P1, I1)
    ]

#create a simulation instance from the blocks and connections
Sim = Simulation(blocks, connections, dt=0.05, log=True)

#run the simulation for 30 seconds
Sim.run(duration=30.0)

#plot the results directly from the scope
Sc.plot()

#read the results from the scope for further processing
time, data = Sc.read();
```

    2025-03-08 10:23:20,415 - INFO - LOGGING enabled
    2025-03-08 10:23:20,417 - INFO - SOLVER -> SSPRK22, adaptive=False, implicit=False
    2025-03-08 10:23:20,417 - INFO - ALGEBRAIC PATH LENGTH 2
    2025-03-08 10:23:20,417 - INFO - RESET, time -> 0.0
    2025-03-08 10:23:20,418 - INFO - TRANSIENT duration=30.0
    2025-03-08 10:23:20,418 - INFO - STARTING progress tracker
    2025-03-08 10:23:20,419 - INFO - progress=0%
    2025-03-08 10:23:20,424 - INFO - progress=10%
    2025-03-08 10:23:20,428 - INFO - progress=20%
    2025-03-08 10:23:20,433 - INFO - progress=30%
    2025-03-08 10:23:20,438 - INFO - progress=40%
    2025-03-08 10:23:20,443 - INFO - progress=50%
    2025-03-08 10:23:20,448 - INFO - progress=60%
    2025-03-08 10:23:20,452 - INFO - progress=70%
    2025-03-08 10:23:20,457 - INFO - progress=80%
    2025-03-08 10:23:20,462 - INFO - progress=90%
    2025-03-08 10:23:20,467 - INFO - progress=100%
    2025-03-08 10:23:20,467 - INFO - FINISHED, steps(total)=600(600), runtime=48.32ms
    


![png](https://raw.githubusercontent.com/milanofthe/pathsim/master/README_files/README_4_1.png)