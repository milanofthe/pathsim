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

    2025-03-08 10:56:04,149 - INFO - LOGGING enabled
    2025-03-08 10:56:04,150 - INFO - SOLVER -> SSPRK22, adaptive=False, implicit=False
    2025-03-08 10:56:04,150 - INFO - ALGEBRAIC PATH LENGTH 2
    2025-03-08 10:56:04,151 - INFO - RESET, time -> 0.0
    2025-03-08 10:56:04,151 - INFO - TRANSIENT duration=30.0
    2025-03-08 10:56:04,151 - INFO - STARTING progress tracker
    2025-03-08 10:56:04,152 - INFO - progress=0%
    2025-03-08 10:56:04,156 - INFO - progress=10%
    2025-03-08 10:56:04,160 - INFO - progress=20%
    2025-03-08 10:56:04,165 - INFO - progress=30%
    2025-03-08 10:56:04,170 - INFO - progress=40%
    2025-03-08 10:56:04,174 - INFO - progress=50%
    2025-03-08 10:56:04,179 - INFO - progress=60%
    2025-03-08 10:56:04,184 - INFO - progress=70%
    2025-03-08 10:56:04,188 - INFO - progress=80%
    2025-03-08 10:56:04,192 - INFO - progress=90%
    2025-03-08 10:56:04,197 - INFO - progress=100%
    2025-03-08 10:56:04,198 - INFO - FINISHED, steps(total)=600(600), runtime=46.33ms
    


        
![png](https://raw.githubusercontent.com/milanofthe/pathsim/master/README_files/README_4_1.png)


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

#implicit adaptive timestep solver 
from pathsim.solvers import ESDIRK54

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
Sim = Simulation(
    blocks, 
    connections, 
    dt=0.05, 
    log=True, 
    Solver=ESDIRK54, 
    tolerance_lte_abs=1e-5, 
    tolerance_lte_rel=1e-3
    )

#run simulation for some number of seconds
Sim.run(3*mu)

#plot the results directly (steps highlighted)
Sc.plot(".-");
```

    2025-03-08 10:56:04,872 - INFO - LOGGING enabled
    2025-03-08 10:56:04,872 - INFO - SOLVER -> ESDIRK54, adaptive=True, implicit=True
    2025-03-08 10:56:04,872 - INFO - ALGEBRAIC PATH LENGTH 1
    2025-03-08 10:56:04,873 - INFO - RESET, time -> 0.0
    2025-03-08 10:56:04,873 - INFO - TRANSIENT duration=3000
    2025-03-08 10:56:04,874 - INFO - STARTING progress tracker
    2025-03-08 10:56:04,880 - INFO - progress=0%
    2025-03-08 10:56:04,974 - INFO - progress=11%
    2025-03-08 10:56:05,031 - INFO - progress=20%
    2025-03-08 10:56:05,936 - INFO - progress=32%
    2025-03-08 10:56:06,023 - INFO - progress=42%
    2025-03-08 10:56:06,119 - INFO - progress=51%
    2025-03-08 10:56:07,035 - INFO - progress=63%
    2025-03-08 10:56:07,103 - INFO - progress=70%
    2025-03-08 10:56:07,384 - INFO - progress=80%
    2025-03-08 10:56:08,276 - INFO - progress=90%
    2025-03-08 10:56:08,325 - INFO - progress=100%
    2025-03-08 10:56:08,326 - INFO - FINISHED, steps(total)=228(397), runtime=3451.53ms
    

![png](https://raw.githubusercontent.com/milanofthe/pathsim/master/README_files/README_6_1.png)


## Differentiable Simulation

PathSim also includes a fully fledged automatic differentiation framework based on a dual number system with overloaded operators and numpy ufunc integration. This makes the system simulation fully differentiable end-to-end with respect to a predefined set of parameters. Works with all integrators, adaptive, fixed, implicit, explicit. 

To demonstrate this lets consider the following linear feedback system and perform a sensitivity analysis on it with respect to some system parameters. 


![png](https://raw.githubusercontent.com/milanofthe/pathsim/master/README_files/linear_feedback_blockdiagram.png)


The source term is a scaled unit step function (scaled by $b$). In this example, the parameters for the sensitivity analysis are the feedback term $a$, the initial condition $x_0$ and the amplitude of the source term $b$.


```python
from pathsim import Simulation, Connection
from pathsim.blocks import Source, Integrator, Amplifier, Adder, Scope

#AD module
from pathsim.optim import Value, der

#values for derivative propagation / parameters for sensitivity analysis
a  = Value(-1)
b  = Value(1)
x0 = Value(2)

#simulation timestep
dt = 0.01

#step function
tau = 3
def s(t):
    return b*int(t>tau)

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

    2025-03-08 10:56:08,480 - INFO - LOGGING enabled
    2025-03-08 10:56:08,481 - INFO - SOLVER -> SSPRK22, adaptive=False, implicit=False
    2025-03-08 10:56:08,481 - INFO - ALGEBRAIC PATH LENGTH 2
    2025-03-08 10:56:08,481 - INFO - RESET, time -> 0.0
    2025-03-08 10:56:08,482 - INFO - TRANSIENT duration=12
    2025-03-08 10:56:08,482 - INFO - STARTING progress tracker
    2025-03-08 10:56:08,483 - INFO - progress=0%
    2025-03-08 10:56:08,505 - INFO - progress=10%
    2025-03-08 10:56:08,525 - INFO - progress=20%
    2025-03-08 10:56:08,545 - INFO - progress=30%
    2025-03-08 10:56:08,564 - INFO - progress=40%
    2025-03-08 10:56:08,583 - INFO - progress=50%
    2025-03-08 10:56:08,603 - INFO - progress=60%
    2025-03-08 10:56:08,623 - INFO - progress=70%
    2025-03-08 10:56:08,642 - INFO - progress=80%
    2025-03-08 10:56:08,661 - INFO - progress=90%
    2025-03-08 10:56:08,680 - INFO - progress=100%
    2025-03-08 10:56:08,680 - INFO - FINISHED, steps(total)=1201(1201), runtime=198.23ms
    


![png](https://raw.githubusercontent.com/milanofthe/pathsim/master/README_files/README_8_1.png)


Now the recorded time series data is of type `Value` and we can evaluate the automatically computed partial derivatives at each timestep. For example the response, differentiated with respect to the linear feedback parameter $\partial x(t) / \partial a$ can be extracted from the data like this `der(data, a)`.


```python
import matplotlib.pyplot as plt

#read data from the scope
time, [step, data] = Sco.read()

fig, ax = plt.subplots(nrows=1, tight_layout=True, figsize=(8, 4), dpi=120)

#evaluate and plot partial derivatives
ax.plot(time, der(data, a), label=r"$\partial x / \partial a$")
ax.plot(time, der(data, x0), label=r"$\partial x / \partial x_0$")
ax.plot(time, der(data, b), label=r"$\partial x / \partial b$")

ax.set_xlabel("time [s]")
ax.grid(True)
ax.legend(fancybox=False);
```


![png](https://raw.githubusercontent.com/milanofthe/pathsim/master/README_files/README_10_0.png)


## Event Detection

PathSim has an event handling system that monitors the simulation state and can find and locate discrete events by evaluating an event function and trigger callbacks or state transformations. Multiple event types are supported such as `ZeroCrossing` or `Schedule`. 

This enables the simulation of hybrid continuous time systems with discrete events. 


![png](https://raw.githubusercontent.com/milanofthe/pathsim/master/README_files/bouncing_ball.png)


Probably the most popular example for this is the bouncing ball (see figure above) where discrete events occur whenever the ball touches the floor. The event in this case is a zero-crossing.


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

#event function for zero crossing detection
def func_evt(t):
    i, o, s = Ix() #get block inputs, outputs and states
    return s

#action function for state transformation
def func_act(t):
    i1, o1, s1 = Ix() 
    i2, o2, s2 = Iv() 
    Ix.engine.set(abs(s1))
    Iv.engine.set(-0.9*s2)

#event (zero-crossing) -> ball makes contact
E1 = ZeroCrossing(
    func_evt=func_evt,                 
    func_act=func_act, 
    tolerance=1e-4
    )

events = [E1]

#initialize simulation with the blocks, connections, timestep and logging enabled
Sim = Simulation(
    blocks, 
    connections, 
    events, 
    dt=0.1, 
    log=True, 
    Solver=RKBS32, 
    dt_max=0.1
    )

#run the simulation
Sim.run(20)

#plot the recordings from the scope
Sc.plot();
```

    2025-03-08 10:56:11,436 - INFO - LOGGING enabled
    2025-03-08 10:56:11,436 - INFO - SOLVER -> RKBS32, adaptive=True, implicit=False
    2025-03-08 10:56:11,437 - INFO - ALGEBRAIC PATH LENGTH 1
    2025-03-08 10:56:11,437 - INFO - RESET, time -> 0.0
    2025-03-08 10:56:11,438 - INFO - TRANSIENT duration=20
    2025-03-08 10:56:11,438 - INFO - STARTING progress tracker
    2025-03-08 10:56:11,439 - INFO - progress=0%
    2025-03-08 10:56:11,441 - INFO - progress=10%
    2025-03-08 10:56:11,446 - INFO - progress=20%
    2025-03-08 10:56:11,452 - INFO - progress=30%
    2025-03-08 10:56:11,456 - INFO - progress=40%
    2025-03-08 10:56:11,461 - INFO - progress=50%
    2025-03-08 10:56:11,467 - INFO - progress=60%
    2025-03-08 10:56:11,473 - INFO - progress=70%
    2025-03-08 10:56:11,480 - INFO - progress=80%
    2025-03-08 10:56:11,488 - INFO - progress=90%
    2025-03-08 10:56:11,500 - INFO - progress=100%
    2025-03-08 10:56:11,501 - INFO - FINISHED, steps(total)=395(496), runtime=61.73ms
    


![png](https://raw.githubusercontent.com/milanofthe/pathsim/master/README_files/README_12_1.png)


During the event handling, the simulator approaches the event until the specified tolerance is met. You can see this by analyzing the timesteps taken by the adaptive integrator `RKBS32`.


```python
import numpy as np
import matplotlib.pyplot as plt

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


![png](https://raw.githubusercontent.com/milanofthe/pathsim/master/README_files/README_14_0.png)


## Contributing

PathSim is in active development and your feedback is highly appreciated! Dont shy away from filing issues or requests! If you want to contribute in a mayor way, send me a message and we can discuss how!

## Roadmap

Some of the possible directions for future PathSim are:

- block level parallelization (fork-join) with Python 3.13 free-threading, batching based on execution cost
- linearization of blocks and subsystems with the AD framework, linear surrogate models, system wide linearization
- improved / more robust steady state solver and algebraic loop solver
- methods for periodic steady state analysis
- more extensive testing and validation (as always)
