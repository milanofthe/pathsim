# ODE Solver Library

Here the ODE solvers available in PathSim are defined. They all inherit from the bas `Solver` class defined in `_solver.py`. Then there is the distinction between implicit and explicit solvers through the classes `ExplicitSolver` and `ImplicitSolver`. Different kinds of methods have their internal workings defined in additional classes, such as eplicit runge-kutta methods in `_rungekutta.py` with `ExplicitRungeKutta` inheriting from `ExplicitSolver`, and `DiagonallyImplicitRungeKutta` inheriting from `ImplicitSolver`. There are also some multistep methods such as `GEAR` and `BDF` defined in their respective modules.

Solvers can be imported directly from the solver library like this:

```python
from pathsim.solvers import EUF, RKCK54
```

Or from the modules where they are defined:

```python
from pathsim.solvers.euler import EUF
from pathsim.solvers.rkck54 import RKCK54
```