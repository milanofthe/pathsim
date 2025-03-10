Stick Slip
----------

In this example we simulate a mechanical system that exhibits stick-slip behaviour, typical for coulomb friction. Lets consider the setup below, where we have a box sitting on a driven conveyor belt. The box is also coupled to a fixation by a spring-damper element. 

You can also find this example as a single file in the `GitHub repository <https://github.com/milanofthe/pathsim/blob/master/examples/examples_event/example_stickslip_event.py>`_.

.. image:: figures/stick_slip.png
   :width: 700
   :align: center
   :alt: schematic of stick slip system


This system has two possible states:
   
   1. The **slip** state where the box oscillates freely. Here we have the dynamical behaviour of a classical damped harmonic oscillator, a 2nd order ODE.

   2. The **stick** state where box exactly follows the belt. Here the box velocity is clamped to the belt velocity (algebraic constraint) and the system dynamics is reduced to a pure 1st order integration.

The two states transition from one to another depending on the relative velocity of the box to the belt and the force acting on the box. If the relative velocity is zero and the force is below some threshold, the system enters the **stick** state. When the force exceeds a certain threshold, the box breaks free and enters the **slip** state.

The continuous time dynamics for the two states have the following ODE(s):

.. math::
   
   \begin{cases}
   m \ddot{x} = F_c - k x - d \dot{x} - \mu_k m g \, \mathrm{sign}\left( \dot{x} - v_b \right) & \text{slip}  \\
   \dot{x} = v_b & \text{stick}
   \end{cases}


with the sticking condition:
   
.. math::
   
   | F_c - k x - d v_b| \leq \mu_s m g


the transition condition from **slip to stick**, when:
   
.. math::

   \dot{x} = v_b \text{ and } |F_c - k x - d v_b| \leq \mu_s m g 


and from **stick to slip**, when 

.. math::

   |F_c - k x - d v_b| > \mu_s m g


The resulting switched system is shown in the block diagram below:

.. image:: figures/stick_slip_blockdiagram.png
   :width: 700
   :align: center
   :alt: block diagram of stick slip system


Note that the **event manager** tracks the system state and sets the switch to select the input of the position integrator.

