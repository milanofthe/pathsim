
Package Structure
=================

Here, we dive deeper into the structure and architecture of PathSim.



Object Hierarchy
----------------

The basic object hierarchy of PathSim is shown in the diagramm below.

.. image:: figures/pathsim_object_hierarchy.png
   :width: 700
   :align: center
   :alt: hierarchy of PathSim objects


Available Solvers
-----------------

The numerical integrators, available in PathSim are structured like this:

.. image:: figures/pathsim_solver_hierarchy.png
   :width: 700
   :align: center
   :alt: hierarchy of PathSim numerical integrators

They are implemented from scratch to expose their interfaces to the PathSim simulation loop. The integrators are validated against reference problems in the test suite. 