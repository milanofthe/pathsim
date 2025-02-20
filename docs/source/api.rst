
API Reference
=============


The basic object hierarchy of PathSim is shown in the diagramm below.

.. image:: figures/pathsim_object_hierarchy.png
   :width: 700
   :align: center
   :alt: hierarchy of PathSim objects


System Definition and Simulation
--------------------------------

The following modules serve the system definition and simulation.

.. toctree::
   :maxdepth: 3
   :caption: Simulation

   modules/pathsim.simulation
   
.. toctree::
   :maxdepth: 3
   :caption: Subsystem

   modules/pathsim.subsystem
   
.. toctree::
   :maxdepth: 3
   :caption: Connections

   modules/pathsim.connection
   
.. toctree::
   :maxdepth: 5
   :caption: Blocks

   modules/pathsim.blocks

.. toctree::
   :maxdepth: 5
   :caption: Events

   modules/pathsim.events



Solvers and Utils
-----------------

These modules are internal to the simulation loop.


.. toctree::
   :maxdepth: 4
   :caption: Solvers

   modules/pathsim.solvers

.. toctree::
   :maxdepth: 4
   :caption: Optimizers and AD

   modules/pathsim.optim

.. toctree::
   :maxdepth: 4
   :caption: Utils

   modules/pathsim.utils
