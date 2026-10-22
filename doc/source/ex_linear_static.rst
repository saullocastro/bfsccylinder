Linear static analysis
======================

.. currentmodule:: bfsccylinder.bfsccylinder

A closed isotropic cylinder with `u`, `v` and `w` constrained at the nodes
of both ends, under a normal force at the node in the middle. The mesh is
closed around the circumference as in :doc:`assembling`, the stiffness
matrix is assembled with :func:`.update_KC0` and the system is solved with
the conjugate gradient method. The code is extracted from one of the
``bfsccylinder`` unit tests:

.. literalinclude:: ../../tests/test_linear_point_load.py
    :pyobject: test_point_load
