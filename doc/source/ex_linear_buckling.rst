Linear buckling analysis
========================

.. currentmodule:: bfsccylinder.bfsccylinder

Geometric stiffness matrix of a static solution
-----------------------------------------------

A laminated cylinder with `u`, `v` and `w` constrained at the nodes of both
ends, compressed by a prescribed axial displacement of the end `x = L`. The
linear static solution gives the geometric stiffness matrix of
:func:`.update_KG`, the lowest load multiplier is computed with
``lobpcg``, and the critical load is the load multiplier times the sum of
the reaction forces at `x = L`:

.. literalinclude:: ../../tests/test_linear_buckling.py
    :pyobject: test_linear_buckling

Constant membrane stress
------------------------

A laminated cylinder with `u`, `v`, `w` and `w_{,x}` constrained at the
nodes of both ends, under an axial compression ``Nxx = -1`` given directly
to :func:`.update_KG_constant_stress`, without a static analysis, such that
the critical load is the load multiplier times `-2 \pi R`:

.. literalinclude:: ../../tests/test_linear_buckling_Z33_constant_stress.py
    :pyobject: test_linear_buckling
