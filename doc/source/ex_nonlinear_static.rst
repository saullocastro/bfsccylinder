Nonlinear static analysis
=========================

.. currentmodule:: bfsccylinder.bfsccylinder

A laminated cylinder under axial compression, with `v` and `w` constrained
at the nodes of both ends and `u` at the nodes of the middle section, solved
in one load step with the full Newton-Raphson method of
``structsolve.Analysis``, whose callables are given in
:doc:`ex_structsolve_nonlinear`. The tangent stiffness matrix `[K_{C0}] +
[K_{CNL}] + [K_G]` is rebuilt at every iteration, zeroing the value arrays
of `[K_{CNL}]` and `[K_G]` before :func:`.update_KCNL` and
:func:`.update_KG` add to them, and the residual is that of the internal
forces of :func:`.update_fint`. The state reached is mild and the test
checks only the convergence, see its docstring;
``tests/test_tangent_consistency.py`` checks that the tangent stiffness
matrix is the Jacobian of the internal forces:

.. literalinclude:: ../../tests/test_nonlinear_axial_compression.py
    :pyobject: test_nonlinear_axial_compression_load_controlled
