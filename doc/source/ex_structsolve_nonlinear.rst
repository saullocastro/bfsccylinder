Nonlinear static analysis with structsolve
==========================================

.. currentmodule:: bfsccylinder.bfsccylinder

The callables of ``structsolve.Analysis``, of `structsolve
<https://saullocastro.github.io/structsolve/>`_, for the cylinder of
:doc:`ex_nonlinear_static`, see :ref:`solving-with-structsolve`. They loop
over the elements with :func:`.update_fint`, :func:`.update_KCNL` and
:func:`.update_KG`, assemble, and work on the unknown degrees of freedom
``bu``, the partition of the free degrees of freedom:

.. literalinclude:: ../../tests/test_nonlinear_axial_compression.py
    :pyobject: Callables

The analysis, at the end of the test of :doc:`ex_nonlinear_static`, applies
the load in one increment, ``Analysis.initialInc = 1.``, with at most seven
corrections, ``Analysis.maxNumIter = 7``, and checks that the increment was
not cut:

.. literalinclude:: ../../tests/test_nonlinear_axial_compression.py
    :pyobject: test_nonlinear_axial_compression_load_controlled
    :start-at: callables = Callables(
    :end-at: raise RuntimeError('Not converged!')
    :dedent: 4

The solution is ``u = callables.expand(cs[-1])``. With the default
convergence criterion of Crisfield, `\zeta \le 10^{-9}`, the iterations
give `\zeta` = 2.5e-2, 2.2e-4, 2.0e-9 and then the round-off, about
1e-15, the quadratic convergence of a consistent tangent stiffness matrix.
