Natural frequencies
===================

.. currentmodule:: bfsccylinder.bfsccylinder

The lowest natural frequencies of a cylinder are clustered, and the tests
compute them with a deterministic and verified eigenvalue solver, which
checks the residual of each eigenpair and, with Sylvester's law of inertia,
that no eigenvalue is missing:

.. literalinclude:: ../../tests/natural_frequency_solver.py
    :pyobject: lowest_frequencies

A closed isotropic cylinder with `u`, `v` and `w` constrained at the nodes
of both ends. The mass matrix of :func:`.update_M` requires the thickness
``h`` and the density ``rho`` of each element:

.. literalinclude:: ../../tests/test_natural_frequency.py
    :pyobject: test_natural_frequency
