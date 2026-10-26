Follower pressure
=================

.. currentmodule:: bfsccylinder.bfsccylinder

A follower pressure stays normal to the deformed mid-surface of the shell
and acts on its deformed area, e.g. a gas pressure, as opposed to a dead
pressure, which keeps the direction of the undeformed normal and the
undeformed area. Its force vector depends on the displacements. Two
functions of each kinematics give it:

- :func:`.update_fext_follower` adds the force vector `\boldsymbol{F}_p(\boldsymbol{u})` of one
  element to a global force vector;
- :func:`.update_KCfollower` writes the load stiffness matrix
  `\boldsymbol{K}_F = -\partial \boldsymbol{F}_p/\partial \boldsymbol{u}` of one element in
  sparse arrays, at the position ``init_k_KCfollower``, with
  ``KCfollower_SPARSE_SIZE`` entries, see :doc:`assembling`.

The theory, from the area vector of the deformed mid-surface, and the
differences between the Donnell-type and the Sanders-type kinematics are
given in the Notes of these functions and of the modules
:mod:`bfsccylinder.bfsccylinder` and :mod:`bfsccylinder.bfsccylinder_sanders`.


Pressure at the integration points
----------------------------------

The pressure ``p`` is given to both functions at the integration points of
each element, as a C-contiguous array of shape ``(nint, nint)``, ``p[i,
j]`` at ``xi = points[i]`` and ``eta = points[j]``, as the stiffness arrays
``A11``, ..., per unit area of the undeformed mid-surface and as a function
of the undeformed coordinates, e.g. ``np.full((nint, nint), p)`` for a
uniform pressure. The pressure is multiplied by the load factor before it
is given to the functions.


Sign
----

A positive ``p`` pushes along `+z`, outwards, the direction of a positive
`w`, such that an external pressure is negative. At `\boldsymbol{u} = \boldsymbol{0}` the
force vector is the dead pressure load, `\int_A p (\boldsymbol{S}^w)^\top dA`, the
consistent load vector of a dead pressure, for both truncations below.


Truncation of the area vector
-----------------------------

``quadratic=0``, the default, truncates the area vector to its terms of
order 0 and 1 in the displacement gradients, the truncation consistent with
the moderate rotations of the strains. `\boldsymbol{F}_p` is then affine in the
displacements and `\boldsymbol{K}_F` constant, computed once for a fixed
pressure. ``quadratic=1`` adds the terms of order 2, e.g. to check that a
result does not depend on the truncation; `\boldsymbol{K}_F` then depends on
the displacements and is computed at each iteration.

``reference=0`` in :func:`.update_fext_follower` leaves out the dead part,
giving `\boldsymbol{F}_p(\boldsymbol{u}) - \boldsymbol{F}_p(\boldsymbol{0})`, which vanishes at `\boldsymbol{u} = \boldsymbol{0}`.


Residual and tangent stiffness matrix
-------------------------------------

With the pressure `\lambda p_0` of a load factor `\lambda`, the residual of
the nonlinear static analysis of :doc:`analyses` and its Jacobian are

.. math::

    \boldsymbol{R}(\boldsymbol{u}) = \boldsymbol{F}_{int}(\boldsymbol{u}) - \boldsymbol{F}_p(\boldsymbol{u}), \qquad
    \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u}) + \boldsymbol{K}_{GNL}(\boldsymbol{u}) + \boldsymbol{K}_G(\boldsymbol{u}) + \boldsymbol{K}_F

with `\boldsymbol{F}_p` and `\boldsymbol{K}_F` of the same ``p = lambda*p0`` and
``quadratic``, see ``residual_and_tangent`` in :doc:`ex_follower_pressure`.
A solver whose external force vector is fixed, `\lambda \boldsymbol{F}_{ext}`, takes the
dead part `\boldsymbol{F}_{ext} = \boldsymbol{F}_p(\boldsymbol{0})` of ``p0`` as external forces and the part
``reference=0``, `\lambda(\boldsymbol{F}_p(\boldsymbol{u}) - \boldsymbol{F}_p(\boldsymbol{0}))`, with the internal
forces.

A linear buckling analysis under a follower pressure `\lambda p_0` adds the
load stiffness matrix `\boldsymbol{K}_{F_{ref}}` of ``p0`` to `\boldsymbol{K}_G` of the linear static solution of
``p0``:

.. math::

    (\boldsymbol{K}_{C0} + \lambda (\boldsymbol{K}_G + \boldsymbol{K}_{F_{ref}}))\boldsymbol{\phi} = \boldsymbol{0}

as in the validation tests of rings and cylinders, see
:doc:`ex_follower_pressure`. Under a dead pressure `\boldsymbol{K}_{F_{ref}}` is left
out. With the Sanders-type kinematics and a dead pressure, a rigid rotation
`v` = constant of a ring left free gives a spurious buckling mode at a
vanishing load, which the follower load stiffness cancels, see the Notes of
:mod:`bfsccylinder.bfsccylinder_sanders`.


Unsymmetric load stiffness matrix
---------------------------------

`\boldsymbol{K}_F` is unsymmetric in general, a non-conservative load. For
the first-order area vector, ``quadratic=0``, it is symmetric, and the
follower pressure conservative, if and only if the pressure is uniform
over the loaded region and, at each point of the contour of the loaded
region, `w = 0` or the in-plane displacement normal to the contour is
zero, e.g. a cylinder closed around the circumference with simply
supported or clamped ends, or symmetry planes. The matrix of one element,
free or partially restrained edges, a pressure on a patch whose edges lie
inside the shell or a varying pressure give an unsymmetric matrix; in the
assembled matrix, restricted to the unknown degrees of freedom, the
contributions of the edges shared by two elements cancel. The conditions are given in the Notes of
:mod:`bfsccylinder.bfsccylinder`. The matrix is written in full and must not
be symmetrized: the linear systems and the eigenvalue problems are solved
with the solvers of unsymmetric matrices, e.g. ``structsolve.lb`` with
``symmetric=False`` in the validation tests.
