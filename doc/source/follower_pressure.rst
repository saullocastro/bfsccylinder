Follower pressure
=================

.. currentmodule:: bfsccylinder.bfsccylinder

A follower pressure stays normal to the deformed mid-surface of the shell
and acts on its deformed area, e.g. a gas pressure, as opposed to a dead
pressure, which keeps the direction of the undeformed normal and the
undeformed area. Its force vector depends on the displacements. Two
functions of each kinematics give it:

- :func:`.update_fext_follower` adds the force vector `\{F_p(u)\}` of one
  element to a global force vector;
- :func:`.update_KCfollower` writes the load stiffness matrix
  `[K_{Cfollower}] = -\partial \{F_p\}/\partial \{u\}` of one element in
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
`w`, such that an external pressure is negative. At `\{u\} = \{0\}` the
force vector is the dead pressure load, `\int_A p [S_w]^T dA`, the
consistent load vector of a dead pressure, for both truncations below.


Truncation of the area vector
-----------------------------

``quadratic=0``, the default, truncates the area vector to its terms of
order 0 and 1 in the displacement gradients, the truncation consistent with
the moderate rotations of the strains. `\{F_p\}` is then affine in the
displacements and `[K_{Cfollower}]` constant, computed once for a fixed
pressure. ``quadratic=1`` adds the terms of order 2, e.g. to check that a
result does not depend on the truncation; `[K_{Cfollower}]` then depends on
the displacements and is computed at each iteration.

``reference=0`` in :func:`.update_fext_follower` leaves out the dead part,
giving `\{F_p(u)\} - \{F_p(0)\}`, which vanishes at `\{u\} = \{0\}`.


Residual and tangent stiffness matrix
-------------------------------------

With the pressure `\lambda p_0` of a load factor `\lambda`, the residual of
the nonlinear static analysis of :doc:`analyses` and its Jacobian are

.. math::

    \{R(u)\} = \{f_{int}(u)\} - \{F_p(u)\}, \qquad
    [K_T] = [K_{C0}] + [K_{CNL}(u)] + [K_G(u)] + [K_{Cfollower}]

with `\{F_p\}` and `[K_{Cfollower}]` of the same ``p = lambda*p0`` and
``quadratic``, see ``residual_and_tangent`` in :doc:`ex_follower_pressure`.
A solver whose external force vector is fixed, `\lambda \{f\}`, takes the
dead part `\{f\} = \{F_p(0)\}` of ``p0`` as external forces and the part
``reference=0``, `\lambda(\{F_p(u)\} - \{F_p(0)\})`, with the internal
forces.

A linear buckling analysis under a follower pressure `\lambda p_0` adds the
load stiffness matrix of ``p0`` to `[K_G]` of the linear static solution of
``p0``:

.. math::

    ([K_{C0}] + \lambda ([K_G] + [K_{Cfollower}]))\{\phi\} = \{0\}

as in the validation tests of rings and cylinders, see
:doc:`ex_follower_pressure`. Under a dead pressure `[K_{Cfollower}]` is left
out. With the Sanders-type kinematics and a dead pressure, a rigid rotation
`v` = constant of a ring left free gives a spurious buckling mode at a
vanishing load, which the follower load stiffness cancels, see the Notes of
:mod:`bfsccylinder.bfsccylinder_sanders`.


Unsymmetric load stiffness matrix
---------------------------------

`[K_{Cfollower}]` is unsymmetric in general, a non-conservative load. The
assembled matrix restricted to the unknown degrees of freedom is symmetric
when the pressure is uniform over the loaded region and, on each edge of
the region, `w` or the displacement normal to the edge is constrained, e.g.
a cylinder closed around the circumference with simply supported or
clamped ends; free or partially restrained edges, a pressure on part of the
shell or a varying pressure give an unsymmetric matrix, and so does the
matrix of one element. The conditions are given in the Notes of
:mod:`bfsccylinder.bfsccylinder`. The matrix is written in full and must not
be symmetrized: the linear systems and the eigenvalue problems are solved
with the solvers of unsymmetric matrices, e.g. ``structsolve.lb`` with
``symmetric=False`` in the validation tests.
