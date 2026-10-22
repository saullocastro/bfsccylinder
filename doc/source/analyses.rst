Analyses
========

.. currentmodule:: bfsccylinder.bfsccylinder

The matrices and vectors of :doc:`assembling`, restricted to the unknown
degrees of freedom ``bu``, give the systems of the analyses below, solved
with SciPy as in the tests. The functions are those of both kinematics, see
:doc:`kinematics`, and the theory of each matrix is in the Notes of its
function.


Linear static analysis
----------------------

.. math::

    [K_{C0}]\{u\} = \{f\}

with `[K_{C0}]` of :func:`.update_KC0` and the force vector or the
prescribed displacements of :doc:`assembling`. The tests solve it with
``scipy.sparse.linalg.spsolve`` or with the conjugate gradient method of
``scipy.sparse.linalg.cg``, with the matrix and the right-hand side scaled
by a scalar, see :doc:`ex_linear_static`.


Linear buckling analysis
------------------------

.. math::

    ([K_{C0}] + \lambda [K_G])\{\phi\} = \{0\}

The critical loads are `\lambda` times the load that gives `[K_G]`, and
`\{\phi\}` are the buckling modes. `[K_G]` is computed in one of two ways:

- :func:`.update_KG`, from the displacements `\{u_0\}` of the linear static
  solution of a reference load or prescribed displacement, such that the
  membrane stress resultants are those of the shell with its boundary
  conditions. Since `[K_G]` is linear in `\{u_0\}`, `\lambda` multiplies
  the reference load, e.g. the sum of the reaction forces of a prescribed
  shortening;
- :func:`.update_KG_constant_stress`, from the membrane stress resultants
  ``Nxx``, ``Nyy`` and ``Nxy`` given for each element, the last arguments
  of the function, without a static analysis: `\lambda` multiplies these
  resultants, e.g. ``Nxx = -1`` for an axial compression, of total force
  `-2 \pi R` along the circumference.

The tests compute the lowest load multipliers with
``scipy.sparse.linalg.lobpcg``, for the problem `[K_{C0}]\{\phi\} =
\lambda (-[K_G])\{\phi\}`, with an incomplete LU factorization of
`[K_{C0}]`, ``spilu``, as preconditioner, see :doc:`ex_linear_buckling`.
The matrices can also be given to the linear buckling solver of
`structsolve <https://github.com/saullocastro/structsolve>`_,
``structsolve.lb``, as in the validation tests of the follower pressure, see
:doc:`ex_follower_pressure`.


Natural frequencies
-------------------

.. math::

    ([K_{C0}] - \omega_n^2 [M])\{\phi\} = \{0\}

with the mass matrix `[M]` of :func:`.update_M`, which needs ``lex``,
``ley``, ``h`` and ``rho`` of each element, and ``R`` for the Sanders-type
kinematics, and no integration points, and `\omega_n` the natural angular
frequencies, in radians per unit time. The lowest frequencies of a cylinder
are clustered; the tests use the function ``lowest_frequencies`` of
``tests/natural_frequency_solver.py``, which solves the shift-inverted
problem with ARPACK from a fixed starting vector and verifies the residual
of each eigenpair and that no eigenvalue is missing, see
:doc:`ex_natural_frequency`.


Nonlinear static analysis
-------------------------

For large displacements, with the nonlinear strains of the kinematics, the
equilibrium of the internal forces of :func:`.update_fint` with the
external forces is solved for the displacements, e.g. with the
Newton-Raphson method:

.. math::

    \{R(u)\} = \{f_{int}(u)\} - \lambda \{f\}, \qquad
    [K_T(u)] = [K_{C0}] + [K_{CNL}(u)] + [K_G(u)], \qquad
    [K_T]\{\Delta u\} = -\{R\}

with the load factor `\lambda` increased in steps and the tangent stiffness
matrix `[K_T]` of :func:`.update_KC0`, :func:`.update_KCNL` and
:func:`.update_KG`, all computed with the current displacements and the
same integration points as `\{f_{int}\}`. `[K_T]` is the exact Jacobian of
`\{f_{int}\}`, such that the Newton-Raphson iterations converge
quadratically; ``tests/test_tangent_consistency.py`` checks it for both
kinematics with Taylor tests and finite differences. `[K_G]` keeps only
the part linear in the displacements, the one of the linear buckling
analysis, and the rest of the geometric terms is in `[K_{CNL}]`, see the
Notes of :func:`.update_KCNL`.

The value arrays of `[K_{CNL}]` and `[K_G]` are zeroed before each
assembly, since the functions add to them, see :doc:`assembling`. The test
of :doc:`ex_nonlinear_static` writes the iterations explicitly, with the
convergence criterion of the scaled norms of the residual and of the
forces. A follower pressure adds its force vector to the residual and its
load stiffness matrix to `[K_T]`, see :doc:`follower_pressure`.
