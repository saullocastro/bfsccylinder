Analyses
========

.. currentmodule:: bfsccylinder.bfsccylinder

The matrices and vectors of :doc:`assembling`, restricted to the unknown
degrees of freedom ``bu``, give the systems of the analyses below, solved
with SciPy as in the tests, or with structsolve, see `Solving with
structsolve`_. The functions are those of both kinematics, see
:doc:`kinematics`, and the theory of each matrix is in the Notes of its
function.


Linear static analysis
----------------------

.. math::

    \boldsymbol{K}_{C0}\boldsymbol{u} = \boldsymbol{F}_{ext}

with `\boldsymbol{K}_{C0}` of :func:`.update_KC0` and the force vector or the
prescribed displacements of :doc:`assembling`. The tests solve it with
``scipy.sparse.linalg.spsolve`` or with the conjugate gradient method of
``scipy.sparse.linalg.cg``, with the matrix and the right-hand side scaled
by a scalar, see :doc:`ex_linear_static`.


Linear buckling analysis
------------------------

.. math::

    (\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G)\boldsymbol{\phi} = \boldsymbol{0}

The critical loads are `\lambda` times the load that gives `\boldsymbol{K}_G`, and
`\boldsymbol{\phi}` are the buckling modes. `\boldsymbol{K}_G` is computed in one of two ways:

- :func:`.update_KG`, from the displacements `\boldsymbol{u}_0` of the linear static
  solution of a reference load or prescribed displacement, such that the
  membrane stress resultants are those of the shell with its boundary
  conditions. Since `\boldsymbol{K}_G` is linear in `\boldsymbol{u}_0`, `\lambda` multiplies
  the reference load, e.g. the sum of the reaction forces of a prescribed
  shortening;
- :func:`.update_KG_constant_stress`, from the membrane stress resultants
  ``Nxx``, ``Nyy`` and ``Nxy`` given for each element, the last arguments
  of the function, without a static analysis: `\lambda` multiplies these
  resultants, e.g. ``Nxx = -1`` for an axial compression, of total force
  `-2 \pi R` along the circumference.

The tests compute the lowest load multipliers with
``scipy.sparse.linalg.lobpcg``, for the problem `\boldsymbol{K}_{C0}\boldsymbol{\phi} =
\lambda (-\boldsymbol{K}_G)\boldsymbol{\phi}`, with an incomplete LU factorization of
`\boldsymbol{K}_{C0}`, ``spilu``, as preconditioner, see :doc:`ex_linear_buckling`.
The matrices can also be given to the linear buckling solver of
`structsolve <https://saullocastro.github.io/structsolve/>`_,
``structsolve.lb``, as in the validation tests of the follower pressure, see
:doc:`ex_follower_pressure`.


Natural frequencies
-------------------

.. math::

    (\boldsymbol{K}_{C0} - \omega_n^2 \boldsymbol{M})\boldsymbol{\phi} = \boldsymbol{0}

with the mass matrix `\boldsymbol{M}` of :func:`.update_M`, which needs ``lex``,
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

    \boldsymbol{R}(\boldsymbol{u}) = \boldsymbol{F}_{int}(\boldsymbol{u}) - \lambda \boldsymbol{F}_{ext}, \qquad
    \boldsymbol{K}_T(\boldsymbol{u}) = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u}) + \boldsymbol{K}_G(\boldsymbol{u}), \qquad
    \boldsymbol{K}_T \Delta \boldsymbol{u} = -\boldsymbol{R}

with the load factor `\lambda` increased in steps and the tangent stiffness
matrix `\boldsymbol{K}_T` of :func:`.update_KC0`, :func:`.update_KCNL` and
:func:`.update_KG`, all computed with the current displacements and the
same integration points as `\boldsymbol{F}_{int}`. `\boldsymbol{K}_T` is the exact Jacobian of
`\boldsymbol{F}_{int}`, such that the Newton-Raphson iterations converge
quadratically; ``tests/test_tangent_consistency.py`` checks it for both
kinematics with Taylor tests and finite differences. `\boldsymbol{K}_G` keeps only
the part linear in the displacements, the one of the linear buckling
analysis, and the rest of the geometric terms is in `\boldsymbol{K}_{CNL}`, see the
Notes of :func:`.update_KCNL`.

The value arrays of `\boldsymbol{K}_{CNL}` and `\boldsymbol{K}_G` are zeroed before each
assembly, since the functions add to them, see :doc:`assembling`. The test
of :doc:`ex_nonlinear_static` solves the iterations with structsolve, with
the convergence criterion of the scaled norms of the residual and of the
forces, see `Solving with structsolve`_ and :doc:`ex_structsolve_nonlinear`.
A follower pressure adds its force vector to the residual and its load
stiffness matrix to `\boldsymbol{K}_T`, see :doc:`follower_pressure`.


.. _solving-with-structsolve:

Solving with structsolve
------------------------

The solvers of `structsolve <https://saullocastro.github.io/structsolve/>`_
receive the matrices and vectors of the model, assembled by the user. It is
used by the tests and the examples, and it is not a requirement of
bfsccylinder, see :doc:`install`. The linear buckling problem above is
solved by ``structsolve.lb``, as in the validation tests of the follower
pressure.

The nonlinear static analysis is solved by ``structsolve.Analysis``, with
four callables that loop over the elements, call the functions of the
element, of either kinematics, and assemble, see
:doc:`ex_structsolve_nonlinear`:

- ``calc_fext(inc=1., silent=False)``, the external forces `\boldsymbol{F}_{ext}` times
  the load factor ``inc``;
- ``calc_fint(c, silent=False)``, the internal forces of
  :func:`.update_fint` at the displacements ``c``;
- ``calc_kC(c=None, NLgeom=False, silent=False)``, `\boldsymbol{K}_{C0}` with
  ``NLgeom=False``, and `\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{c})` with ``NLgeom=True``;
- ``calc_kG(c=None, NLgeom=False, silent=False)``, `\boldsymbol{K}_G(\boldsymbol{c})` of
  :func:`.update_KG`, called with ``NLgeom=True``;

such that ``calc_kC + calc_kG`` is the tangent stiffness matrix `\boldsymbol{K}_T` and
structsolve solves the equilibrium `\boldsymbol{F}_{int}(\boldsymbol{c}) =
\lambda \boldsymbol{F}_{ext}` of the residual above. The value arrays are zeroed in each call. The
load factor goes from 0 to 1 in increments, the first one of size
``Analysis.initialInc`` and the next ones growing up to
``Analysis.maxInc``, each starting from a predictor extrapolated from the
previous increment, the linear solution of `\boldsymbol{K}_{C0}` in the first one;
``Analysis.static`` returns the load factors and the solutions of the
converged increments.

structsolve imposes no boundary condition itself. The constrained degrees
of freedom are removed in one of two ways:

- the partition of the free degrees of freedom, as in the tests: the
  callables receive ``c = u[bu]`` and return the vectors restricted to
  ``bu`` and the matrices to ``bu`` by ``bu``, and the solutions are
  expanded with ``u[bu] = c``;
- null rows and columns: the callables return the vectors and matrices of
  the whole model, with the rows and columns of the constrained degrees of
  freedom zeroed in all the matrices. The solvers remove them and return
  vectors with the size of the model, zero at these degrees of freedom.

The convergence criterion is, by default, Crisfield's: the residual and the
reference forces, the external forces and the internal forces, are scaled
with the diagonal of the linear stiffness matrix ``calc_kC()`` over the
free degrees of freedom, and the iterations stop when their ratio is below
``Analysis.relTOL = 1e-9``. The scaled norm does not depend on the units of
the degrees of freedom, which here mix displacements, their first
derivatives and `w_{,xy}`, and leaves out the reactions at the constrained
ones; ``Analysis.convergence = 'relative'`` selects the Euclidean norm
instead. With a consistent tangent the criterion is reached in a few
iterations.

A follower pressure `\lambda p`, see :doc:`follower_pressure`, enters
through the optional keyword argument ``inc``, the load factor, which
structsolve passes to the callables that declare it. The external forces
are the dead part `\boldsymbol{F}_p(\boldsymbol{0})` of ``p``, ``calc_fint`` subtracts the part
that depends on the displacements, ``reference=0``, and ``calc_kC`` adds the
load stiffness matrix of `\lambda p`, as in
``tests/test_follower_pressure_structsolve.py``, see
:doc:`ex_structsolve_nonlinear`:

.. literalinclude:: ../../tests/test_follower_pressure_structsolve.py
    :pyobject: FollowerCallables

The tangent stiffness matrix is then unsymmetric in general, which the
solvers of structsolve support with LU factorizations. For the arc-length methods, ``Analysis.NL_method``, ``calc_fext`` also
declares the configuration ``c`` and returns the follower force vector
`\boldsymbol{F}_p(\boldsymbol{c})` of ``p``, minus the derivative of the
residual `\boldsymbol{R} = \boldsymbol{F}_{int} - \lambda
\boldsymbol{F}_p(\boldsymbol{c})` with respect to the load factor.
