Analyses
========

.. currentmodule:: bfsccylinder.bfsccylinder

The element gives the matrices and vectors of :doc:`matrices`; the global
systems are assembled as in :doc:`assembling` and solved with SciPy or with
structsolve. This chapter derives the systems of the four analyses from the
equilibrium of the shell, for either kinematics, and states what each
assumes.


Equilibrium
-----------

With the global displacement vector `\boldsymbol{u}`, the internal forces
`\boldsymbol{F}_{int}(\boldsymbol{u})` of :func:`.update_fint`, the gradient
of the strain energy, and the external forces `\lambda
\boldsymbol{F}_{ext}` of a reference load scaled by the load multiplier
`\lambda`, the principle of virtual work gives the residual

.. math::
    :label: an-residual

    \boldsymbol{R}(\boldsymbol{u}, \lambda) =
    \boldsymbol{F}_{int}(\boldsymbol{u}) - \lambda \boldsymbol{F}_{ext} =
    \boldsymbol{0}

over the unknown degrees of freedom, with the Jacobian
`\boldsymbol{K}_T(\boldsymbol{u})` of Eq. :eq:`mat-KT`. For a follower
pressure, `\boldsymbol{F}_{ext}` depends on `\boldsymbol{u}` and the
Jacobian has the load stiffness matrix as well, Eq. :eq:`fp-residual`. The
analyses below are this equation, its linearisation, or its stability.


Linear static analysis
----------------------

For small displacements, `\boldsymbol{F}_{int} = \boldsymbol{K}_{C0}
\boldsymbol{u}` and

.. math::
    :label: an-linear-static

    \boldsymbol{K}_{C0} \boldsymbol{u} = \boldsymbol{F}_{ext}

with `\boldsymbol{K}_{C0}` of :func:`.update_KC0`, symmetric and positive
definite once the rigid-body motions are constrained. The degrees of
freedom are partitioned into the unknown ones, `u`, and the constrained
ones, `k`, with the known values `\bar{\boldsymbol{u}}_k`, zero for a
support and nonzero for a prescribed displacement:

.. math::
    :label: an-partition

    \boldsymbol{K}_{uu} \boldsymbol{u}_u = \boldsymbol{F}_u -
    \boldsymbol{K}_{uk} \bar{\boldsymbol{u}}_k, \qquad
    \boldsymbol{F}_k = \boldsymbol{K}_{uk}^\top \boldsymbol{u}_u +
    \boldsymbol{K}_{kk} \bar{\boldsymbol{u}}_k

with `\boldsymbol{F}_k` the reaction forces. A prescribed displacement,
e.g. the axial shortening of one end, is the load of the analysis, and the
sum of its reactions is the corresponding force. A pressure enters through
the force vector of :func:`.update_fext_follower` at `\boldsymbol{u} =
\boldsymbol{0}`, the consistent load vector of a dead pressure.


Linear buckling analysis
------------------------

A load `\lambda \boldsymbol{F}_{ext}` produces, in the linear range, the
prebuckling state `\lambda \boldsymbol{u}_0`, with `\boldsymbol{u}_0` the
linear static solution of the reference load. The equilibrium becomes
neutral, and the shell buckles, when the tangent stiffness matrix of this
state is singular. Since `\boldsymbol{K}_G` is homogeneous of degree one in
the displacements, `\boldsymbol{K}_G(\lambda \boldsymbol{u}_0) = \lambda
\boldsymbol{K}_G(\boldsymbol{u}_0)`, and neglecting the terms of the
prebuckling rotations in `\boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}`,
which vanish for a membrane prebuckling state, the tangent stiffness matrix
is `\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G(\boldsymbol{u}_0)` and

.. math::
    :label: an-linear-buckling

    \left( \boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G \right)
    \boldsymbol{\phi} = \boldsymbol{0}

The eigenvalues `\lambda` are the load multipliers of the reference load,
the critical load being the one of smallest magnitude, and the
eigenvectors `\boldsymbol{\phi}` the buckling modes. `\boldsymbol{K}_G` is
computed in one of two ways:

- :func:`.update_KG` from `\boldsymbol{u}_0`, with the resultants of the
  linear strains at each integration point, Eq. :eq:`mat-KG`. They are the
  resultants of the shell with its boundary conditions, including the
  bending boundary layers near the supports, the hoop resultants that a
  restrained Poisson expansion creates, and the effect of an unbalanced
  laminate. For a prescribed displacement, `\lambda` multiplies the sum of
  its reactions;
- :func:`.update_KG_constant_stress` from given membrane resultants
  `N_{xx}`, `N_{yy}` and `N_{xy}`, without a static analysis. `\lambda`
  multiplies these resultants, e.g. `N_{xx} = -1` for an axial compression
  of total force `-2 \pi R` along the circumference. The prebuckling state
  is then assumed to be a membrane state, and the boundary conditions enter
  the problem only through `\boldsymbol{K}_{C0}`.

`\boldsymbol{K}_{C0}` and `\boldsymbol{K}_G` are both symmetric, and the
problem is a generalised symmetric eigenvalue problem, e.g. solved for the
lowest load multipliers as `\boldsymbol{K}_{C0} \boldsymbol{\phi} = \lambda
(-\boldsymbol{K}_G) \boldsymbol{\phi}`, or with ``structsolve.lb``, which
solves Eq. :eq:`an-linear-buckling`. Under a follower pressure the load
stiffness matrix of the reference pressure is added to
`\boldsymbol{K}_G`, Eq. :eq:`fp-buckling`, and the problem is unsymmetric
in general. The two matrices must come from the same kinematics, see
:doc:`kinematics`. Linear buckling neglects the nonlinearity of the
prebuckling path, which the nonlinear static analysis below takes into
account, and the imperfections of the shell; the element describes a
perfect cylinder.


Natural frequencies
-------------------

The kinetic energy of Eq. :eq:`mat-T` and the strain energy linearised
about the undeformed state give the equations of free vibration, `\boldsymbol{M}
\ddot{\boldsymbol{u}} + \boldsymbol{K}_{C0} \boldsymbol{u} = \boldsymbol{0}`,
whose harmonic solutions `\boldsymbol{u} = \boldsymbol{\phi} \sin(\omega_n
t)` solve

.. math::
    :label: an-frequencies

    \left( \boldsymbol{K}_{C0} - \omega_n^2 \boldsymbol{M} \right)
    \boldsymbol{\phi} = \boldsymbol{0}

with the consistent mass matrix `\boldsymbol{M}` of :func:`.update_M`, the
natural angular frequencies `\omega_n`, in radians per unit time, and the
modes `\boldsymbol{\phi}`. Both matrices are symmetric, `\boldsymbol{M}`
positive definite. The lowest frequencies of a cylinder are clustered,
close to each other, and those of a closed cylinder come in pairs of equal
frequencies, the modes `\cos(n\theta)` and `\sin(n\theta)` of the same
number `n` of circumferential waves. A solver of the lowest eigenpairs
must therefore be verified for missing eigenvalues, e.g. with a
shift-invert iteration about zero and the check of the residual of each
eigenpair. Since the lowest modes may have few waves, the Sanders-type
kinematics are the safer choice for them, see :doc:`kinematics`.


Nonlinear static analysis
-------------------------

For large displacements, Eq. :eq:`an-residual` is solved for
`\boldsymbol{u}` with the nonlinear strains of the kinematics, e.g. with the
Newton-Raphson method under load control:

.. math::
    :label: an-newton

    \boldsymbol{K}_T(\boldsymbol{u}^{(i)}) \Delta \boldsymbol{u} =
    -\boldsymbol{R}(\boldsymbol{u}^{(i)}, \lambda), \qquad
    \boldsymbol{u}^{(i+1)} = \boldsymbol{u}^{(i)} + \Delta \boldsymbol{u}

with the load multiplier `\lambda` increased in steps, `\lambda = 1` being
the full load, and

.. math::
    :label: an-tangent

    \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}
    (\boldsymbol{u}) + \boldsymbol{K}_{GNL}(\boldsymbol{u}) +
    \boldsymbol{K}_G(\boldsymbol{u})

of :func:`.update_KC0`, :func:`.update_KCNL` and :func:`.update_KG`, all
computed with the current displacements and with the same integration
points as `\boldsymbol{F}_{int}`. `\boldsymbol{K}_T` is the exact Jacobian of
`\boldsymbol{F}_{int}`, such that the iterations converge quadratically
near the solution. A follower pressure adds its force vector to the
external forces and its load stiffness matrix to `\boldsymbol{K}_T`, Eq.
:eq:`fp-residual`. Past a limit point of the load, load control fails, and
an arc-length method, in which `\lambda` is an unknown constrained together
with `\boldsymbol{u}` [Crisfield1981]_, follows the equilibrium path.

The degrees of freedom mix displacements, their first derivatives and
`w_{,xy}`, of different dimensions. A convergence criterion on the
Euclidean norm of `\boldsymbol{R}` depends on the units. Scaling each
component of the residual and of the reference forces with the diagonal
entry of a stiffness matrix, as in the convergence test of Crisfield
[Crisfield1981]_, removes this dependence; over the free degrees of freedom
it also leaves out the reactions at the constrained ones.


Solving with structsolve
------------------------

The solvers of `structsolve <https://saullocastro.github.io/structsolve/>`_
receive the matrices and vectors of the model, assembled by the user; it is
not a requirement of bfsccylinder, see :doc:`install`. The linear buckling
problem is solved by ``structsolve.lb``, with ``symmetric=False`` when it
contains the load stiffness matrix of a follower pressure.

The nonlinear static analysis is solved by ``structsolve.Analysis``, with
four callables that loop over the elements, call the functions of either
kinematics and assemble:

- ``calc_fext(inc=1., silent=False)``, the external forces
  `\boldsymbol{F}_{ext}` times the load factor ``inc``;
- ``calc_fint(c, silent=False)``, the internal forces of
  :func:`.update_fint` at the displacements ``c``;
- ``calc_kC(c=None, NLgeom=False, silent=False)``, `\boldsymbol{K}_{C0}`
  with ``NLgeom=False``, and `\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}
  (\boldsymbol{c}) + \boldsymbol{K}_{GNL}(\boldsymbol{c})` with
  ``NLgeom=True``;
- ``calc_kG(c=None, NLgeom=False, silent=False)``,
  `\boldsymbol{K}_G(\boldsymbol{c})` of :func:`.update_KG`;

such that ``calc_kC + calc_kG`` is `\boldsymbol{K}_T` of Eq.
:eq:`an-tangent` and the residual of structsolve is Eq.
:eq:`an-residual`, the convention of structsolve 0.9.3 or newer. The value
arrays of the sparse matrices are zeroed in each call, since the functions
add to them. ``Analysis.static`` increases the load factor from 0 to 1, the
first increment of size ``Analysis.initialInc`` and the next ones growing up
to ``Analysis.maxInc``, and returns the load factors and the solutions of
the converged increments. ``Analysis.NL_method`` selects the
Newton-Raphson method, ``'NR'``, or an arc-length method, and the
convergence criterion is Crisfield's by default, with the tolerance
``Analysis.relTOL = 1e-9``; ``Analysis.convergence = 'relative'`` selects
the Euclidean norm instead.

structsolve imposes no boundary condition itself. The constrained degrees
of freedom are removed either by the partition of Eq. :eq:`an-partition`,
the callables receiving and returning the vectors and matrices of the
unknown degrees of freedom, or with null rows and columns: the callables
return the vectors and matrices of the whole model with the rows and
columns of the constrained degrees of freedom zeroed, which the solvers
remove.

A follower pressure `\lambda p_0` enters through the keyword argument
``inc``, the load factor, which structsolve passes to the callables that
declare it: the external forces are the dead part
`\boldsymbol{F}_p(\boldsymbol{0})` of `p_0`, ``calc_fint`` subtracts the
part that depends on the displacements, `\lambda (\boldsymbol{F}_p
(\boldsymbol{u}) - \boldsymbol{F}_p(\boldsymbol{0}))` of
:func:`.update_fext_follower` with ``reference=0``, and ``calc_kC`` adds the
load stiffness matrix of `\lambda p_0`. The tangent stiffness matrix is then
unsymmetric in general, which the solvers support with LU factorisations.
For the arc-length methods, ``calc_fext`` also declares the configuration
``c`` and returns the follower force vector
`\boldsymbol{F}_p(\boldsymbol{c})` of `p_0`, minus the derivative of the
residual with respect to the load factor.


References
----------

.. [Crisfield1981] Crisfield, M. A., A fast incremental/iterative solution
   procedure that handles "snap-through", Computers & Structures, 13(1-3),
   55-62, 1981, doi:10.1016/0045-7949(81)90108-5.
