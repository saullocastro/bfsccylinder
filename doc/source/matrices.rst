Element matrices and internal forces
====================================

.. currentmodule:: bfsccylinder.bfsccylinder

Every matrix and vector of the element follows from two energies, the
strain energy of the kinematics of :doc:`kinematics` with the constitutive
relation of :doc:`constitutive`, and the kinetic energy, discretised with
the interpolation of :doc:`element`. The functions named below are those of
:mod:`bfsccylinder.bfsccylinder`; :mod:`bfsccylinder.bfsccylinder_sanders`
has the same functions, with the operators of the Sanders-type kinematics.


Strain energy
-------------

With the membrane strains `\boldsymbol{\varepsilon} =
\boldsymbol{\varepsilon}_L + \boldsymbol{\varepsilon}_{NL}` and the
curvatures `\boldsymbol{\kappa}`, the strain energy of the element is

.. math::
    :label: mat-U

    U(\boldsymbol{u}_e) = \frac{1}{2} \int_A \left(
    \boldsymbol{\varepsilon}^\top \boldsymbol{A} \boldsymbol{\varepsilon} +
    2 \boldsymbol{\varepsilon}^\top \boldsymbol{B} \boldsymbol{\kappa} +
    \boldsymbol{\kappa}^\top \boldsymbol{D} \boldsymbol{\kappa} \right) dA

with, from Eqs. :eq:`el-Bm` to :eq:`el-BNL`,

.. math::
    :label: mat-strains

    \boldsymbol{\varepsilon} = \left( \boldsymbol{B}_m + \frac{1}{2}
    \boldsymbol{B}_{NL} \right) \boldsymbol{u}_e, \qquad
    \delta \boldsymbol{\varepsilon} = \left( \boldsymbol{B}_m +
    \boldsymbol{B}_{NL} \right) \delta \boldsymbol{u}_e, \qquad
    \boldsymbol{\kappa} = \boldsymbol{B}_b \boldsymbol{u}_e

The factor `1/2` in the strains and its absence in their variation come
from `\boldsymbol{\varepsilon}_{NL}` being quadratic in the displacements.


Internal force vector
---------------------

The internal force vector of :func:`.update_fint` is the gradient of the
strain energy:

.. math::
    :label: mat-fint

    \boldsymbol{F}_{int} = \frac{\partial U}{\partial \boldsymbol{u}_e} =
    \int_A \left( \left( \boldsymbol{B}_m + \boldsymbol{B}_{NL}
    \right)^\top \boldsymbol{N} + \boldsymbol{B}_b^\top \boldsymbol{M}
    \right) dA, \qquad
    \begin{Bmatrix} \boldsymbol{N} \\ \boldsymbol{M} \end{Bmatrix} =
    \begin{bmatrix} \boldsymbol{A} & \boldsymbol{B} \\ \boldsymbol{B} &
    \boldsymbol{D} \end{bmatrix} \begin{Bmatrix} \boldsymbol{\varepsilon} \\
    \boldsymbol{\kappa} \end{Bmatrix}

with the resultants of the current displacements. At equilibrium,
`\boldsymbol{F}_{int}` balances the external forces, see :doc:`analyses`.


Tangent stiffness matrix
------------------------

The Hessian of the strain energy, the Jacobian of
`\boldsymbol{F}_{int}`, is split into four matrices:

.. math::
    :label: mat-KT

    \boldsymbol{K}_T = \frac{\partial \boldsymbol{F}_{int}}{\partial
    \boldsymbol{u}_e} = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}
    (\boldsymbol{u}_e) + \boldsymbol{K}_{GNL}(\boldsymbol{u}_e) +
    \boldsymbol{K}_G(\boldsymbol{u}_e)

computed by :func:`.update_KC0` (`\boldsymbol{K}_{C0}`),
:func:`.update_KCNL` (`\boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}`) and
:func:`.update_KG` (`\boldsymbol{K}_G`). Since the three functions
differentiate the same quadrature sums as :func:`.update_fint`,
`\boldsymbol{K}_T` is the exact Jacobian of the discrete internal forces for
any number of integration points, such that the Newton-Raphson iterations
of :doc:`analyses` converge quadratically.

**Linear constitutive stiffness matrix.** The term of the linear strains,

.. math::
    :label: mat-KC0

    \boldsymbol{K}_{C0} = \int_A \left( \boldsymbol{B}_m^\top \boldsymbol{A}
    \boldsymbol{B}_m + \boldsymbol{B}_m^\top \boldsymbol{B} \boldsymbol{B}_b
    + \boldsymbol{B}_b^\top \boldsymbol{B} \boldsymbol{B}_m +
    \boldsymbol{B}_b^\top \boldsymbol{D} \boldsymbol{B}_b \right) dA

does not depend on the displacements: it is the stiffness matrix of a
linear static analysis and the tangent stiffness matrix at
`\boldsymbol{u}_e = \boldsymbol{0}`. It is symmetric. With the Sanders-type
kinematics, `\boldsymbol{B}_b` has terms of `u` and `v`, such that the
bending stiffness `\boldsymbol{D}` couples the in-plane and the normal
displacements even for a homogeneous wall, `\boldsymbol{B} = \boldsymbol{0}`.

**Nonlinear constitutive stiffness matrix.** The terms of
`\boldsymbol{B}_{NL}` in the derivative of `(\boldsymbol{B}_m +
\boldsymbol{B}_{NL})^\top \boldsymbol{N}` and of `\boldsymbol{B}_b^\top
\boldsymbol{M}` with `\boldsymbol{N}` and `\boldsymbol{M}` varying are

.. math::
    :label: mat-KCNL

    \boldsymbol{K}_{CNL} = \underbrace{\int_A \left( \boldsymbol{B}_m^\top
    \boldsymbol{A} + \boldsymbol{B}_b^\top \boldsymbol{B} \right)
    \boldsymbol{B}_{NL} dA}_{\boldsymbol{K}_{C0L}} +
    \underbrace{\int_A \boldsymbol{B}_{NL}^\top \left( \boldsymbol{A}
    \boldsymbol{B}_m + \boldsymbol{B} \boldsymbol{B}_b \right)
    dA}_{\boldsymbol{K}_{CL0}} +
    \underbrace{\int_A \boldsymbol{B}_{NL}^\top \boldsymbol{A}
    \boldsymbol{B}_{NL} dA}_{\boldsymbol{K}_{CLL}}

`\boldsymbol{K}_{C0L}` and `\boldsymbol{K}_{CL0}` are linear in the
displacements and `\boldsymbol{K}_{CLL}` is quadratic.

**Geometric stiffness matrices.** The derivative of `\boldsymbol{B}_{NL}`
in `\boldsymbol{B}_{NL}^\top \boldsymbol{N}`, with `\boldsymbol{N}` held
fixed, gives a geometric stiffness matrix of the membrane resultants. The
resultants are split into the part of the linear strains and the part of
the nonlinear strains,

.. math::
    :label: mat-N-split

    \boldsymbol{N}_L = \boldsymbol{A} \boldsymbol{B}_m \boldsymbol{u}_e +
    \boldsymbol{B} \boldsymbol{B}_b \boldsymbol{u}_e, \qquad
    \boldsymbol{N}_{NL} = \boldsymbol{A} \boldsymbol{\varepsilon}_{NL}

and so is the geometric stiffness matrix:

.. math::
    :label: mat-KG

    \boldsymbol{K}_G = \int_A \boldsymbol{G}^\top \begin{bmatrix} N_{Lxx} &
    N_{Lxy} \\ N_{Lxy} & N_{Lyy} \end{bmatrix} \boldsymbol{G} \, dA, \qquad
    \boldsymbol{K}_{GNL} = \int_A \boldsymbol{G}^\top \begin{bmatrix}
    N_{NLxx} & N_{NLxy} \\ N_{NLxy} & N_{NLyy} \end{bmatrix} \boldsymbol{G}
    \, dA

with the rows `\boldsymbol{G}` of the rotations, Eq. :eq:`el-G`. Since
`\boldsymbol{B}_{NL}^\top \boldsymbol{N}` is, row by row, `\boldsymbol{G}^\top`
times the resultants times the slopes, its derivative at fixed
`\boldsymbol{N}` is this quadratic form. :func:`.update_KG` computes
`\boldsymbol{K}_G` and :func:`.update_KCNL` adds `\boldsymbol{K}_{GNL}` to
`\boldsymbol{K}_{CNL}`. The split keeps `\boldsymbol{K}_G` homogeneous of
degree one in the displacements, `\boldsymbol{K}_G(\lambda
\boldsymbol{u}_0) = \lambda \boldsymbol{K}_G(\boldsymbol{u}_0)`, which is
what a linear buckling analysis needs, while `\boldsymbol{K}_{CNL} +
\boldsymbol{K}_{GNL}` vanishes at zero displacements and collects the rest
of the Hessian.

The geometric stiffness matrix depends on the kinematics through
`\boldsymbol{G}`. For the Donnell-type kinematics, `\boldsymbol{G}` has
entries only at the degrees of freedom of `w`, and `\boldsymbol{K}_G` is the
geometric stiffness matrix of a flat plate: the curvature of the shell
enters a buckling problem only through `\boldsymbol{K}_{C0}` and the
resultants. For the Sanders-type kinematics, the row `\boldsymbol{S}^w_{,y}
- \boldsymbol{S}^v/R` populates the blocks `v`-`v` and `v`-`w`, multiplied
by `N_{yy}` and `N_{xy}` only; for a membrane state of axial compression,
`N_{yy} = N_{xy} = 0`, the two matrices coincide.

**Geometric stiffness matrix of a constant stress state.**
:func:`.update_KG_constant_stress` evaluates the `\boldsymbol{K}_G` of
Eq. :eq:`mat-KG` with given resultants `N_{xx}`, `N_{yy}` and `N_{xy}`,
constant over the element, instead of those of a displacement field. It is
linear in the given resultants, and is the geometric stiffness matrix of an
assumed membrane state, e.g. a uniform axial compression without the
bending boundary layers that the supports of a real cylinder produce.


Mass matrix
-----------

The kinetic energy of the displacement field of Eq.
:eq:`kin-displacement-field`, for a density `\rho` = ``rho`` uniform over
the thickness `h` = ``h``, with the shallow-shell volume `dV = dz\, dA` and
the reference surface at the middle of the thickness, is

.. math::
    :label: mat-T

    T = \frac{1}{2} \int_A \left( \rho h \left( \dot{u}^2 + \dot{v}^2 +
    \dot{w}^2 \right) + \rho \frac{h^3}{12} \left( \dot{\beta}_x^2 +
    \dot{\beta}_y^2 \right) \right) dA

the terms of `\int z \, dz = 0` vanishing. With Eqs. :eq:`el-rows` and
:eq:`el-G`, the consistent mass matrix of :func:`.update_M` is

.. math::
    :label: mat-M

    \boldsymbol{M} = \int_A \rho \left( h \left( (\boldsymbol{S}^u)^\top
    \boldsymbol{S}^u + (\boldsymbol{S}^v)^\top \boldsymbol{S}^v +
    (\boldsymbol{S}^w)^\top \boldsymbol{S}^w \right) + \frac{h^3}{12}
    \boldsymbol{G}^\top \boldsymbol{G} \right) dA

The terms in `h^3/12` are the rotary inertia of the rotations of the
normal of each kinematics: the Sanders-type rotation `\beta_y = -w_{,y} +
v/R` couples `v` and `w` and adds `\rho h^3/(12 R^2)
(\boldsymbol{S}^v)^\top \boldsymbol{S}^v`, terms that the Donnell-type
matrix does not have. The rotation used in the kinetic energy is the one of
the strains, such that the rigid translation of Eq.
:eq:`kin-rigid-translation` has the exact kinetic energy with the
Sanders-type kinematics. The integrand is a polynomial and the matrix is
computed in closed form from ``lex``, ``ley``, ``h``, ``rho`` and, for the
Sanders-type kinematics, ``R``.


Entries per element
-------------------

Each function writes the entries of one element, both triangles included,
in sparse arrays; the entries that vanish identically, e.g. the block of
the in-plane degrees of freedom of the Donnell-type `\boldsymbol{K}_{CNL}`,
are not written. The number of entries per element is a constant of each
module:

.. list-table::
    :header-rows: 1

    * - Matrix
      - Function
      - Donnell-type
      - Sanders-type
    * - `\boldsymbol{K}_{C0}`
      - :func:`.update_KC0`
      - 1600
      - 1600
    * - `\boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}`
      - :func:`.update_KCNL`
      - 1024
      - 1456
    * - `\boldsymbol{K}_G`
      - :func:`.update_KG`, :func:`.update_KG_constant_stress`
      - 256
      - 784
    * - `\boldsymbol{M}`
      - :func:`.update_M`
      - 544
      - 920
    * - `\boldsymbol{K}_F`
      - :func:`.update_KCfollower`
      - 1312
      - 1456

``KC0_SPARSE_SIZE``, ``KCNL_SPARSE_SIZE``, ``KG_SPARSE_SIZE``,
``M_SPARSE_SIZE`` and ``KCfollower_SPARSE_SIZE`` of each namespace. For the
Donnell-type element, `\boldsymbol{K}_G` has the `16 \times 16` entries of
the degrees of freedom of `w` and `\boldsymbol{K}_{CNL}` the `40 \times 40`
entries minus the `24 \times 24` of `u` and `v`; for the Sanders-type
element, `\boldsymbol{K}_G` has the `28 \times 28` entries of `v` and `w`,
and `\boldsymbol{K}_{CNL}` all entries except the `12 \times 12` of `u`.
The load stiffness matrix `\boldsymbol{K}_F` of a follower pressure is
derived in :doc:`follower_pressure`.
