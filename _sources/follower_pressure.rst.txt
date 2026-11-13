Follower pressure
=================

.. currentmodule:: bfsccylinder.bfsccylinder

A gas or a liquid presses normal to the current surface of the shell and on
its current area: the pressure follows the deformation. A dead pressure, in
contrast, keeps the direction of the undeformed normal and the undeformed
area. The difference matters for the buckling of rings and cylinders under
external pressure, for which the buckling pressure of a follower load and
of a dead load differ, and for any nonlinear analysis with pressure. This
chapter derives the force vector of a follower pressure, its load stiffness
matrix and the conditions under which this matrix is symmetric. Two
functions of each kinematics implement them, :func:`.update_fext_follower`
and :func:`.update_KCfollower`.


Virtual work of the pressure
----------------------------

The pressure `p(x, y)` is given per unit area of the undeformed
mid-surface, as a function of the undeformed coordinates, a body-attached
follower load in the classification of Schweizerhof and Ramm
[SchweizerhofRamm1984]_. With `\boldsymbol{X} = \boldsymbol{X}_0 +
\boldsymbol{u}` the position of the deformed mid-surface, the pressure does
the virtual work

.. math::
    :label: fp-virtual-work

    \delta W_p = \int_A p \, \boldsymbol{a}^\top \delta \boldsymbol{u} \, dA,
    \qquad \boldsymbol{a} = \boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y}

where `\boldsymbol{a}` is the area vector of the deformed mid-surface per
unit undeformed area and `\delta \boldsymbol{u} = \{\delta u, \delta v,
\delta w\}^\top`. With the turning basis of Eq. :eq:`kin-basis`, the tangent
vectors in the basis of the undeformed mid-surface are

.. math::
    :label: fp-tangents

    \boldsymbol{X}_{,x} = \begin{Bmatrix} 1 + u_{,x} \\ v_{,x} \\ w_{,x}
    \end{Bmatrix}, \qquad
    \boldsymbol{X}_{,y} = \begin{Bmatrix} u_{,y} \\ 1 + v_{,y} + w/R \\
    w_{,y} - k_v v \end{Bmatrix}

with `k_v` of Eq. :eq:`kin-rotations`: the Sanders-type kinematics,
`k_v = 1/R`, give the exact area vector, and the Donnell-type kinematics
neglect the term `v/R` of the rotation of the normal, as their strains do.
A positive `p` pushes along `+\boldsymbol{e}_z`, outwards, the direction of
a positive `w`: an external pressure is negative.


Truncation of the area vector
-----------------------------

The area vector is split into its parts of order 0, 1 and 2 in the
displacement gradients, `\boldsymbol{a} = \boldsymbol{a}_0 +
\boldsymbol{a}_1 + \boldsymbol{a}_2`:

.. math::
    :label: fp-area-vector

    \boldsymbol{a}_0 = \begin{Bmatrix} 0 \\ 0 \\ 1 \end{Bmatrix}, \qquad
    \boldsymbol{a}_1 = \begin{Bmatrix} -w_{,x} \\ -w_{,y} + k_v v \\
    u_{,x} + v_{,y} + w/R \end{Bmatrix}, \qquad
    \boldsymbol{a}_2 = \begin{Bmatrix} v_{,x} (w_{,y} - k_v v) - w_{,x}
    (v_{,y} + w/R) \\ w_{,x} u_{,y} - u_{,x} (w_{,y} - k_v v) \\ u_{,x}
    (v_{,y} + w/R) - v_{,x} u_{,y} \end{Bmatrix}

`\boldsymbol{a}_0` is the dead load; `\boldsymbol{a}_1` has the rotations
of the normal, `\beta_x` and `\beta_y`, in its in-plane components and the
change of area, the linear membrane dilatation `\varepsilon_{xx} +
\varepsilon_{yy}`, in its normal component. With the ordering of the
moderate rotations of the strains, `u_{,x} \sim \epsilon` and `w_{,x} \sim
\epsilon^{1/2}`, `\boldsymbol{a}_0 + \boldsymbol{a}_1` keeps every term of
`p \, \boldsymbol{a}^\top \delta \boldsymbol{u}` up to the order `\epsilon p
\, \delta w` of the strains, and `\boldsymbol{a}_2` adds terms of order
`\epsilon^2 p \, \delta w`, of the order of those that the strains neglect.

The functions take two flags. ``quadratic=0``, the default, uses the
consistent truncation `\boldsymbol{a}_0 + \boldsymbol{a}_1`, and
``quadratic=1`` adds `\boldsymbol{a}_2`, the complete area vector of the
kinematics, e.g. to check that a result does not depend on the truncation.
``reference=0`` leaves out `\boldsymbol{a}_0`. With the interpolation of
:doc:`element`, the force vector is

.. math::
    :label: fp-force

    \boldsymbol{F}_p(\boldsymbol{u}_e) = \int_A p \left(
    (\boldsymbol{S}^u)^\top a_x + (\boldsymbol{S}^v)^\top a_y +
    (\boldsymbol{S}^w)^\top a_z \right) dA, \qquad
    \boldsymbol{a} = r \boldsymbol{a}_0 + \boldsymbol{a}_1 + q
    \boldsymbol{a}_2

with `r` = ``reference`` and `q` = ``quadratic``. At `\boldsymbol{u}_e =
\boldsymbol{0}` it is the consistent load vector of a dead pressure, `\int_A
p (\boldsymbol{S}^w)^\top dA`, for both truncations; with ``reference=0``
it is `\boldsymbol{F}_p(\boldsymbol{u}_e) -
\boldsymbol{F}_p(\boldsymbol{0})`, the part that depends on the
displacements. For ``quadratic=0``, `\boldsymbol{F}_p` is affine in the
displacements.

The pressure is given to the functions at the integration points of each
element, as an array ``p`` of shape ``(nint, nint)``, ``p[i, j]`` at `\xi` =
``points[i]`` and `\eta` = ``points[j]``, like the stiffnesses of
:doc:`constitutive`, multiplied by the load factor of the analysis. A
pressure that depends on the current position, e.g. a hydrostatic head of a
liquid, is not modelled.


Load stiffness matrix
---------------------

The load stiffness matrix of :func:`.update_KCfollower` is the derivative
of the force vector, with the sign of a stiffness:

.. math::
    :label: fp-KF

    \boldsymbol{K}_F = -\frac{\partial \boldsymbol{F}_p}{\partial
    \boldsymbol{u}_e} = -\int_A p \left( (\boldsymbol{S}^u)^\top
    \frac{\partial a_x}{\partial \boldsymbol{u}_e} + (\boldsymbol{S}^v)^\top
    \frac{\partial a_y}{\partial \boldsymbol{u}_e} + (\boldsymbol{S}^w)^\top
    \frac{\partial a_z}{\partial \boldsymbol{u}_e} \right) dA

with the derivatives of the displacement gradients, e.g. `\partial
u_{,x}/\partial \boldsymbol{u}_e = \boldsymbol{S}^u_{,x}`. `\boldsymbol{a}_0`
does not contribute. For ``quadratic=0`` the matrix is constant for a given
pressure,

.. math::
    :label: fp-KF-linear

    \boldsymbol{K}_F = \int_A p \left( (\boldsymbol{S}^u)^\top
    \boldsymbol{S}^w_{,x} + (\boldsymbol{S}^v)^\top \left(
    \boldsymbol{S}^w_{,y} - k_v \boldsymbol{S}^v \right) -
    (\boldsymbol{S}^w)^\top \left( \boldsymbol{S}^u_{,x} +
    \boldsymbol{S}^v_{,y} + \boldsymbol{S}^w/R \right) \right) dA

and for ``quadratic=1`` it is linear in the displacements. The residual of
a nonlinear analysis with the follower pressure and its Jacobian are

.. math::
    :label: fp-residual

    \boldsymbol{R}(\boldsymbol{u}) = \boldsymbol{F}_{int}(\boldsymbol{u}) -
    \boldsymbol{F}_p(\boldsymbol{u}), \qquad
    \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} +
    \boldsymbol{K}_{GNL} + \boldsymbol{K}_G + \boldsymbol{K}_F

with the matrices of Eq. :eq:`mat-KT` and `\boldsymbol{F}_p` and
`\boldsymbol{K}_F` of the same pressure and truncation.


Symmetry of the load stiffness matrix
-------------------------------------

`\boldsymbol{K}_F` is unsymmetric in general: the follower pressure is a
non-conservative load. For the first-order area vector, the skew part of
the bilinear form `P(\delta \boldsymbol{u}, \Delta \boldsymbol{u}) = \int_A
p \, \delta \boldsymbol{u}^\top \boldsymbol{a}_1(\Delta \boldsymbol{u}) \,
dA` over the loaded region is, after an integration by parts,

.. math::
    :label: fp-skew

    \begin{aligned}
    P(\delta \boldsymbol{u}, \Delta \boldsymbol{u}) - P(\Delta \boldsymbol{u},
    \delta \boldsymbol{u}) = &-\oint p \left( (\delta u \Delta w - \delta w
    \Delta u) n_x + (\delta v \Delta w - \delta w \Delta v) n_y \right) ds
    \\ &+ \int_A \left( p_{,x} (\delta u \Delta w - \delta w \Delta u) +
    p_{,y} (\delta v \Delta w - \delta w \Delta v) \right) dA
    \end{aligned}

with `\{n_x, n_y\}^\top` the outward normal of the contour of the loaded
region. The terms of the curvature of the cylinder, `w/R` and `k_v v`, are
symmetric and cancel, such that Eq. :eq:`fp-skew` holds for both
kinematics. The load stiffness matrix is therefore symmetric, and the
follower pressure conservative, if and only if:

- `p` is uniform over the loaded region, such that the area integral
  vanishes; and
- at each point of the contour of the loaded region, either `w = 0` or the
  in-plane displacement normal to the contour vanishes, `u = 0` on an edge
  `x` = constant and `v = 0` on an edge `y` = constant, such that the
  contour integral vanishes.

Simply supported and clamped edges, symmetry planes and closed surfaces
meet these conditions; for a cylinder closed around the circumference, the
contour of the loaded region is that of its ends. The load potential is
then the pressure times the change of the enclosed volume. Free or
partially restrained edges, a pressure applied on a patch whose edges lie
inside the shell, and a pressure that varies over the surface give an
unsymmetric matrix. These are the results of Hibbitt [Hibbitt1979]_ and of
Schweizerhof and Ramm [SchweizerhofRamm1984]_, who showed that the pressure
load stiffness is symmetric for closed surfaces or loaded surfaces with a
fixed boundary.

For the matrix of one element, the contour of the loaded region is the
boundary of the element, where the second condition does not hold: the
element matrix is unsymmetric. In the assembled matrix, the contour
integrals along an edge shared by two elements cancel when `p` is
continuous, since `u`, `v` and `w` are continuous, and the conditions apply
to the contour of the loaded region of the whole shell, restricted to the
unknown degrees of freedom. The matrices are written in full and must not
be symmetrized; the linear systems and eigenvalue problems that contain
`\boldsymbol{K}_F` are solved with solvers of unsymmetric matrices. For
``quadratic=1`` the matrix depends on the displacements and these
conditions do not apply.


Linear buckling under a follower pressure
-----------------------------------------

For a pressure `\lambda p_0` proportional to the load multiplier, the
tangent stiffness matrix of the prebuckling state `\lambda
\boldsymbol{u}_0`, linearised in `\lambda`, adds the load stiffness matrix
`\boldsymbol{K}_{F_{ref}}` of `p_0` to the geometric stiffness matrix of the
linear static solution `\boldsymbol{u}_0` of `p_0`:

.. math::
    :label: fp-buckling

    \left( \boldsymbol{K}_{C0} + \lambda \left( \boldsymbol{K}_G
    (\boldsymbol{u}_0) + \boldsymbol{K}_{F_{ref}} \right) \right)
    \boldsymbol{\phi} = \boldsymbol{0}

with ``quadratic=0``, an eigenvalue problem that is unsymmetric in general,
see :doc:`analyses`. For a dead pressure, `\boldsymbol{K}_{F_{ref}}` is left
out. The two loads give different buckling pressures: for a ring in
generalised plane strain buckling with `n = 2` circumferential waves, the
inextensional limit of the exact second variation of the ring with the
strains of :doc:`kinematics` gives

.. list-table::
    :header-rows: 1

    * - Kinematics
      - Load
      - `p R^3/D`
    * - Sanders-type
      - follower
      - `-3`
    * - Sanders-type
      - dead
      - `-4`
    * - Donnell-type
      - follower
      - `-16/5`

with `D` the bending stiffness of the wall. With the Sanders-type
kinematics and a dead pressure, a ring free to rotate has a spurious
buckling mode at a vanishing load: the rigid rotation `v` = constant under
the hoop prestress `N_{yy} = pR` has the energy `\int_A N_{yy} (v/R)^2 dA`
in `\boldsymbol{K}_G`, negative under an external pressure, which the
follower load stiffness, `-\int_A p v^2/R \, dA`, cancels. The rigid
rotation must then be constrained, or the follower load used.


References
----------

.. [Hibbitt1979] Hibbitt, H. D., Some follower forces and load stiffness,
   International Journal for Numerical Methods in Engineering, 14(6),
   937-941, 1979, doi:10.1002/nme.1620140613.

.. [SchweizerhofRamm1984] Schweizerhof, K., and Ramm, E., Displacement
   dependent pressure loads in nonlinear finite element analyses, Computers
   & Structures, 18(6), 1099-1114, 1984,
   doi:10.1016/0045-7949(84)90154-8.
