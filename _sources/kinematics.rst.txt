Kinematics of the cylindrical shell
===================================

.. currentmodule:: bfsccylinder.bfsccylinder

A thin cylindrical shell can be described by several strain-displacement
relations that differ in terms of the order of the thickness over the
radius, or of the wavelength of the deformation over the radius. Two of them
are implemented, with the same element, nodes, degrees of freedom,
interpolation and quadrature:

- the **Donnell-type** kinematics [Donnell1933]_, also called shallow-shell
  or Donnell-Mushtari-Vlasov kinematics, in which the curvature of the shell
  enters only the hoop membrane strain;
- the **Sanders-type** kinematics, whose linear part is the
  first-approximation theory of Sanders [Sanders1959]_ and Koiter
  [Koiter1960]_ and whose nonlinear part is that of Sanders [Sanders1963]_
  for small strains and moderate rotations.

This chapter derives both from the same displacement field, compares them
and states when each applies. The interpolation is given in :doc:`element`
and the matrices that follow from the kinematics in :doc:`matrices`.


Geometry and coordinates
------------------------

The cylinder has the radius `R` of its mid-surface and the cylindrical
coordinates `(x, \theta, r)`: `x` along the axis, `\theta` the
circumferential angle and `r` the distance from the axis. The element uses
the arc length along the circumference and the distance from the
mid-surface,

.. math::
    :label: kin-arc-length

    y = R \theta, \qquad z = r - R

with `z` positive outwards. The local basis of the mid-surface,
`\boldsymbol{e}_x`, `\boldsymbol{e}_y` and the outward normal
`\boldsymbol{e}_z`, is right-handed, `\boldsymbol{e}_x \times
\boldsymbol{e}_y = \boldsymbol{e}_z`, which fixes the sense of `\theta`, and
it turns along `y`:

.. math::
    :label: kin-basis

    \boldsymbol{e}_{y,y} = -\frac{1}{R} \boldsymbol{e}_z, \qquad
    \boldsymbol{e}_{z,y} = \frac{1}{R} \boldsymbol{e}_y

The displacements of the mid-surface are `u` along `\boldsymbol{e}_x`
(axial), `v` along `\boldsymbol{e}_y` (circumferential) and `w` along
`\boldsymbol{e}_z` (normal, positive outwards, away from the axis). The
attribute ``R`` of the element is `R`, and its lengths ``lex`` and ``ley``
are measured along `x` and `y`. Derivatives are written with a comma,
`w_{,x} = \partial w/\partial x`.


Assumptions common to both kinematics
-------------------------------------

- **Thin shell, Kirchhoff-Love hypothesis**: the normals to the
  mid-surface remain straight, normal and unstretched, such that the
  transverse shear deformation is neglected and the strains vary linearly
  through the thickness, `\boldsymbol{\varepsilon}(z) =
  \boldsymbol{\varepsilon} + z \boldsymbol{\kappa}`.
- **Shallow-shell thickness integration**: the factor `1 + z/R` of the
  metric is approximated by one, in the strains and in the volume `dV =
  (1 + z/R)\, dz\, dA`, such that the stress resultants are those of a flat
  laminate, see :doc:`constitutive`.
- **Small strains and moderate rotations**: of the nonlinear terms of the
  Green-Lagrange strains, the membrane strains keep only the squares and
  products of the rotations of the normal. With the ordering `u_{,x} \sim
  \epsilon` and `w_{,x} \sim \epsilon^{1/2}`, every term of order `\epsilon`
  is retained.
- The rotation of the material about the normal does not enter the
  nonlinear strains.

With the rotations of the normal `\beta_x`, in the plane `xz`, and
`\beta_y`, in the plane `yz`, the displacement field through the thickness
is

.. math::
    :label: kin-displacement-field

    u(x, y, z) = u + z \beta_x, \qquad
    v(x, y, z) = v + z \beta_y, \qquad
    w(x, y, z) = w

and the two kinematics differ in the rotation `\beta_y`.


Rotation of the normal
----------------------

A circumferential displacement `v` carries a material point along an arc of
the circle of radius `R`, which turns the local frame by `v/R`,
Eq. :eq:`kin-basis`. The rotation of the normal about `x` is the rotation of
the slope of `w` plus this rotation of the frame:

.. math::
    :label: kin-rotations

    \beta_x = -w_{,x}, \qquad
    \beta_y = -w_{,y} + k_v v, \qquad
    k_v = \begin{cases} 0 & \text{Donnell-type} \\ 1/R & \text{Sanders-type}
    \end{cases}

The Donnell-type kinematics neglect the term `v/R` and keep the rotations
of a flat plate.


Linear membrane strains
-----------------------

The linear membrane strains are the same for both kinematics:

.. math::
    :label: kin-eps-linear

    \boldsymbol{\varepsilon}_L = \begin{Bmatrix} \varepsilon_{xx} \\
    \varepsilon_{yy} \\ \gamma_{xy} \end{Bmatrix}_L = \begin{Bmatrix} u_{,x}
    \\ v_{,y} + w/R \\ u_{,y} + v_{,x} \end{Bmatrix}

with the engineering shear strain `\gamma_{xy}`. The curvature of the shell
appears in the hoop strain: a uniform `w` stretches the circumference by
`w/R`. This term couples the membrane and the bending behaviour of the
shell even for a homogeneous wall.


Curvatures
----------

The changes of curvature are the derivatives of the rotations. For the
Donnell-type kinematics, `k_v = 0`, they are those of a flat plate:

.. math::
    :label: kin-kappa-donnell

    \boldsymbol{\kappa}^{D} = \begin{Bmatrix} \kappa_{xx} \\ \kappa_{yy} \\
    \kappa_{xy} \end{Bmatrix} = \begin{Bmatrix} -w_{,xx} \\ -w_{,yy} \\
    -2 w_{,xy} \end{Bmatrix}

For the Sanders-type kinematics, the term `v/R` of `\beta_y` adds
`v_{,y}/R` to `\kappa_{yy}` and `v_{,x}/R` to the twist. The rotation of the
material about the normal,

.. math::
    :label: kin-omega

    \omega = \frac{1}{2} \left( v_{,x} - u_{,y} \right)

is not a rotation of the normal and cannot be produced by a change of
`\beta_y`; Sanders' twist [Sanders1959]_ adds it, divided by `R`, such that
the twist vanishes for every small rigid-body motion. The result is

.. math::
    :label: kin-kappa-sanders

    \boldsymbol{\kappa}^{S} = \begin{Bmatrix} -w_{,xx} \\
    -w_{,yy} + \dfrac{1}{R} v_{,y} \\
    -2 w_{,xy} + \dfrac{3}{2R} v_{,x} - \dfrac{1}{2R} u_{,y} \end{Bmatrix}

where `3 v_{,x}/(2R) = v_{,x}/R + v_{,x}/(2R)` collects the term of
`\beta_y` and the term of `\omega/R`. The curvatures of both kinematics are
linear in the displacements.

.. list-table:: Linear bending measures of the two kinematics
    :header-rows: 1

    * - Kinematics
      - `\kappa_{xx}`
      - `\kappa_{yy}`
      - `\kappa_{xy}`
    * - Donnell-type
      - `-w_{,xx}`
      - `-w_{,yy}`
      - `-2 w_{,xy}`
    * - Sanders-type
      - `-w_{,xx}`
      - `-w_{,yy} + v_{,y}/R`
      - `-2 w_{,xy} + 3 v_{,x}/(2R) - u_{,y}/(2R)`


Nonlinear membrane strains
--------------------------

For moderate rotations, the nonlinear membrane strains are the products of
the rotations of the normal, Eq. :eq:`kin-rotations`:

.. math::
    :label: kin-eps-nonlinear

    \boldsymbol{\varepsilon}_{NL} = \begin{Bmatrix} \beta_x^2/2 \\
    \beta_y^2/2 \\ \beta_x \beta_y \end{Bmatrix} = \begin{Bmatrix}
    w_{,x}^2/2 \\ (w_{,y} - k_v v)^2/2 \\ w_{,x} (w_{,y} - k_v v)
    \end{Bmatrix}

For the Donnell-type kinematics these are the von Kármán terms of a flat
plate. For the Sanders-type kinematics [Sanders1963]_, `\beta_y^2/2 =
w_{,y}^2/2 + v^2/(2R^2) - v w_{,y}/R`, and the terms in `v` are those of the
Green-Lagrange strains in curvilinear coordinates that the Donnell-type
kinematics neglect. The term `\omega^2/2` of the rotation about the normal,
Eq. :eq:`kin-omega`, is neglected in both. The membrane strains are

.. math::
    :label: kin-eps-total

    \boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}_L +
    \boldsymbol{\varepsilon}_{NL}

and the curvatures remain linear. The rotations of Eq. :eq:`kin-rotations`
enter three other places, which follow the kinematics chosen: the geometric
stiffness matrix, Eq. :eq:`mat-KG`, the rotary inertia of the mass matrix,
Eq. :eq:`mat-M`, and the area vector of a follower pressure,
Eq. :eq:`fp-tangents`.


The rigid-body criterion
------------------------

Strain measures must vanish for a rigid-body motion. A small translation of
the cross-section by `c`, perpendicular to the axis, is

.. math::
    :label: kin-rigid-translation

    u = 0, \qquad v = -c \sin(y/R), \qquad w = c \cos(y/R)

for which `\varepsilon_{yy} = v_{,y} + w/R = 0` for both kinematics, but

.. math::

    \kappa_{yy}^{D} = -w_{,yy} = \frac{w}{R^2} \neq 0, \qquad
    \kappa_{yy}^{S} = -w_{,yy} + \frac{v_{,y}}{R} = 0

and likewise the Donnell-type rotation `\beta_y = -w_{,y} = c \sin(y/R)/R`
does not vanish, while the Sanders-type rotation `-w_{,y} + v/R` does. The
Donnell-type kinematics predict a bending strain of order `c/R^2` in a shell
that has not deformed; the Sanders-type strains and curvatures vanish for
every small rigid-body motion, the defining property of the Sanders-Koiter
theory [Sanders1959]_.


Range of validity
-----------------

Both kinematics share the limits of the assumptions above: thin walls,
`h/R \ll 1`, without transverse shear deformation; small strains and
moderate rotations; a linear elastic material, see :doc:`constitutive`.
Within these limits, the Donnell-type kinematics add the approximation that
the deformation varies over lengths small compared with `R`. For an
inextensional mode with `n` circumferential waves, `v_{,y} = -w/R`, the
neglected term of `\kappa_{yy}` is `|v_{,y}/R| / |w_{,yy}| = 1/n^2` of the
retained one, and the term `v/R` of `\beta_y` is `1/n^2` of `w_{,y}`. The
Donnell-type kinematics are therefore accurate when `1/n^2 \ll 1`, in
practice for `n \geq 4` or `5` circumferential waves [Amabili2008]_, and
lose accuracy for few waves [Leissa1973]_. For laminated panels under axial
compression, the error of the Donnell-type kinematics also depends on the
anisotropy of the laminate and on the ratios `R/h` and `L/R`
[JaunkyKnight1999]_.

Two results show the size of the difference. For a ring under a uniform
follower pressure buckling with `n = 2` waves, the inextensional buckling
pressure is `p R^3/D = -3` with the Sanders-type kinematics and `-16/5`
with the Donnell-type kinematics, `D` being the bending stiffness of the
wall, see :doc:`follower_pressure`. Under axial compression with a membrane
prebuckling state, `N_{yy} = N_{xy} = 0`, the geometric stiffness matrices
of the two kinematics coincide, Eq. :eq:`mat-KG`, and the difference comes
only from the curvatures in `\boldsymbol{K}_{C0}`: for a laminated cylinder
of ten plies with `R/h = 200` and `L/R \approx 2`, clamped at both ends and
under a constant axial membrane force, the critical loads of the two
elements differ by about 0.6 %.


How to choose
-------------

- The Sanders-type kinematics are the general choice: they are correct for
  rigid-body motions and apply to modes with few circumferential waves,
  e.g. rings, long cylinders under external pressure or torsion, the lowest
  natural frequencies. They cost only more nonzero entries in
  `\boldsymbol{K}_{CNL}`, `\boldsymbol{K}_G`, `\boldsymbol{M}` and the load
  stiffness matrix of the follower pressure, see :doc:`matrices`.
- The Donnell-type kinematics are adequate for short-wavelength,
  membrane-dominated behaviour, e.g. the buckling of a cylinder under axial
  compression, whose critical modes have many waves, and when a comparison
  with the classical shallow-shell results is wanted.
- The same kinematics must be used for every matrix of an analysis: the
  stiffness and the geometric stiffness matrices of a buckling problem, or
  the internal forces and the tangent stiffness matrix of a nonlinear
  analysis, belong to one strain measure. Taking every function from one
  namespace guarantees it.


Implementation
--------------

Each kinematics is an extension module with the same names of classes,
functions and sizes of the sparse arrays:

.. list-table::
    :header-rows: 1

    * - Kinematics
      - Module
      - Namespace
      - Element class
    * - Donnell-type
      - :mod:`bfsccylinder.bfsccylinder`
      - :mod:`bfsccylinder`
      - :class:`.BFSCCylinder`
    * - Sanders-type
      - :mod:`bfsccylinder.bfsccylinder_sanders`
      - :mod:`bfsccylinder.sanders`
      - :class:`bfsccylinder.bfsccylinder_sanders.BFSCCylinderSanders`

A model changes from one kinematics to the other by its imports, e.g.
``from bfsccylinder import BFSCCylinder, update_KC0, KC0_SPARSE_SIZE`` and
``from bfsccylinder.sanders import BFSCCylinderSanders, update_KC0,
KC0_SPARSE_SIZE``, the sizes being taken from the namespace of the
functions. The operators of the linear membrane strains and of the
curvatures, Eqs. :eq:`kin-eps-linear`, :eq:`kin-kappa-donnell` and
:eq:`kin-kappa-sanders`, are evaluated by :meth:`.BFSCCylinder.update_Bm`,
:meth:`.BFSCCylinder.update_Bb` and
:meth:`bfsccylinder.bfsccylinder_sanders.BFSCCylinderSanders.update_Bb`.


References
----------

.. [Donnell1933] Donnell, L. H., Stability of thin-walled tubes under
   torsion, NACA Report 479, 1933.

.. [Sanders1959] Sanders, J. L., An improved first-approximation theory for
   thin shells, NASA Technical Report R-24, 1959.

.. [Koiter1960] Koiter, W. T., A consistent first approximation in the
   general theory of thin elastic shells, Proceedings of the IUTAM Symposium
   on the Theory of Thin Elastic Shells, Delft, 1959, North-Holland,
   Amsterdam, 1960, pp. 12-33.

.. [Sanders1963] Sanders, J. L., Nonlinear theories for thin shells,
   Quarterly of Applied Mathematics, 21(1), 21-36, 1963,
   doi:10.1090/qam/147023.

.. [Amabili2008] Amabili, M., Nonlinear Vibrations and Stability of Shells
   and Plates, Cambridge University Press, 2008,
   doi:10.1017/CBO9780511619694.

.. [Leissa1973] Leissa, A. W., Vibration of Shells, NASA SP-288, National
   Aeronautics and Space Administration, Washington, DC, 1973.

.. [JaunkyKnight1999] Jaunky, N., and Knight Jr, N. F., An assessment of
   shell theories for buckling of circular cylindrical laminated composite
   panels loaded in axial compression, International Journal of Solids and
   Structures, 36(25), 3799-3820, 1999, doi:10.1016/S0020-7683(98)00177-2.
