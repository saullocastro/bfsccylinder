Conventions and limits
======================

.. currentmodule:: bfsccylinder.bfsccylinder

This chapter collects the conventions of the library and the limits of the
models of the preceding chapters.


Conventions
-----------

.. list-table::
    :header-rows: 1

    * - Item
      - Convention
    * - Coordinates
      - cylindrical `(x, \theta, r)`, `x` along the axis; the element uses
        `x`, the arc length `y = R\theta` of the mid-surface and `z = r - R`,
        Eq. :eq:`kin-arc-length`
    * - Local basis
      - `\boldsymbol{e}_x`, `\boldsymbol{e}_y`, `\boldsymbol{e}_z`
        right-handed, `\boldsymbol{e}_z` the outward normal
    * - Displacements
      - `u` axial, `v` circumferential, `w` normal, positive outwards
    * - Rotations of the normal
      - `\beta_x = -w_{,x}`, `\beta_y = -w_{,y} + k_v v`, `k_v = 0`
        (Donnell-type) or `1/R` (Sanders-type), Eq. :eq:`kin-rotations`
    * - Strains
      - engineering shear strain `\gamma_{xy}`; curvatures as derivatives
        of the rotations of the normal, `\kappa_{xx} = -w_{,xx}`, Eqs.
        :eq:`kin-kappa-donnell` and :eq:`kin-kappa-sanders`
    * - Resultants
      - `N_{xx}`, `N_{yy}`, `N_{xy}` positive in tension; `N_{xx} < 0` for
        an axial compression
    * - Node order
      - counterclockwise in `(x, y)` from the corner of the smallest `x`
        and `y`, see :doc:`element`
    * - Degrees of freedom of a node
      - `u`, `u_{,x}`, `u_{,y}`, `v`, `v_{,x}`, `v_{,y}`, `w`, `w_{,x}`,
        `w_{,y}`, `w_{,xy}`, at the positions ``c`` to ``c + 9``
    * - Pressure
      - per unit undeformed area, positive along `+\boldsymbol{e}_z`; an
        external pressure is negative
    * - Load multiplier
      - `\lambda` multiplies the reference load that produced
        `\boldsymbol{K}_G`, or the resultants given to
        :func:`.update_KG_constant_stress`
    * - Load stiffness
      - `\boldsymbol{K}_F = -\partial \boldsymbol{F}_p/\partial
        \boldsymbol{u}`, added to the tangent stiffness matrix
    * - Residual
      - `\boldsymbol{R} = \boldsymbol{F}_{int} - \lambda
        \boldsymbol{F}_{ext}`
    * - Units
      - any consistent set; the library has no unit conversion
    * - Sparse storage
      - row, column and value arrays per element, summed on assembly,
        see :doc:`assembling`

The functions of both kinematics take the same arguments and use the same
conventions; :func:`bfsccylinder.quadrature.get_points_weights` gives the
integration points, in increasing order, used in both directions.


Limits of the models
--------------------

- **Geometry**: a perfect circular cylinder of constant radius, meshed with
  a structured grid of rectangles in `x` and `y`. Cones, doubly curved
  shells, geometric imperfections and cut-outs that break the structured
  grid are not modelled.
- **Thin walls**: Kirchhoff-Love kinematics without transverse shear
  deformation and stiffnesses without terms in `h/R`; the element is not
  meant for thick or sandwich walls, whose transverse shear flexibility
  matters.
- **Kinematics**: small strains and moderate rotations; the rotation about
  the normal is not in the nonlinear strains. The Donnell-type kinematics
  are in addition restricted to deformations with many circumferential
  waves, see :doc:`kinematics`.
- **Material**: linear elastic laminate, stiffnesses referred to the
  mid-surface, given at the integration points; the mass matrix assumes a
  density uniform through the thickness and the reference surface at its
  middle.
- **Loads**: concentrated nodal forces, prescribed displacements, a dead or
  follower pressure function of the undeformed coordinates, and, for linear
  buckling, an assumed constant membrane state. A pressure that depends on
  the current position, e.g. a hydrostatic head, is not modelled.
- **Integration**: exact for constant stiffnesses with the numbers of points
  of :ref:`integration`; with a varying stiffness or pressure, the
  integration points sample it.
- **Analyses**: the library provides the element matrices and vectors; the
  assembly, the boundary conditions and the solution of the linear,
  eigenvalue and nonlinear systems are done by the user, with SciPy or
  structsolve, see :doc:`analyses`.
