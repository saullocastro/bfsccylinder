The BFSC element on the cylinder
================================

.. currentmodule:: bfsccylinder.bfsccylinder

The element interpolates the three displacements of the mid-surface with
products of cubic Hermite polynomials, whose nodal values are the
displacements and their derivatives. For the normal displacement `w` this is
the plate-bending rectangle of Bogner, Fox and Schmit [Bogner1965]_, the
BFS element, with the cross derivative `w_{,xy}` as a degree of freedom;
the in-plane displacements `u` and `v` use the same functions without the
cross derivative, as in the BFSC plate element of Castro and Jansen
[CastroJansen2021]_. The single-curvature version, for the developed
mid-surface of a cylinder, was presented by Castro and Jansen
[CastroJansen2022]_ with Donnell-type and Sanders-type kinematics. The
interpolation and the integration are the same for both kinematics.


Geometry and natural coordinates
--------------------------------

The element is a rectangle of the developed mid-surface, of sides `l_{ex}`
= ``lex`` along `x` and `l_{ey}` = ``ley`` along `y = R\theta`, with
`R` = ``R``. The natural coordinates `\xi, \eta \in [-1, 1]` give

.. math::
    :label: el-natural

    x = x_1 + \frac{l_{ex}}{2} (\xi + 1), \qquad
    y = y_1 + \frac{l_{ey}}{2} (\eta + 1), \qquad
    \frac{\partial}{\partial x} = \frac{2}{l_{ex}}
    \frac{\partial}{\partial \xi}, \qquad
    \frac{\partial}{\partial y} = \frac{2}{l_{ey}}
    \frac{\partial}{\partial \eta}, \qquad
    dA = \frac{l_{ex} l_{ey}}{4} d\xi \, d\eta

with `(x_1, y_1)` the position of node 1. The nodes are numbered
counterclockwise from the corner of the smallest `x` and `y`, as the CQUAD4
element of Nastran:

.. list-table::
    :header-rows: 1

    * - Node
      - `(\xi_i, \eta_i)`
      - Position
    * - 1
      - `(-1, -1)`
      - `(x_1, y_1)`
    * - 2
      - `(+1, -1)`
      - `(x_1 + l_{ex}, y_1)`
    * - 3
      - `(+1, +1)`
      - `(x_1 + l_{ex}, y_1 + l_{ey})`
    * - 4
      - `(-1, +1)`
      - `(x_1, y_1 + l_{ey})`

The element knows neither `x_1` nor `y_1`: its matrices depend on
`l_{ex}`, `l_{ey}`, `R`, its stiffnesses and its displacements, not on its
position. The mid-surface of the element is curved, `R` entering the
strains, while its interpolation is written on the developed rectangle.


Cubic Hermite polynomials
-------------------------

On `[-1, 1]`, the cubic Hermite polynomials of the end `\xi_i = \pm 1` are

.. math::
    :label: el-hermite-1d

    \phi_i(\xi) = \frac{1}{4} (\xi + \xi_i)^2 (2 - \xi \xi_i), \qquad
    \psi_i(\xi) = \frac{1}{4} \xi_i (\xi + \xi_i)^2 (\xi \xi_i - 1)

`\phi_i` is one at its end and zero at the other, with zero slope at both;
`\psi_i` is zero at both ends, with the slope `d\psi_i/d\xi` one at its end
and zero at the other. A cubic on `[-1, 1]` is fully determined by its
values and slopes at the two ends, which is the basis of the continuity
of the element.


Interpolation functions
-----------------------

The two-dimensional functions of node `i` are the tensor products of
Eq. :eq:`el-hermite-1d`, scaled by `l_{ex}/2` and `l_{ey}/2` such that
their nodal derivatives in `x` and `y`, not in `\xi` and `\eta`, are one:

.. math::
    :label: el-hermite-2d

    H_i = \phi_i(\xi) \phi_i(\eta), \qquad
    H_{xi} = \frac{l_{ex}}{2} \psi_i(\xi) \phi_i(\eta), \qquad
    H_{yi} = \frac{l_{ey}}{2} \phi_i(\xi) \psi_i(\eta), \qquad
    H_{xyi} = \frac{l_{ex} l_{ey}}{4} \psi_i(\xi) \psi_i(\eta)

where `\phi_i(\eta)` and `\psi_i(\eta)` are those of the end `\eta_i`.
These are the functions `H_i`, `H_{xi}`, `H_{yi}` and `H_{xyi}` of the Notes
of :mod:`bfsccylinder.bfsccylinder`. At the nodes, `H_i` is one at its node
and zero at the others, with zero derivatives `\partial/\partial x`,
`\partial/\partial y` and `\partial^2/\partial x \partial y`; the other
three functions vanish at all nodes, and so do these derivatives, except
`\partial H_{xi}/\partial x = \partial H_{yi}/\partial y = \partial^2
H_{xyi}/\partial x \partial y = 1` at their own node. The displacements are

.. math::
    :label: el-interpolation

    \begin{aligned}
    u &= \sum_{i=1}^{4} \left( H_i u_i + H_{xi} {u_{,x}}_i + H_{yi}
         {u_{,y}}_i \right) \\
    v &= \sum_{i=1}^{4} \left( H_i v_i + H_{xi} {v_{,x}}_i + H_{yi}
         {v_{,y}}_i \right) \\
    w &= \sum_{i=1}^{4} \left( H_i w_i + H_{xi} {w_{,x}}_i + H_{yi}
         {w_{,y}}_i + H_{xyi} {w_{,xy}}_i \right)
    \end{aligned}

The 16 functions of `w` span the complete bicubic polynomials, `\xi^a
\eta^b` with `a, b \leq 3`; the 12 functions of `u` and of `v` span a
subset of them.


Degrees of freedom
------------------

Each node has ``DOF = 10`` degrees of freedom, at the positions ``c`` to
``c + 9`` of the global vectors and matrices, ``c`` being the attribute
``c1``, ``c2``, ``c3`` or ``c4`` of the element for the nodes 1 to 4:

==========  ===================  ===================
Position    Degree of freedom    Dimension
==========  ===================  ===================
``c + 0``   `u`                  length
``c + 1``   `u_{,x}`             dimensionless
``c + 2``   `u_{,y}`             dimensionless
``c + 3``   `v`                  length
``c + 4``   `v_{,x}`             dimensionless
``c + 5``   `v_{,y}`             dimensionless
``c + 6``   `w`                  length
``c + 7``   `w_{,x}`             dimensionless
``c + 8``   `w_{,y}`             dimensionless
``c + 9``   `w_{,xy}`            one over length
==========  ===================  ===================

The element vector `\boldsymbol{u}_e`, of size 40, the attribute ``u``,
holds the degrees of freedom of the nodes 1, 2, 3 and 4 in sequence, 10 per
node in the order of the table. In row form,

.. math::
    :label: el-rows

    u = \boldsymbol{S}^u \boldsymbol{u}_e, \qquad
    v = \boldsymbol{S}^v \boldsymbol{u}_e, \qquad
    w = \boldsymbol{S}^w \boldsymbol{u}_e

with the rows of 40 entries evaluated at a point `(\xi, \eta)` by
:meth:`.BFSCCylinder.update_Su`, :meth:`.BFSCCylinder.update_Sv` and
:meth:`.BFSCCylinder.update_Sw`, and their derivatives, e.g.
`w_{,x} = \boldsymbol{S}^w_{,x} \boldsymbol{u}_e`, by
:meth:`.BFSCCylinder.update_Su_x`, :meth:`.BFSCCylinder.update_Su_y`,
:meth:`.BFSCCylinder.update_Sv_x`, :meth:`.BFSCCylinder.update_Sv_y`,
:meth:`.BFSCCylinder.update_Sw_x` and :meth:`.BFSCCylinder.update_Sw_y`.
The rows of the Sanders-type element,
:class:`bfsccylinder.bfsccylinder_sanders.BFSCCylinderSanders`, are the
same. The derivatives as degrees of freedom have a mechanical meaning:
`-w_{,x}` and `-w_{,y}` are the Donnell-type rotations of the normal,
`u_{,x}` the axial strain and `u_{,y} + v_{,x}` the shear strain at the
node.


Continuity between elements
---------------------------

Along an edge `x` = constant, `\xi = \pm 1`, the functions `\psi_i(\xi)`
vanish and `\phi_i(\xi)` is one for the nodes of the edge and zero for the
others, such that `w` along the edge is the cubic in `\eta` of `w` and
`w_{,y}` of the two nodes of the edge. The normal slope `w_{,x}` along the
edge has the contributions of `H_{xi}` and `H_{xyi}` only, since
`d\phi_i/d\xi` vanishes at `\xi = \pm 1`: it is the cubic in `\eta` of
`w_{,x}` and `w_{,xy}` of the same two nodes. The same holds for the edges
`y` = constant and for `u` and `v`, whose normal derivatives along an edge
depend on `u_{,x}`, `v_{,x}` (or `u_{,y}`, `v_{,y}`) of the nodes of the
edge. Two elements that share an edge, and therefore its two nodes, have
the same `u`, `v`, `w` and the same first derivatives along it: the
interpolation is `C^1` continuous, as required by the second derivatives of
`w` in the curvatures, and the membrane strains are continuous as well. The
continuity requires a structured mesh of
rectangles whose sides follow `x` and `y`, with the nodes of the elements
that share an edge at the same positions, see :doc:`assembling`.


Strain operators
----------------

Substituting Eq. :eq:`el-rows` into the strains of :doc:`kinematics` gives
the operators of the element:

.. math::
    :label: el-Bm

    \boldsymbol{\varepsilon}_L = \boldsymbol{B}_m \boldsymbol{u}_e, \qquad
    \boldsymbol{B}_m = \begin{bmatrix} \boldsymbol{S}^u_{,x} \\
    \boldsymbol{S}^v_{,y} + \boldsymbol{S}^w/R \\
    \boldsymbol{S}^u_{,y} + \boldsymbol{S}^v_{,x} \end{bmatrix}

.. math::
    :label: el-Bb

    \boldsymbol{\kappa} = \boldsymbol{B}_b \boldsymbol{u}_e, \qquad
    \boldsymbol{B}_b = \begin{bmatrix} -\boldsymbol{S}^w_{,xx} \\
    -\boldsymbol{S}^w_{,yy} + s \, \boldsymbol{S}^v_{,y}/R \\
    -2 \boldsymbol{S}^w_{,xy} + s \left( \dfrac{3}{2R} \boldsymbol{S}^v_{,x}
    - \dfrac{1}{2R} \boldsymbol{S}^u_{,y} \right) \end{bmatrix}

with `s = 0` for the Donnell-type and `s = 1` for the Sanders-type
kinematics, evaluated by :meth:`.BFSCCylinder.update_Bm`,
:meth:`.BFSCCylinder.update_Bb` and
:meth:`bfsccylinder.bfsccylinder_sanders.BFSCCylinderSanders.update_Bb`.
With the rows of the rotations,

.. math::
    :label: el-G

    \begin{Bmatrix} -\beta_x \\ -\beta_y \end{Bmatrix} =
    \begin{Bmatrix} w_{,x} \\ w_{,y} - k_v v \end{Bmatrix} =
    \boldsymbol{G} \boldsymbol{u}_e, \qquad
    \boldsymbol{G} = \begin{bmatrix} \boldsymbol{S}^w_{,x} \\
    \boldsymbol{S}^w_{,y} - k_v \boldsymbol{S}^v \end{bmatrix}

the rows ``G1`` and ``G2`` of the code, the variation of the nonlinear
strains of Eq. :eq:`kin-eps-nonlinear` is `\delta
\boldsymbol{\varepsilon}_{NL} = \boldsymbol{B}_{NL} \delta
\boldsymbol{u}_e`, and `\boldsymbol{\varepsilon}_{NL} = \frac{1}{2}
\boldsymbol{B}_{NL} \boldsymbol{u}_e`, with

.. math::
    :label: el-BNL

    \boldsymbol{B}_{NL} = \begin{bmatrix} w_{,x} \boldsymbol{G}_1 \\
    (w_{,y} - k_v v) \boldsymbol{G}_2 \\
    w_{,x} \boldsymbol{G}_2 + (w_{,y} - k_v v) \boldsymbol{G}_1
    \end{bmatrix}

`\boldsymbol{G}_1` and `\boldsymbol{G}_2` being the rows of `\boldsymbol{G}`
and the slopes those of the current displacements; the rows of
`\boldsymbol{B}_{NL}` are ``BmL1``, ``BmL2`` and ``BmL3`` in the code.
`\boldsymbol{B}_{NL}` depends on the displacements, `\boldsymbol{B}_m` and
`\boldsymbol{B}_b` do not.


.. _integration:

Numerical integration
---------------------

The integrals over the element are computed with the product
Gauss-Legendre rule of :func:`bfsccylinder.quadrature.get_points_weights`:

.. math::
    :label: el-quadrature

    \int_A f \, dA \approx \frac{l_{ex} l_{ey}}{4} \sum_{i=0}^{n-1}
    \sum_{j=0}^{n-1} w_i w_j f(\xi_i, \eta_j)

with `n` = ``nint`` points `\xi_i` in each direction, the roots of the
Legendre polynomial of degree `n`, and weights `w_i`. The rule is exact for
polynomials of degree up to `2n - 1` in each of `\xi` and `\eta`. With
`\boldsymbol{A}`, `\boldsymbol{B}` and `\boldsymbol{D}` constant over the
element, the integrands are polynomials, whose degree in each variable gives
the number of points of an exact integration:

.. list-table::
    :header-rows: 1

    * - Integrand
      - Degree in each variable
      - Exact with
    * - `\boldsymbol{K}_{C0}`, `\boldsymbol{K}_G` of a constant stress,
        follower pressure (``quadratic=0``)
      - 6
      - ``nint >= 4``
    * - `\boldsymbol{K}_G`, follower pressure (``quadratic=1``)
      - 9
      - ``nint >= 5``
    * - `\boldsymbol{K}_{CNL}`, internal forces
      - 12
      - ``nint >= 7``

the follower pressure for a pressure uniform over the element. The mass
matrix of :func:`.update_M` is integrated in closed form, without
integration points. The tangent stiffness matrix is the exact derivative of
the quadrature sums of the internal forces for any ``nint``, see
:doc:`matrices`, such that a reduced integration changes the discrete model
but not the consistency of the Newton-Raphson iterations. The same
``nint`` is given to the constructor of the element, ``BFSCCylinder(nint)``,
which sizes the arrays of the stiffnesses, and to
:func:`bfsccylinder.quadrature.get_points_weights`, whose ``points`` and
``weights`` are given to the functions.


Attributes of the element
-------------------------

Before the functions are called, each element receives:

- ``c1``, ``c2``, ``c3``, ``c4``: positions of its nodes in the global
  vectors;
- ``lex``, ``ley`` and ``R``: the lengths of its sides and the radius,
  ``R`` nonzero;
- ``A11``, ..., ``D66``: the stiffnesses at its integration points, see
  :doc:`constitutive`;
- ``h`` and ``rho``: thickness and density, for the mass matrix only;
- ``init_k_KC0``, ``init_k_KCNL``, ``init_k_KG``, ``init_k_M`` and
  ``init_k_KCfollower``: positions of its entries in the sparse arrays of
  each matrix, see :doc:`assembling`.

``n1``, ``n2``, ``n3`` and ``n4`` are identification numbers for the user,
not read by the functions. The methods ``update_Su``, ..., ``update_Bb``
evaluate the rows and operators at one point, e.g. to recover the strains
from a solution; the ``update_*`` functions of the modules evaluate the same
quantities in local variables at each integration point.


References
----------

.. [Bogner1965] Bogner, F. K., Fox, R. L., and Schmit, L. A., The
   generation of inter-element-compatible stiffness and mass matrices by the
   use of interpolation formulas, Proceedings of the Conference on Matrix
   Methods in Structural Mechanics, AFFDL-TR-66-80, Wright-Patterson Air
   Force Base, Dayton, Ohio, 26-28 October 1965, pp. 395-444.

.. [CastroJansen2021] Castro, S. G. P., and Jansen, E. L., Displacement-based
   formulation of Koiter's method: application to multi-modal post-buckling
   finite element analysis of plates, Thin-Walled Structures, 159, 107217,
   2021, doi:10.1016/j.tws.2020.107217.

.. [CastroJansen2022] Castro, S. G. P., and Jansen, E. L., Displacement-based
   multi-modal formulation of Koiter's method applied to cylindrical shells,
   AIAA SciTech 2022 Forum, AIAA 2022-0256, 2022, doi:10.2514/6.2022-0256.
