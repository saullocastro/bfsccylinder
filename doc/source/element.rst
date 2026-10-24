The element and its degrees of freedom
======================================

.. currentmodule:: bfsccylinder.bfsccylinder

The single-curvature BFSC element is a four-node rectangle of the developed
mid-surface of a cylinder of radius ``R``, with `x` the axial coordinate and
`y = R \theta` the arc length along the circumference, of sides ``lex``
along `x` and ``ley`` along `y`. The displacements `u`, `v` and `w` are
interpolated with products of cubic Hermite polynomials, whose nodal values
are the degrees of freedom below. Two classes share the same nodes, degrees
of freedom and attributes, and differ in their kinematics, see
:doc:`kinematics`:

- :class:`bfsccylinder.bfsccylinder.BFSCCylinder`, Donnell-type
  kinematics, given by the ``bfsccylinder`` namespace;
- :class:`bfsccylinder.bfsccylinder_sanders.BFSCCylinderSanders`,
  Sanders-type kinematics, given by the ``bfsccylinder.sanders`` namespace.

The interpolation functions, the kinematics and the integration of each
matrix are given in the Notes of :mod:`bfsccylinder.bfsccylinder` and
:mod:`bfsccylinder.bfsccylinder_sanders`.


Node ordering
-------------

The nodes are numbered counterclockwise from the corner with the smallest
`x` and `y`, as the CQUAD4 element of Nastran:

.. list-table::
    :header-rows: 1

    * - Node
      - Natural coordinates
      - Position
    * - 1
      - `\xi = -1, \eta = -1`
      - `(x_1, y_1)`
    * - 2
      - `\xi = +1, \eta = -1`
      - `(x_1 + lex, y_1)`
    * - 3
      - `\xi = +1, \eta = +1`
      - `(x_1 + lex, y_1 + ley)`
    * - 4
      - `\xi = -1, \eta = +1`
      - `(x_1, y_1 + ley)`

with `x = x_1 + lex (\xi + 1)/2` and `y = y_1 + ley (\eta + 1)/2`. The
element knows neither `x_1` nor `y_1`: its matrices depend on ``lex``,
``ley``, ``R``, its properties and its degrees of freedom, not on its
position.


Degrees of freedom of a node
----------------------------

Each node has ``DOF = 10`` degrees of freedom, at the positions ``c`` to
``c + 9`` of the global vectors and matrices, with ``c`` the attribute
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

`u` is the axial displacement, `v` the circumferential displacement and `w`
the normal displacement, positive outwards, away from the axis of the
cylinder. The cross derivative `w_{,xy}` is a degree of freedom of `w`
only; `u` and `v` have no cross derivative. With ``c = DOF*i`` for the node
of index ``i``, the usual choice of the tests, the slice ``u[6::DOF]`` of a
global vector gives `w` at all the nodes, ``u[0::DOF]`` gives `u`, and so
on, which is how the boundary conditions and the loads are applied, see
:doc:`assembling`.

The 40 degrees of freedom of the element, e.g. its attribute ``u``, are
those of the nodes 1, 2, 3 and 4 in sequence, 10 per node in the order of
the table.


Continuity between elements
---------------------------

Along an edge, `u`, `v`, `w` and their derivatives normal to the edge
depend only on the degrees of freedom of the two nodes of that edge, such
that `u`, `v`, `w` and their first derivatives are continuous between
elements that share an edge. Along an edge `x` = constant, for instance,
`w` is interpolated from `w` and `w_{,y}` of the two nodes, and the normal
slope `w_{,x}` from `w_{,x}` and `w_{,xy}`.


Properties of the element
-------------------------

The constructor takes ``nint``, the number of integration points in each
direction, the size of the ``points`` and ``weights`` later given to the
functions, e.g. ``BFSCCylinder(nint)``. Before the functions are called, the
attributes below are set for each element, see
:class:`bfsccylinder.bfsccylinder.BFSCCylinder` for their definitions:

- ``c1``, ``c2``, ``c3``, ``c4``: positions of the nodes in the global
  vectors;
- ``lex``, ``ley`` and ``R``: lengths of the sides and radius, ``R``
  nonzero;
- ``A11``, ..., ``D66``: the laminate stiffnesses at each of the ``nint``
  by ``nint`` integration points, arrays of shape ``(nint, nint)``, the
  entry ``[i, j]`` at ``xi = points[i]`` and ``eta = points[j]``.
  :func:`bfsccylinder.utils.assign_constant_ABD` copies the same values,
  e.g. of :func:`composites.laminated_plate`, to all the points; they can
  also be set point by point, e.g. for fiber angles that vary along `x`, as
  in ``tests/test_linear_buckling_FW_VAT_P_VT.py``;
- ``h`` and ``rho``: thickness and density, used only by the mass matrix of
  :func:`.update_M`, which with the Sanders-type kinematics also uses
  ``R``;
- ``init_k_KC0``, ``init_k_KCNL``, ``init_k_KG``, ``init_k_M``,
  ``init_k_KCfollower``: positions of the element in the sparse arrays of
  each matrix, see :doc:`assembling`.

``n1``, ``n2``, ``n3`` and ``n4`` are identification numbers for the user,
not read by the functions. The methods ``update_Su``, ..., ``update_Sw_y``,
``update_Bm`` and ``update_Bb`` of the element evaluate the interpolation
functions and the strain operators at a point `(\xi, \eta)`, e.g. to
integrate a load or to recover the strains; the ``update_*`` functions of
the modules do not use them.


Integration points
------------------

All the functions except :func:`.update_M` integrate over the element with
``nint`` by ``nint`` Gauss-Legendre points, the products of the
one-dimensional points and weights of
:func:`bfsccylinder.quadrature.get_points_weights`, given to each function.
The number of points that integrates each matrix exactly is given in the
Notes of :mod:`bfsccylinder.bfsccylinder`.
