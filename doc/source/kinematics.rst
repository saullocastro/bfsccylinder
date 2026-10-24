Donnell-type and Sanders-type kinematics
========================================

The package has the same element with two kinematics, in two extension
modules with the same names of classes, functions and sizes:

.. list-table::
    :header-rows: 1

    * - Kinematics
      - Module
      - Imported from
      - Element class
    * - Donnell-type
      - :mod:`bfsccylinder.bfsccylinder`
      - ``bfsccylinder``
      - ``BFSCCylinder``
    * - Sanders-type
      - :mod:`bfsccylinder.bfsccylinder_sanders`
      - ``bfsccylinder.sanders``
      - ``BFSCCylinderSanders``

A model changes from one to the other by its imports, e.g.::

    from bfsccylinder import (BFSCCylinder, update_KC0, update_KG, DOF,
                              DOUBLE, INT, KC0_SPARSE_SIZE, KG_SPARSE_SIZE)

and::

    from bfsccylinder.sanders import (BFSCCylinderSanders, update_KC0,
                                      update_KG, DOF, DOUBLE, INT,
                                      KC0_SPARSE_SIZE, KG_SPARSE_SIZE)

with the sizes of the sparse arrays taken from the same namespace, since
those of KCNL, KG, M and KCfollower are larger for the Sanders-type
element. The tests ``tests/test_sanders_*.py`` are the tests of the
Donnell-type element with these imports, see :doc:`ex_kinematics`. The
nodes, the degrees of freedom, the interpolation, the constitutive relation
and the integration points are the same, see :doc:`element`.


What differs
------------

Both kinematics are those of a thin cylindrical shell under the
Kirchhoff-Love hypothesis, with the same linear membrane strains, where the
curvature of the shell enters through the term `w/R` of
`\varepsilon_{yy}`. They differ in the curvatures and in the rotation of
the normal about `x`:

- Donnell-type: the curvatures are those of a flat plate, `\kappa_{yy} =
  -w_{,yy}`, and the rotation about `x` is `-w_{,y}`, in the nonlinear
  membrane strains, in the rotary inertia of the mass matrix and in the
  area vector of the follower pressure;
- Sanders-type: the curvatures also have terms of `u` and `v` divided by
  `R`, and the rotation about `x` is `\beta_y = -w_{,y} + v/R`, in the
  same three places, such that KCNL, KG and KCfollower have entries at the
  degrees of freedom of `v` and the mass matrix couples `v` and `w`.

The strains and curvatures of both are given in the Notes of
:mod:`bfsccylinder.bfsccylinder` and
:mod:`bfsccylinder.bfsccylinder_sanders`.


When each applies
-----------------

The Donnell-type approximations are accurate when the displacements vary
over lengths that are small compared with `R`, e.g. buckling and vibration
modes with many circumferential waves, and lose accuracy for long waves.
For instance, they do not give zero curvatures for a small rigid-body
translation of the cross-section. The Sanders-type kinematics give zero
strains and curvatures for it, and their linear part is not restricted to
short waves. The rings of the
validation tests of the follower pressure, which buckle with two
circumferential waves, show the difference: the inextensional buckling
pressure `p R^3/D` is `-3` with the Sanders-type kinematics and `-16/5`
with the Donnell-type kinematics, under a follower pressure, see
:doc:`ex_follower_pressure`. For the laminated cylinder under axial compression of
:doc:`ex_kinematics`, the reference critical loads asserted by the tests of
the two kinematics differ by about 0.7 %.

The Sanders-type element has more entries per element, see
:mod:`bfsccylinder.bfsccylinder_sanders`, and a model with it more nonzero
entries in KCNL, KG, M and KCfollower.
