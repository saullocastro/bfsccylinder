Constitutive relation
=====================

.. currentmodule:: bfsccylinder.bfsccylinder

The wall of the shell is a laminate of linear elastic orthotropic plies,
described by the classical laminated plate theory [Reddy2003]_. Both
kinematics of :doc:`kinematics` use the same constitutive relation; only the
strains and curvatures that enter it differ.


Stress resultants
-----------------

With the Kirchhoff-Love strains `\boldsymbol{\varepsilon}(z) =
\boldsymbol{\varepsilon} + z \boldsymbol{\kappa}` and the plane-stress
stiffness `\bar{\boldsymbol{Q}}(z)` of the ply at `z`, in the axes `x`, `y`
of the shell, the membrane stress resultants `\boldsymbol{N} = \{N_{xx},
N_{yy}, N_{xy}\}^\top` (force per unit length) and the moment resultants
`\boldsymbol{M} = \{M_{xx}, M_{yy}, M_{xy}\}^\top` (force times length per
unit length) are

.. math::
    :label: con-ABD

    \begin{Bmatrix} \boldsymbol{N} \\ \boldsymbol{M} \end{Bmatrix} =
    \int_{-h/2}^{h/2} \begin{Bmatrix} 1 \\ z \end{Bmatrix}
    \bar{\boldsymbol{Q}}(z) \left( \boldsymbol{\varepsilon} + z
    \boldsymbol{\kappa} \right) dz =
    \begin{bmatrix} \boldsymbol{A} & \boldsymbol{B} \\ \boldsymbol{B} &
    \boldsymbol{D} \end{bmatrix}
    \begin{Bmatrix} \boldsymbol{\varepsilon} \\ \boldsymbol{\kappa}
    \end{Bmatrix}, \qquad
    (\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{D}) = \int_{-h/2}^{h/2}
    \bar{\boldsymbol{Q}}(z) (1, z, z^2) \, dz

with

.. math::
    :label: con-A

    \boldsymbol{A} = \begin{bmatrix} A_{11} & A_{12} & A_{16} \\ A_{12} &
    A_{22} & A_{26} \\ A_{16} & A_{26} & A_{66} \end{bmatrix}

and likewise `\boldsymbol{B}` and `\boldsymbol{D}`. `\boldsymbol{A}` is the
membrane stiffness (force per unit length), `\boldsymbol{B}` the
membrane-bending coupling (force) and `\boldsymbol{D}` the bending
stiffness (force times length). `\boldsymbol{\varepsilon}` is the total
membrane strain of Eq. :eq:`kin-eps-total`, linear plus nonlinear, such that
`\boldsymbol{N}` and `\boldsymbol{M}` are the resultants of the current
configuration in a nonlinear analysis.

The integral over `z` is the one of a flat laminate: the factor `1 + z/R`
of the area of a layer of the curved wall is neglected, consistently with
the shallow-shell thickness integration of the kinematics. The stiffnesses
have no terms in `h/R`, which limits the element to thin walls. `z` is
positive outwards, as `w`, such that a ply at the bottom of the laminate,
the smallest `z`, is at the inner surface of the cylinder. The stiffnesses
are referred to the mid-surface of the element: a laminate whose reference
surface is offset from its mid-plane has nonzero `\boldsymbol{B}` terms
from the offset, which the stiffness matrices take into account, while the
mass matrix of Eq. :eq:`mat-M` assumes the reference surface at the middle
of the thickness. The transverse shear stiffnesses of a laminate are not
used: the Kirchhoff-Love kinematics have no transverse shear strain.


Stiffnesses at the integration points
-------------------------------------

The element stores the 18 entries ``A11``, ..., ``D66`` as arrays of shape
``(nint, nint)``, one value per integration point, the entry ``[i, j]`` at
`\xi` = ``points[i]`` and `\eta` = ``points[j]``, see :doc:`element`. Every
integral of the element is evaluated with the `\boldsymbol{A}`,
`\boldsymbol{B}` and `\boldsymbol{D}` of the integration point, such that:

- a laminate uniform over the element has the same values at every point,
  which :func:`bfsccylinder.utils.assign_constant_ABD` copies from an object
  with the attributes ``A11``, ..., ``D66``, e.g. a laminate of
  :func:`composites.laminated_plate` or :func:`composites.isotropic_plate`;
- a laminate whose stiffness varies over the shell, e.g. a variable-angle
  laminate whose fiber angles vary along `x`, is described point by point,
  by setting the arrays to the stiffnesses of the laminate at the position
  of each integration point.

The fiber angles of the laminate are measured in the plane `x y` of the
shell, `x` axial and `y` circumferential. The stiffnesses are integrated
exactly only when they are constant over the element, see
:ref:`integration`; with a varying stiffness, the integration points sample
it, and the error decreases as the mesh is refined.


References
----------

.. [Reddy2003] Reddy, J. N., Mechanics of Laminated Composite Plates and
   Shells: Theory and Analysis, 2nd ed., CRC Press, 2003, doi:10.1201/b12409.
