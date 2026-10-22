Follower pressure
=================

.. currentmodule:: bfsccylinder.bfsccylinder

Residual and tangent stiffness matrix
-------------------------------------

The residual `\{f_{int}(u)\} - \{F_p(u)\}` and its Jacobian
`[K_{C0}] + [K_{CNL}] + [K_G] + [K_{Cfollower}]` of one element, for both
kinematics, the modules of :doc:`kinematics` being selected by the
dictionary ``KINEMATICS``, with a pressure that varies over the element:

.. literalinclude:: ../../tests/test_follower_pressure.py
    :pyobject: make_element

.. literalinclude:: ../../tests/test_follower_pressure.py
    :pyobject: residual_and_tangent

The tangent stiffness matrix against a Taylor test of the residual:

.. literalinclude:: ../../tests/test_follower_pressure.py
    :pyobject: test_tangent_is_jacobian_of_residual

Linear buckling of a ring
-------------------------

A long cylinder under a uniform external pressure buckles as a ring. The
ring is modelled with one element along `x`, the mesh closed around the
circumference, and symmetry planes that keep the modes with an even number
of circumferential waves; see the docstring of
``tests/test_follower_pressure_validation.py``. The matrices of the model,
with `[K_G]` of the linear static solution of the dead pressure load and
`[K_{Cfollower}]` at `\{u\} = \{0\}`:

.. literalinclude:: ../../tests/test_follower_pressure_validation.py
    :pyobject: cylinder_model

.. literalinclude:: ../../tests/test_follower_pressure_validation.py
    :pyobject: assemble

.. literalinclude:: ../../tests/test_follower_pressure_validation.py
    :pyobject: matrices

.. literalinclude:: ../../tests/test_follower_pressure_validation.py
    :pyobject: ring

The linear buckling problem `([K_{C0}] + \lambda ([K_G] +
[K_{Cfollower}]))\{\phi\} = \{0\}` of the follower pressure, or without
`[K_{Cfollower}]` for the dead pressure, solved with ``structsolve.lb``,
with the unsymmetric solvers for the follower pressure:

.. literalinclude:: ../../tests/test_follower_pressure_validation.py
    :pyobject: buckling

.. literalinclude:: ../../tests/test_follower_pressure_validation.py
    :pyobject: ring_buckling_pressure

The buckling pressure against the closed-form solution of the ring with the
same strains and load terms, and against its inextensional limit, for both
kinematics and both pressures:

.. literalinclude:: ../../tests/test_follower_pressure_validation.py
    :pyobject: test_isotropic_ring
