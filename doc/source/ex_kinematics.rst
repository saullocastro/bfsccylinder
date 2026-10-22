Donnell-type and Sanders-type kinematics
========================================

The tests of the Sanders-type element are those of the Donnell-type element
with the imports of ``bfsccylinder.sanders``, see :doc:`kinematics`. The
linear buckling analysis of a laminated cylinder under a constant axial
compression of :doc:`ex_linear_buckling`, with the Sanders-type element:

.. literalinclude:: ../../tests/test_sanders_linear_buckling_Z33_constant_stress.py
    :start-at: from bfsccylinder.sanders import
    :end-at: from bfsccylinder.utils import

.. literalinclude:: ../../tests/test_sanders_linear_buckling_Z33_constant_stress.py
    :pyobject: test_linear_buckling

The reference critical load of the Sanders-type element, -215444.5 N, is
about 0.7 % smaller in magnitude than the -217000 N of the Donnell-type
element in ``tests/test_linear_buckling_Z33_constant_stress.py``.

The ring of :doc:`ex_follower_pressure` is solved with both kinematics by
the same functions, which take the module and the class from the dictionary
``KINEMATICS`` of ``tests/test_follower_pressure_validation.py``:

.. literalinclude:: ../../tests/test_follower_pressure_validation.py
    :start-at: KINEMATICS = {
    :end-at: }
