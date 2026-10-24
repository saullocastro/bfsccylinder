"""
BFSCCYLINDER - Single-curvature BFSC (cylinder) finite element with Donnell-type
kinematics

The ``bfsccylinder`` namespace gives the element class :class:`.BFSCCylinder`
and the functions ``update_KC0``, ``update_KCNL``, ``update_KG``,
``update_KG_constant_stress``, ``update_M``, ``update_fint``,
``update_fext_follower`` and ``update_KCfollower`` of the module
:mod:`bfsccylinder.bfsccylinder`, with the sizes ``KC0_SPARSE_SIZE``,
``KCNL_SPARSE_SIZE``, ``KG_SPARSE_SIZE``, ``M_SPARSE_SIZE`` and
``KCfollower_SPARSE_SIZE`` of the sparse arrays that one element fills. The module :mod:`bfsccylinder.sanders` gives
the same names for the element with Sanders-type kinematics. The module
:mod:`bfsccylinder.quadrature` gives the integration points and weights, and
:mod:`bfsccylinder.utils` the assignment of the laminate stiffnesses to an
element. It also defines:

- ``DOF = 10``: number of degrees of freedom per node;
- ``INT``: NumPy integer type of the C ``long`` of the platform,
  ``np.int64`` when a C ``long`` has 8 bytes and ``np.int32`` otherwise,
  matching the ``long`` arrays of row and column indices of the ``update_*``
  functions;
- ``DOUBLE = np.float64``: NumPy type of the arrays of values.


"""
import ctypes

import numpy as np


from .bfsccylinder import (BFSCCylinder, update_KC0, update_KCNL, update_KG,
        update_KG_constant_stress, update_M, update_fint,
        update_fext_follower, update_KCfollower)
from .bfsccylinder import (KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE, KG_SPARSE_SIZE,
        M_SPARSE_SIZE, KCfollower_SPARSE_SIZE)


if ctypes.sizeof(ctypes.c_long) == 8:
    # here the C long will correspond to np.int64
    INT = np.int64
else:
    # here the C long will correspond to np.int32
    INT = np.int32

DOUBLE = np.float64
DOF = 10

