"""
BFSCCYLINDER - Single-curvature BFSC (cylinder) finite element with Sanders-type
kinematics

The module gives, for the element with Sanders-type kinematics, the names
that the ``bfsccylinder`` namespace gives for the element with Donnell-type
kinematics: the element class :class:`.BFSCCylinderSanders` and the functions
``update_KC0``, ``update_KCNL``, ``update_KG``, ``update_KG_constant_stress``,
``update_M``, ``update_fint``, ``update_fext_follower`` and
``update_KCfollower`` of the module :mod:`bfsccylinder.bfsccylinder_sanders`,
with the sizes ``KC0_SPARSE_SIZE``, ``KCNL_SPARSE_SIZE``, ``KG_SPARSE_SIZE``,
``M_SPARSE_SIZE`` and ``KCfollower_SPARSE_SIZE`` of the sparse arrays that
one element fills, and ``DOF = 10``, ``INT`` and ``DOUBLE``,
defined as in the ``bfsccylinder`` namespace. ``KCNL_SPARSE_SIZE``,
``KG_SPARSE_SIZE``, ``M_SPARSE_SIZE`` and ``KCfollower_SPARSE_SIZE`` are
larger than those of the element with Donnell-type kinematics.


"""
import ctypes

import numpy as np


from .bfsccylinder_sanders import (BFSCCylinderSanders, update_KC0,
                                   update_KCNL, update_KG,
                                   update_KG_constant_stress, update_M,
                                   update_fint, update_fext_follower,
                                   update_KCfollower)
from .bfsccylinder_sanders import (KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE,
                                   KG_SPARSE_SIZE, M_SPARSE_SIZE,
                                   KCfollower_SPARSE_SIZE)

if ctypes.sizeof(ctypes.c_long) == 8:
    # here the C long will correspond to np.int64
    INT = np.int64
else:
    # here the C long will correspond to np.int32
    INT = np.int32

DOUBLE = np.float64
DOF = 10

