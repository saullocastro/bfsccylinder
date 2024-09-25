"""
BFSCCYLINDER - Single-curvature BFSC (cylinder) finite element with Donnell-type
kinematics


"""
import ctypes

import numpy as np


from .bfsccylinder import (BFSCCylinder, update_KC0, update_KCNL, update_KG,
        update_KG_constant_stress, update_M, update_fint)
from .bfsccylinder import KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE, KG_SPARSE_SIZE, M_SPARSE_SIZE


if ctypes.sizeof(ctypes.c_long) == 8:
    # here the C long will correspond to np.int64
    INT = np.int64
else:
    # here the C long will correspond to np.int32
    INT = np.int32

DOUBLE = np.float64
DOF = 10

