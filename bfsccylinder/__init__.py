"""
BFSCCYLINDER - Single-curvature BFSC (cylinder) finite element with Donnell-type
kinematics


"""
import numpy as np

from .bfsccylinder import (BFSCCylinder, update_KC0, update_KCNL, update_KG,
        update_KG_constant_stress, update_M, update_fint)
from .bfsccylinder import KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE, KG_SPARSE_SIZE, M_SPARSE_SIZE


INT = int
DOUBLE = np.float64
DOF = 10

