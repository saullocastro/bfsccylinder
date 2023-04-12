"""
BFSCCYLINDER - Single-curvature BFSC (cylinder) finite element with Sanders-type
kinematics


"""
import numpy as np

from .bfsccylinder_sanders import (BFSCCylinderSanders, update_KC0,
                                   update_KCNL, update_KG,
                                   update_KG_constant_stress, update_M,
                                   update_fint)
from .bfsccylinder_sanders import (KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE,
                                   KG_SPARSE_SIZE, M_SPARSE_SIZE)

INT = int
DOUBLE = np.float64
DOF = 10

