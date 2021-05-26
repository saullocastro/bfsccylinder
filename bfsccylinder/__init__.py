"""
BFSCCYLINDER - Implementation of the single-curvature BFSC (cylinder) finite
element

Author: Saullo G. P. Castro

"""
from .bfsccylinder import (BFSCCylinder, update_KC0, update_KCNL, update_KG,
        update_KG_constant_stress, update_M, update_fint)
from .bfsccylinder import INT, DOUBLE, KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE, KG_SPARSE_SIZE, M_SPARSE_SIZE
DOF = 10

