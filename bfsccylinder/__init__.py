"""
BFSCCYLINDER - Implementation of the BFSC cylinder finite element

Author: Saullo G. P. Castro

"""
from .bfsccylinder import (BFSCCylinder, update_KC0, update_KG,
        update_KG_constant_stress, update_M)
from .bfsccylinder import INT, DOUBLE, KC0_SPARSE_SIZE, KG_SPARSE_SIZE, M_SPARSE_SIZE
DOF = 10

