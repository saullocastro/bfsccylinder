r"""The constructor initialises every ``init_k_*`` position to zero

A fresh Cython object has its C attributes zeroed by the allocation, so the
test sets them first and calls the constructor again: an attribute that
``__init__`` forgets keeps the value set before.
"""
import sys
sys.path.append(r'..')

import pytest

import bfsccylinder
import bfsccylinder.sanders

INIT_K = {'init_k_KC0', 'init_k_KCNL', 'init_k_KG', 'init_k_M',
          'init_k_KCfollower'}


@pytest.mark.parametrize('cls', [bfsccylinder.BFSCCylinder,
                                 bfsccylinder.sanders.BFSCCylinderSanders])
def test_init_k_are_zero(cls):
    elem = cls(4)
    names = {n for n in dir(elem) if n.startswith('init_k_')}
    assert names == INIT_K
    assert all(getattr(elem, n) == 0 for n in names)
    for n in names:
        setattr(elem, n, 7)
    elem.__init__(4)
    assert {n: getattr(elem, n) for n in names} == dict.fromkeys(names, 0)
