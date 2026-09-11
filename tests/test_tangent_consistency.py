"""The tangent stiffness matrix must be the derivative of the internal force

KT = KC0 + KCNL(u) + KG(u) is what every Newton-Raphson built on these
elements uses as the Jacobian of fint. If the two drift apart, the analyses
still converge, but linearly instead of quadratically, with a contraction
factor set by how far KT is from the true Jacobian, which on a fine mesh is
close enough to one to exhaust the iteration cap at every load step.

The check is a directional Taylor test,

    |fint(u + h d) - fint(u) - h KT d| / |h KT d|

which falls proportionally to h for a consistent tangent and plateaus for an
inconsistent one. Both kinematics are covered, over several random states
and directions, in a deformed state of the order of the shell thickness so
that the nonlinear terms actually carry weight.
"""
import sys

sys.path.append(r'..')

import numpy as np
import pytest
from numpy import pi
from scipy.sparse import coo_matrix
from composites import laminated_plate

import bfsccylinder
import bfsccylinder.sanders
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD

DOF = 10
NINT = 4

KINEMATICS = {
    'donnell': (bfsccylinder, 'BFSCCylinder'),
    'sanders': (bfsccylinder.sanders, 'BFSCCylinderSanders'),
}


def make_element(name):
    """One element of the Arbocz/Starnes cylinder, with its laminate."""
    mod, cls = KINEMATICS[name]
    R = 0.2032
    L = 0.3556
    prop = laminated_plate(stack=[45, -45, 0, 90, 90, 0, -45, 45],
                           laminaprop=(127.629e9, 11.3074e9, 0.300235,
                                       6.00257e9, 6.00257e9, 6.00257e9),
                           plyt=0.00101539/8)
    points, weights = get_points_weights(nint=NINT)
    elem = getattr(mod, cls)(NINT)
    elem.n1, elem.n2, elem.n3, elem.n4 = 1, 2, 3, 4
    elem.c1, elem.c2, elem.c3, elem.c4 = 0, DOF, 2*DOF, 3*DOF
    elem.R = R
    elem.lex = L/16
    elem.ley = 2*pi*R/40
    elem.init_k_KC0 = 0
    elem.init_k_KCNL = 0
    elem.init_k_KG = 0
    assign_constant_ABD(elem, prop)
    return mod, elem, points, weights, prop


def assemble(mod, update, elem, points, weights, size, u=None):
    n = 4*DOF
    r = np.zeros(size, dtype=mod.INT)
    c = np.zeros(size, dtype=mod.INT)
    v = np.zeros(size, dtype=mod.DOUBLE)
    if u is None:
        update(elem, points, weights, r, c, v)
    else:
        update(u, elem, points, weights, r, c, v)
    return coo_matrix((v, (r, c)), shape=(n, n)).toarray()


def make_callables(name):
    mod, elem, points, weights, prop = make_element(name)
    n = 4*DOF

    def fint(u):
        f = np.zeros(n, dtype=mod.DOUBLE)
        mod.update_fint(u, elem, points, weights, f)
        return f

    KC0 = assemble(mod, mod.update_KC0, elem, points, weights,
                   mod.KC0_SPARSE_SIZE)

    def KT(u):
        return (KC0
                + assemble(mod, mod.update_KCNL, elem, points, weights,
                           mod.KCNL_SPARSE_SIZE, u)
                + assemble(mod, mod.update_KG, elem, points, weights,
                           mod.KG_SPARSE_SIZE, u))

    return fint, KT, prop.h, n


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_tangent_is_derivative_of_fint(kinematics, seed):
    """Taylor test: the error must fall by about ten for each decade of h."""
    fint, KT, h_shell, n = make_callables(kinematics)
    rng = np.random.default_rng(seed)
    u = h_shell*rng.standard_normal(n)
    d = h_shell*rng.standard_normal(n)

    f0 = fint(u)
    KTd = KT(u) @ d
    scale = np.linalg.norm(KTd)
    assert scale > 0

    steps = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]
    errors = [np.linalg.norm(fint(u + h*d) - f0 - h*KTd)/(h*scale)
              for h in steps]

    # a consistent tangent leaves a second order remainder, so the error
    # falls by ten for every decade of h; an inconsistent one leaves a first
    # order remainder, so the error plateaus instead
    for h, prev, err in zip(steps[1:], errors[:-1], errors[1:]):
        assert err < 0.2*prev, (
            'error did not fall by ten going to h=%.0e: %.3e -> %.3e; the '
            'tangent is not the derivative of fint' % (h, prev, err))

    # the same statement without an arbitrary scale: error/h is the constant
    # that multiplies the second derivative, so it must not drift with h.
    # An inconsistent tangent spreads this over four orders of magnitude
    coefficients = [err/h for h, err in zip(steps, errors)]
    spread = max(coefficients)/min(coefficients)
    assert spread < 2., (
        'error/h drifted by a factor %.1f over %.0e..%.0e, so the remainder '
        'is not second order' % (spread, steps[0], steps[-1]))

    # and the remainder is small in absolute terms once h is small, rather
    # than merely self-consistent
    assert errors[-1] < 1.e-6, (
        'residual %.3e at h=%.0e is too large for a consistent tangent'
        % (errors[-1], steps[-1]))


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
def test_tangent_matches_finite_difference_jacobian(kinematics):
    """Every entry of KT, against a central difference of fint."""
    fint, KT, h_shell, n = make_callables(kinematics)
    rng = np.random.default_rng(7)
    u = h_shell*rng.standard_normal(n)

    step = 1.e-8
    J = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = step
        J[:, j] = (fint(u + e) - fint(u - e))/(2*step)

    K = KT(u)
    err = np.linalg.norm(K - J)/np.linalg.norm(J)
    assert err < 1.e-6, 'KT differs from d(fint)/du by %.3e' % err


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
def test_tangent_is_symmetric(kinematics):
    """A tangent that comes from a strain energy is symmetric."""
    fint, KT, h_shell, n = make_callables(kinematics)
    rng = np.random.default_rng(11)
    u = h_shell*rng.standard_normal(n)
    K = KT(u)
    assert np.linalg.norm(K - K.T)/np.linalg.norm(K) < 1.e-12


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
def test_fint_and_tangent_vanish_in_the_undeformed_state(kinematics):
    """At u = 0 there is no internal force, and KT reduces to KC0."""
    mod, elem, points, weights, prop = make_element(kinematics)
    n = 4*DOF
    u = np.zeros(n, dtype=mod.DOUBLE)
    f = np.zeros(n, dtype=mod.DOUBLE)
    mod.update_fint(u, elem, points, weights, f)
    assert np.all(f == 0)

    KC0 = assemble(mod, mod.update_KC0, elem, points, weights,
                   mod.KC0_SPARSE_SIZE)
    KCNL = assemble(mod, mod.update_KCNL, elem, points, weights,
                    mod.KCNL_SPARSE_SIZE, u)
    KG = assemble(mod, mod.update_KG, elem, points, weights,
                  mod.KG_SPARSE_SIZE, u)
    assert np.abs(KCNL).max() == 0
    assert np.abs(KG).max() == 0
    assert np.abs(KC0).max() > 0


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
def test_fint_is_linear_for_small_displacements(kinematics):
    """For a state well below the thickness, fint approaches KC0 @ u."""
    mod, elem, points, weights, prop = make_element(kinematics)
    n = 4*DOF
    rng = np.random.default_rng(3)
    u = 1.e-6*prop.h*rng.standard_normal(n)
    f = np.zeros(n, dtype=mod.DOUBLE)
    mod.update_fint(u, elem, points, weights, f)
    KC0 = assemble(mod, mod.update_KC0, elem, points, weights,
                   mod.KC0_SPARSE_SIZE)
    assert np.linalg.norm(f - KC0 @ u)/np.linalg.norm(KC0 @ u) < 1.e-6


if __name__ == '__main__':
    for kin in sorted(KINEMATICS):
        fint, KT, h_shell, n = make_callables(kin)
        rng = np.random.default_rng(0)
        u = h_shell*rng.standard_normal(n)
        d = h_shell*rng.standard_normal(n)
        f0 = fint(u)
        KTd = KT(u) @ d
        print(kin)
        prev = None
        for h in (1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7):
            err = (np.linalg.norm(fint(u + h*d) - f0 - h*KTd)
                   / np.linalg.norm(h*KTd))
            ratio = '' if prev is None else '   (x %.2f)' % (err/prev)
            print('   h %.0e   rel err %.6e%s' % (h, err, ratio))
            prev = err
