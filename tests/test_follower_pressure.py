r"""Verification of the follower pressure of both kinematics

A follower pressure stays normal to the deformed mid-surface and acts on its
deformed area. Its force vector `\boldsymbol{F}_p(\boldsymbol{u})` is added by
``update_fext_follower`` and its load stiffness `\boldsymbol{K}_F =
-\partial \boldsymbol{F}_p/\partial \boldsymbol{u}` is written by
``update_KCfollower``, for the area vector truncated to the
terms of order 0 and 1 in the displacement gradients (``quadratic=0``) or
complete (``quadratic=1``). The checks, for the Donnell-type and the
Sanders-type kinematics and for both truncations:

1. in the undeformed state the follower force equals the dead pressure load
   `\int p \boldsymbol{S}^w dA`, computed here independently, for a
   non-uniform `p`;
2. `\boldsymbol{F}_p(\boldsymbol{u})` against an independent reference, with
   cubic Hermite rows written in ``test_mass_matrix.py`` and the area vector
   of ``np.cross``;
3. directional Taylor test of `\boldsymbol{F}_p` against `-\boldsymbol{K}_F
   \boldsymbol{d}`: the remainder is at round-off for ``quadratic=0``,
   `\boldsymbol{F}_p` being affine, and falls as `O(h)` for
   ``quadratic=1``;
4. ``reference=0`` gives `\boldsymbol{F}_p(\boldsymbol{u}) -
   \boldsymbol{F}_p(\boldsymbol{0})`;
5. ``KC0 + KCNL + KG + KCfollower`` is the Jacobian of the residual
   `\boldsymbol{R} = \boldsymbol{F}_{int} - \boldsymbol{F}_p`, by a Taylor
   test and by finite differences;
6. the sparse layout: every entry written once, ``KCfollower_SPARSE_SIZE``
   exact, the values independent of ``u`` for ``quadratic=0``, and the
   element matrix unsymmetric.

The validation against closed forms is in
``test_follower_pressure_validation.py``.
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

from test_mass_matrix import rows

DOF = 10
N = 4*DOF
KINEMATICS = {
    'donnell': (bfsccylinder, 'BFSCCylinder'),
    'sanders': (bfsccylinder.sanders, 'BFSCCylinderSanders'),
}
TRUNCATIONS = [0, 1]


def make_element(kinematics, nint=5):
    r"""One element of the cylinder of ``test_tangent_consistency.py``,
    with a pressure that varies over the element"""
    mod, cls = KINEMATICS[kinematics]
    R = 0.2032
    L = 0.3556
    prop = laminated_plate(stack=[45, -45, 0, 90, 90, 0, -45, 45],
                           laminaprop=(127.629e9, 11.3074e9, 0.300235,
                                       6.00257e9, 6.00257e9, 6.00257e9),
                           plyt=0.00101539/8)
    points, weights = get_points_weights(nint=nint)
    elem = getattr(mod, cls)(nint)
    elem.c1, elem.c2, elem.c3, elem.c4 = 0, DOF, 2*DOF, 3*DOF
    elem.R = R
    elem.lex = L/16
    elem.ley = 2*pi*R/40
    assign_constant_ABD(elem, prop)
    xi, eta = np.meshgrid(points, points, indexing='ij')
    # p[i, j] at (points[i], points[j]), external pressure
    p = np.ascontiguousarray(-2.e5*(1 + 0.3*xi - 0.2*eta**2 + 0.1*xi*eta))
    return mod, elem, points, weights, p, prop


def fext_follower(mod, elem, points, weights, p, u, quadratic, reference=1):
    f = np.zeros(N, dtype=mod.DOUBLE)
    mod.update_fext_follower(u, elem, points, weights, p, f, quadratic,
                             reference)
    return f


def sparse(mod, elem, points, weights, p, u, quadratic):
    size = mod.KCfollower_SPARSE_SIZE
    r = np.zeros(size, dtype=mod.INT)
    c = np.zeros(size, dtype=mod.INT)
    v = np.zeros(size, dtype=mod.DOUBLE)
    elem.init_k_KCfollower = 0
    mod.update_KCfollower(u, elem, points, weights, p, r, c, v, quadratic)
    return r, c, v


def KCfollower(mod, elem, points, weights, p, u, quadratic):
    r, c, v = sparse(mod, elem, points, weights, p, u, quadratic)
    return coo_matrix((v, (r, c)), shape=(N, N)).toarray()


def reference_force(kinematics, elem, points, weights, p, u, quadratic,
                    reference=1):
    r"""`\int p \boldsymbol{S}^\top \boldsymbol{a} dA` with independent Hermite rows and
    ``np.cross``"""
    R = elem.R
    kv = 1/R if kinematics == 'sanders' else 0.
    ex = np.array([1., 0., 0.])
    ey = np.array([0., 1., 0.])
    out = np.zeros(N)
    for i, (xi, wi) in enumerate(zip(points, weights)):
        for j, (eta, wj) in enumerate(zip(points, weights)):
            S = rows(xi, eta, elem.lex, elem.ley)
            d = {k: r @ u for k, r in S.items()}
            dA = np.array([d['u_x'], d['v_x'], d['w_x']])
            dB = np.array([d['u_y'], d['v_y'] + d['w']/R,
                           d['w_y'] - kv*d['v']])
            a = (reference*np.cross(ex, ey) + np.cross(ex, dB)
                 + np.cross(dA, ey) + quadratic*np.cross(dA, dB))
            wgt = wi*wj*elem.lex*elem.ley/4*p[i, j]
            out += wgt*(S['u']*a[0] + S['v']*a[1] + S['w']*a[2])
    return out


def relerr(a, b):
    return np.linalg.norm(a - b)/np.linalg.norm(b)


def random_state(prop, seed, scale=1.):
    rng = np.random.default_rng(seed)
    return scale*prop.h*rng.standard_normal(N)


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
@pytest.mark.parametrize('quadratic', TRUNCATIONS)
def test_undeformed_state_equals_dead_load(kinematics, quadratic):
    mod, elem, points, weights, p, prop = make_element(kinematics)
    f = fext_follower(mod, elem, points, weights, p, np.zeros(N), quadratic)
    dead = np.zeros(N)
    for i, (xi, wi) in enumerate(zip(points, weights)):
        for j, (eta, wj) in enumerate(zip(points, weights)):
            Sw = rows(xi, eta, elem.lex, elem.ley)['w']
            dead += wi*wj*elem.lex*elem.ley/4*p[i, j]*Sw
    assert relerr(f, dead) < 1e-13
    # no in-plane components in the undeformed state
    inplane = [i for i in range(N) if i % DOF < 6]
    assert np.all(f[inplane] == 0)


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
@pytest.mark.parametrize('quadratic', TRUNCATIONS)
@pytest.mark.parametrize('seed', [0, 1])
def test_force_against_independent_reference(kinematics, quadratic, seed):
    mod, elem, points, weights, p, prop = make_element(kinematics)
    # displacements of the order of 10 thicknesses, with gradients of the
    # order of 1e-1, such that a2 is not negligible
    u = random_state(prop, seed, scale=10.)
    for reference in (1, 0):
        f = fext_follower(mod, elem, points, weights, p, u, quadratic,
                          reference)
        ref = reference_force(kinematics, elem, points, weights, p, u,
                              quadratic, reference)
        assert relerr(f, ref) < 1e-12, (reference, relerr(f, ref))
    if quadratic:
        lin = reference_force(kinematics, elem, points, weights, p, u, 0)
        assert relerr(lin, ref) > 1e-4


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
@pytest.mark.parametrize('quadratic', TRUNCATIONS)
def test_reference_zero_is_the_change_of_force(kinematics, quadratic):
    mod, elem, points, weights, p, prop = make_element(kinematics)
    u = random_state(prop, 3, scale=10.)
    f1 = fext_follower(mod, elem, points, weights, p, u, quadratic, 1)
    f0 = fext_follower(mod, elem, points, weights, p, np.zeros(N), quadratic,
                       1)
    df = fext_follower(mod, elem, points, weights, p, u, quadratic, 0)
    assert relerr(df, f1 - f0) < 1e-12


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
@pytest.mark.parametrize('quadratic', TRUNCATIONS)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_taylor_follower_force(kinematics, quadratic, seed):
    mod, elem, points, weights, p, prop = make_element(kinematics)
    u = random_state(prop, seed, scale=10.)
    d = random_state(prop, seed + 100, scale=10.)
    F = lambda uu: fext_follower(mod, elem, points, weights, p, uu, quadratic)
    Kd = KCfollower(mod, elem, points, weights, p, u, quadratic) @ d
    F0 = F(u)
    steps = [1e-1, 1e-2, 1e-3, 1e-4]
    # K = -dF/du, hence F(u + h d) - F(u) + h K d = O(h**2)
    errors = [np.linalg.norm(F(u + h*d) - F0 + h*Kd)/np.linalg.norm(h*Kd)
              for h in steps]
    if quadratic == 0:
        # F affine: the remainder is round-off
        assert max(errors) < 1e-10, errors
    else:
        for prev, err in zip(errors[:-1], errors[1:]):
            assert 0.08 < err/prev < 0.12, errors


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
def test_sparse_layout(kinematics):
    mod, elem, points, weights, p, prop = make_element(kinematics)
    u = random_state(prop, 5, scale=10.)
    r, c, v = sparse(mod, elem, points, weights, p, u, 1)
    pairs = set(zip(r.tolist(), c.tolist()))
    # every entry written once
    assert len(pairs) == mod.KCfollower_SPARSE_SIZE
    # the entries not written are identically zero: u-u always, and v-v for
    # the Donnell-type kinematics
    field = lambda i: 'u' if i % DOF < 3 else ('v' if i % DOF < 6 else 'w')
    missing = {(field(i), field(j)) for i in range(N) for j in range(N)
               if (i, j) not in pairs}
    expected = {('u', 'u')} | ({('v', 'v')} if kinematics == 'donnell'
                               else set())
    assert missing == expected
    sizes = {'donnell': 1312, 'sanders': 1456}
    assert mod.KCfollower_SPARSE_SIZE == sizes[kinematics]
    # the values of quadratic=0 do not depend on u, and the u-v entries are
    # zero there
    K0 = KCfollower(mod, elem, points, weights, p, np.zeros(N), 0)
    Ku = KCfollower(mod, elem, points, weights, p, u, 0)
    assert np.array_equal(K0, Ku)
    iu = [i for i in range(N) if i % DOF < 3]
    iv = [i for i in range(N) if 3 <= i % DOF < 6]
    assert np.all(K0[np.ix_(iu, iv)] == 0)
    assert np.all(K0[np.ix_(iv, iu)] == 0)
    # the matrix of one element is unsymmetric: its edges are inside the
    # loaded region
    assert np.linalg.norm(K0 - K0.T) > 1e-3*np.linalg.norm(K0)
    # init_k_KCfollower is honoured and the values are added
    size = mod.KCfollower_SPARSE_SIZE
    r2 = np.zeros(2*size, dtype=mod.INT)
    c2 = np.zeros(2*size, dtype=mod.INT)
    v2 = np.ones(2*size, dtype=mod.DOUBLE)
    elem.init_k_KCfollower = size
    mod.update_KCfollower(u, elem, points, weights, p, r2, c2, v2, 0)
    assert np.all(v2[:size] == 1.) and np.all(r2[:size] == 0)
    r0, c0, v0 = sparse(mod, elem, points, weights, p, u, 0)
    assert np.array_equal(r2[size:], r0) and np.array_equal(c2[size:], c0)
    assert np.allclose(v2[size:] - 1., v0, rtol=0, atol=1e-12*abs(v0).max())


def residual_and_tangent(kinematics, quadratic):
    mod, elem, points, weights, p, prop = make_element(kinematics)
    # a pressure large enough for the load stiffness to be a sizable part
    # of the tangent, which a realistic pressure is not compared with the
    # membrane stiffness
    p = 1e3*p
    elem.init_k_KC0 = elem.init_k_KCNL = elem.init_k_KG = 0

    def assemble(update, size, u=None):
        r = np.zeros(size, dtype=mod.INT)
        c = np.zeros(size, dtype=mod.INT)
        v = np.zeros(size, dtype=mod.DOUBLE)
        if u is None:
            update(elem, points, weights, r, c, v)
        else:
            update(u, elem, points, weights, r, c, v)
        return coo_matrix((v, (r, c)), shape=(N, N)).toarray()

    KC0 = assemble(mod.update_KC0, mod.KC0_SPARSE_SIZE)

    def residual(u):
        f = np.zeros(N, dtype=mod.DOUBLE)
        mod.update_fint(u, elem, points, weights, f)
        return f - fext_follower(mod, elem, points, weights, p, u, quadratic)

    def tangent(u):
        return (KC0 + assemble(mod.update_KCNL, mod.KCNL_SPARSE_SIZE, u)
                + assemble(mod.update_KG, mod.KG_SPARSE_SIZE, u)
                + KCfollower(mod, elem, points, weights, p, u, quadratic))

    def follower(u):
        return KCfollower(mod, elem, points, weights, p, u, quadratic)

    return residual, tangent, follower, prop


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
@pytest.mark.parametrize('quadratic', TRUNCATIONS)
@pytest.mark.parametrize('seed', [0, 1])
def test_tangent_is_jacobian_of_residual(kinematics, quadratic, seed):
    residual, tangent, follower, prop = residual_and_tangent(kinematics,
                                                             quadratic)
    u = random_state(prop, seed)
    d = random_state(prop, seed + 10)
    KTd = tangent(u) @ d
    # the load stiffness carries weight in the tangent
    assert np.linalg.norm(follower(u) @ d) > 1e-3*np.linalg.norm(KTd)
    r0 = residual(u)
    steps = [1e-2, 1e-3, 1e-4, 1e-5]
    errors = [np.linalg.norm(residual(u + h*d) - r0 - h*KTd)
              /np.linalg.norm(h*KTd) for h in steps]
    for prev, err in zip(errors[:-1], errors[1:]):
        assert err < 0.2*prev, errors
    coefficients = [err/h for h, err in zip(steps, errors)]
    assert max(coefficients)/min(coefficients) < 2., errors


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
@pytest.mark.parametrize('quadratic', TRUNCATIONS)
def test_tangent_matches_finite_difference_jacobian(kinematics, quadratic):
    residual, tangent, follower, prop = residual_and_tangent(kinematics,
                                                             quadratic)
    u = random_state(prop, 7)
    step = 1e-8
    J = np.empty((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = step
        J[:, j] = (residual(u + e) - residual(u - e))/(2*step)
    assert relerr(tangent(u), J) < 1e-6
    # and the load stiffness alone, against the difference of F_p
    Kf = follower(u)
    Jf = np.empty((N, N))
    mod, elem, points, weights, p, prop = make_element(kinematics)
    p = 1e3*p
    for j in range(N):
        e = np.zeros(N)
        e[j] = step
        Jf[:, j] = -(fext_follower(mod, elem, points, weights, p, u + e,
                                   quadratic)
                     - fext_follower(mod, elem, points, weights, p, u - e,
                                     quadratic))/(2*step)
    assert relerr(Kf, Jf) < 1e-6


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
@pytest.mark.parametrize('quadratic', TRUNCATIONS)
def test_quadrature_is_exact_for_uniform_pressure(kinematics, quadratic):
    r"""With a uniform ``p`` the integrands are polynomials of degree up to
    6 in each natural coordinate for ``quadratic=0``, exact with ``nint >=
    4``, and up to 9 for ``quadratic=1``, exact with ``nint >= 5``"""
    nmin = 4 + quadratic
    out = {}
    for nint in (nmin - 1, nmin, 8):
        mod, elem, points, weights, p, prop = make_element(kinematics, nint)
        p = np.full((nint, nint), -2.e5)
        u = random_state(prop, 9, scale=10.)
        out[nint] = (fext_follower(mod, elem, points, weights, p, u,
                                   quadratic),
                     KCfollower(mod, elem, points, weights, p, u, quadratic))
    for a, b in zip(out[nmin], out[8]):
        assert relerr(a, b) < 1e-13
    # one point less is not exact
    assert max(relerr(a, b) for a, b in zip(out[nmin - 1], out[8])) > 1e-10
