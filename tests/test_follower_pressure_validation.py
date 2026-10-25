r"""Validation of the follower pressure against closed forms

1. Rings. A long cylinder under a uniform pressure buckles as a ring, in
   generalized plane strain. The ring is modelled with one element along
   `x` and the mesh closed around the circumference, with `u`, `u_{,x}`,
   `u_{,y}`, `v_{,x}`, `w_{,x}` and `w_{,xy}` zero at every node, such that
   `\varepsilon_{xx} = 0` and the fields do not vary along `x`, and with
   symmetry planes at `y = 0, \pi R/2, \pi R, 3 \pi R/2` (`v = w_{,y} = 0`),
   which remove the rigid-body motions and keep the modes `w = \cos(n y/R)`
   with `n` even. The linear buckling problem

   .. math::
       ([K_{C0}] + \lambda ([K_G(u_0)] + [K_{Cfollower}]))\{\phi\} = \{0\}

   with `u_0` the linear static solution of a unit external pressure, is
   solved with :func:`structsolve.lb` and the unsymmetric solvers, and
   compared with :func:`ring_closed_form`, the exact second variation of the
   ring with the same strains and load terms, for the mode `n = 2`: the
   Sanders-type and Donnell-type kinematics, with the follower pressure
   (first-order area vector) or with a pressure of constant direction (dead
   load, `[K_{Cfollower}]` left out). In the inextensional limit, `p R^3/D =
   -3` (Sanders, follower), `-4` (dead) and `-16/5` (Donnell, follower).

2. Finite cylinders with shear-diaphragm ends under a lateral or a
   hydrostatic pressure, with the membrane prestress of
   ``update_KG_constant_stress``, against :func:`cylinder_closed_form`, the
   exact solution of the same strains with trigonometric modes. Their modes
   depend on `x` and test the terms of `u` of the follower load.

Both closed forms are those of ``tests/tests_shell/
test_follower_pressure_validation.py`` of the panels repository, with the
same cases. Units: N, m, Pa.
"""
import sys
sys.path.append(r'..')

import numpy as np
import pytest
from numpy import pi
from scipy.linalg import eig
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from composites import isotropic_plate
from structsolve import lb

import bfsccylinder
import bfsccylinder.sanders
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD

DOF = 10
KINEMATICS = {
    'donnell': (bfsccylinder, 'BFSCCylinder'),
    'sanders': (bfsccylinder.sanders, 'BFSCCylinderSanders'),
}
E = 70.e9
NU = 0.3


def ring_closed_form(kinematics, load, n, A, D, R):
    r"""Buckling pressure of a ring in plane strain, mode ``cos(n y/R)``

    The displacements are ``w = W cos(n y/R)`` and ``v = V sin(n y/R)``,
    the prestress is the membrane state ``Nyy = p R`` of the pressure ``p``
    (positive outwards, an external pressure is negative), and the second
    variation of the total potential is

        U2 = 1/2 int (A eyy**2 + D kyy**2 + Nyy byy**2) dy
             - 1/2 int p u.a1(u) dy

    with ``eyy = v,y + w/R``, ``kyy = -w,yy + kv v,y``, ``byy = w,y - kv v``,
    ``kv = 1/R`` (Sanders) or ``0`` (Donnell), and ``u.a1(u) = v (-w,y + kv
    v) + w (v,y + w/R)`` for the follower pressure or zero for the dead
    pressure. The integral over the circumference is ``pi R`` times the
    products of the amplitudes, and the pressure of smallest modulus that
    makes ``U2`` singular is returned.

    """
    k = n/R
    kv = 1/R if kinematics == 'sanders' else 0.
    # coefficient vectors of the amplitudes (W, V)
    eyy = np.array([1/R, k])
    kyy = np.array([k**2, kv*k])
    byy = np.array([-k, -kv])
    H0 = A*np.outer(eyy, eyy) + D*np.outer(kyy, kyy)
    H1 = R*np.outer(byy, byy)
    if load == 'follower':
        H1 -= np.array([[1/R, k], [k, kv]])
    elif load != 'dead':
        raise ValueError(load)
    ps = eig(H0, -H1, right=False)
    ps = ps[np.isfinite(ps)]
    ps = ps[np.abs(ps.imag) <= 1e-10*np.abs(ps)].real
    ps = ps[ps != 0]
    return ps[np.argmin(np.abs(ps))]


def cylinder_model(kinematics, L, R, prop, nx, ny, nint=4):
    r"""Cylinder of length ``L`` with ``nx`` nodes along `x` and ``ny``
    elements around the closed circumference"""
    mod, cls = KINEMATICS[kinematics]
    points, weights = get_points_weights(nint=nint)
    ids = np.arange(nx*ny).reshape(nx, ny)
    x = np.repeat(np.linspace(0, L, nx), ny)
    y = np.tile(np.arange(ny)*2*pi*R/ny, nx)
    elements = []
    for ix in range(nx - 1):
        for iy in range(ny):
            elem = getattr(mod, cls)(nint)
            n1, n2 = ids[ix, iy], ids[ix + 1, iy]
            n3, n4 = ids[ix + 1, (iy + 1) % ny], ids[ix, (iy + 1) % ny]
            elem.n1, elem.n2, elem.n3, elem.n4 = n1, n2, n3, n4
            elem.c1, elem.c2, elem.c3, elem.c4 = (DOF*n1, DOF*n2, DOF*n3,
                                                  DOF*n4)
            elem.R = R
            elem.lex = L/(nx - 1)
            elem.ley = 2*pi*R/ny
            assign_constant_ABD(elem, prop)
            elements.append(elem)
    return mod, elements, points, weights, x, y, DOF*nx*ny


def assemble(mod, elements, update, tag, size, *args):
    n = len(elements)
    spsize = getattr(mod, tag + '_SPARSE_SIZE')
    r = np.zeros(spsize*n, dtype=mod.INT)
    c = np.zeros(spsize*n, dtype=mod.INT)
    v = np.zeros(spsize*n, dtype=mod.DOUBLE)
    for i, elem in enumerate(elements):
        setattr(elem, 'init_k_' + tag, i*spsize)
        update(*args[:-1], elem, *args[-1], r, c, v)
    return coo_matrix((v, (r, c)), shape=(size, size)).tocsc()


def matrices(mod, elements, points, weights, size, p, u0=None,
             Nxx=None, Nyy=None):
    r"""``KC0``, ``KG`` (of ``u0`` or of the constant ``Nxx``, ``Nyy``),
    ``KCfollower`` and the follower force vector at zero displacements, the
    dead load"""
    pp = np.full((len(points), len(points)), p)
    KC0 = assemble(mod, elements, mod.update_KC0, 'KC0', size,
                   (points, weights))
    zero = np.zeros(size)
    KCf = assemble(mod, elements, mod.update_KCfollower, 'KCfollower', size,
                   zero, (points, weights, pp))
    f = np.zeros(size)
    for elem in elements:
        mod.update_fext_follower(zero, elem, points, weights, pp, f)
    if u0 is not None:
        KG = assemble(mod, elements, mod.update_KG, 'KG', size, u0,
                      (points, weights))
    else:
        n = len(elements)
        spsize = mod.KG_SPARSE_SIZE
        r = np.zeros(spsize*n, dtype=mod.INT)
        c = np.zeros(spsize*n, dtype=mod.INT)
        v = np.zeros(spsize*n, dtype=mod.DOUBLE)
        for i, elem in enumerate(elements):
            elem.init_k_KG = i*spsize
            mod.update_KG_constant_stress(elem, points, weights, r, c, v,
                                          Nxx, Nyy, 0.)
        KG = coo_matrix((v, (r, c)), shape=(size, size)).tocsc()
    return KC0, KG, KCf, f


def ring(kinematics, R, h, ny):
    prop = isotropic_plate(thickness=h, E=E, nu=NU)
    out = cylinder_model(kinematics, 0.1*R, R, prop, 2, ny)
    mod, elements, points, weights, x, y, size = out
    bk = np.zeros(size, dtype=bool)
    for d in (0, 1, 2, 4, 7, 9):
        bk[d::DOF] = True
    sym = np.zeros(size//DOF, dtype=bool)
    for q in range(4):
        sym |= np.isclose(y, q*pi*R/2)
    bk[3::DOF] |= sym
    bk[8::DOF] |= sym
    return mod, elements, points, weights, size, ~bk, prop


def buckling(KC0, KG, KCf, bu, load):
    K = KG + KCf if load == 'follower' else KG
    Kuu = KC0[bu, :][:, bu]
    KGuu = K[bu, :][:, bu]
    eigvals, eigvecs = lb(Kuu, KGuu, silent=True, num_eigvalues=4,
                          symmetric=False if load == 'follower' else None)
    return np.real(eigvals[0]), eigvecs[:, 0]


def ring_buckling_pressure(kinematics, load, R=1., h=0.01, ny=192):
    r"""Load multiplier of a unit external pressure on the ring"""
    mod, elements, points, weights, size, bu, prop = ring(kinematics, R, h,
                                                          ny)
    p = -1.
    KC0, _, KCf, f = matrices(mod, elements, points, weights, size, p,
                              u0=np.zeros(size))
    u0 = np.zeros(size)
    u0[bu] = spsolve(KC0[bu, :][:, bu], f[bu])
    _, KG, _, _ = matrices(mod, elements, points, weights, size, p, u0=u0)
    lam, mode = buckling(KC0, KG, KCf, bu, load)
    return lam, prop


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
@pytest.mark.parametrize('load', ['follower', 'dead'])
def test_isotropic_ring(kinematics, load):
    R = 1.
    lam, prop = ring_buckling_pressure(kinematics, load, R=R)
    A22 = prop.A[1, 1]
    D22 = prop.D[1, 1]
    ref = -ring_closed_form(kinematics, load, 2, A22, D22, R)
    # discretization error for n = 2, relative: 7e-3, 1.5e-4, 2.9e-6 and
    # 7.9e-8 (Sanders), 3.9e-3, 8.3e-5, 1.7e-6 and 4.5e-8 (Donnell) with 24,
    # 48, 96 and 192 elements around the circumference, see the module
    # __main__; the convergence is that of the inextensional mode, which
    # the cubic v and w only approximate
    assert np.isclose(lam, ref, rtol=2e-7), (lam, ref)
    inextensional = {('sanders', 'follower'): 3., ('sanders', 'dead'): 4.,
                     ('donnell', 'follower'): 16/5, ('donnell', 'dead'): 4.}
    # R/h = 100: the extensibility is of relative order (h/R)**2
    assert np.isclose(lam*R**3/D22, inextensional[(kinematics, load)],
                      rtol=1e-3)


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
def test_follower_load_stiffness_is_symmetric_for_the_ring(kinematics):
    r"""Uniform pressure on a closed ring with `u = 0` at the ends: the
    contour term of the skew part vanishes, see the module Notes"""
    mod, elements, points, weights, size, bu, prop = ring(kinematics, 1.,
                                                          0.01, 16)
    _, _, KCf, _ = matrices(mod, elements, points, weights, size, -1.,
                            u0=np.zeros(size))
    K = KCf[bu, :][:, bu].toarray()
    assert np.linalg.norm(K) > 0
    assert np.linalg.norm(K - K.T) <= 1e-13*np.linalg.norm(K)
    # without the constraint of u at the ends the matrix is unsymmetric
    Kfull = KCf.toarray()
    assert np.linalg.norm(Kfull - Kfull.T) > 1e-3*np.linalg.norm(Kfull)


def test_rigid_rotation_of_ring_is_neutral_only_for_follower():
    r"""Sanders-type kinematics: the rigid rotation ``v = const`` of a ring
    has no strain energy, the energy ``int Nyy (v/R)**2 dA < 0`` in KG under
    an external pressure, and ``-int p v**2/R dA`` in KCfollower, which
    cancels it, such that only the follower pressure leaves the rotation
    neutral"""
    mod, elements, points, weights, size, bu, prop = ring('sanders', 1.,
                                                          0.01, 16)
    p = -1.
    KC0, _, KCf, f = matrices(mod, elements, points, weights, size, p,
                              u0=np.zeros(size))
    u0 = np.zeros(size)
    u0[bu] = spsolve(KC0[bu, :][:, bu], f[bu])
    _, KG, _, _ = matrices(mod, elements, points, weights, size, p, u0=u0)
    d = np.zeros(size)
    d[3::DOF] = 1.
    eG = d @ (KG @ d)
    assert eG < 0
    assert abs(d @ (KC0 @ d)) <= 1e-12*np.abs(KC0).max()*size
    assert abs(d @ ((KG + KCf) @ d)) <= 1e-10*abs(eG)


def cylinder_closed_form(kinematics, load, m, n, L, R, ABD, Nxx, Nyy, p):
    r"""Load multiplier of a cylinder with shear-diaphragm ends

    Mode ``u = U cos(l x) cos(n t)``, ``v = V sin(l x) sin(n t)``, ``w = W
    sin(l x) cos(n t)``, ``t = y/R``, ``l = m pi/L``, under the membrane
    prestate ``lam (Nxx, Nyy)`` and the pressure ``lam p``, a ``'follower'``
    pressure with the first-order area vector or a ``'dead'`` one, with the
    strains of the Notes of the two element modules. The fields are
    evaluated at quadrature points that integrate the trigonometric products
    exactly, and ``ABD`` must be orthotropic (no 16, 26 or B terms). Returns
    the smallest positive ``lam``.

    """
    l = m*np.pi/L
    k = n/R
    kv = 1/R if kinematics == 'sanders' else 0.
    xg, wg = np.polynomial.legendre.leggauss(2*m + 8)
    xs = (xg + 1)*L/2
    nt = 4*n + 8
    ts = np.arange(nt)*2*np.pi/nt
    X, T = np.meshgrid(xs, ts, indexing='ij')
    W8 = np.outer(wg*L/2, np.full(nt, 2*np.pi*R/nt))
    cx, sx, cn, sn = np.cos(l*X), np.sin(l*X), np.cos(n*T), np.sin(n*T)
    zero = np.zeros_like(X)
    names = ('u', 'u_x', 'u_y', 'v', 'v_x', 'v_y', 'w', 'w_x', 'w_y', 'w_xx',
             'w_yy', 'w_xy')
    F = [dict.fromkeys(names, zero) for _ in range(3)]
    F[0].update(u=cx*cn, u_x=-l*sx*cn, u_y=-k*cx*sn)
    F[1].update(v=sx*sn, v_x=l*cx*sn, v_y=k*sx*cn)
    F[2].update(w=sx*cn, w_x=l*cx*cn, w_y=-k*sx*sn, w_xx=-l**2*sx*cn,
                w_yy=-k**2*sx*cn, w_xy=-l*k*cx*sn)

    def strains(f):
        kyy = -f['w_yy']
        kxy = -2*f['w_xy']
        if kinematics == 'sanders':
            kyy = kyy + f['v_y']/R
            kxy = kxy + 1.5*f['v_x']/R - 0.5*f['u_y']/R
        return np.array([f['u_x'], f['v_y'] + f['w']/R, f['u_y'] + f['v_x'],
                         -f['w_xx'], kyy, kxy])

    Es = [strains(f) for f in F]
    H0 = np.zeros((3, 3))
    H1 = np.zeros((3, 3))
    for i, fi in enumerate(F):
        for j, fj in enumerate(F):
            H0[i, j] = np.sum(W8*np.einsum('a...,ab,b...->...', Es[i], ABD,
                                           Es[j]))
            byi = fi['w_y'] - kv*fi['v']
            byj = fj['w_y'] - kv*fj['v']
            H1[i, j] = np.sum(W8*(Nxx*fi['w_x']*fj['w_x'] + Nyy*byi*byj))
            if load == 'follower':
                a1 = (-fj['w_x'], -fj['w_y'] + kv*fj['v'],
                      fj['u_x'] + fj['v_y'] + fj['w']/R)
                H1[i, j] -= np.sum(W8*p*(fi['u']*a1[0] + fi['v']*a1[1]
                                         + fi['w']*a1[2]))
    lam = eig(H0, -H1, right=False)
    lam = lam[np.isfinite(lam)]
    lam = lam[np.abs(lam.imag) <= 1e-8*np.abs(lam)].real
    return lam[lam > 0].min()


def cylinder_buckling(kinematics, prestate, L=1.5, R=1., h=0.002, nx=13,
                      ny=96):
    r"""Load multipliers of the cylinder with shear-diaphragm ends, ``v = w
    = 0`` at ``x = 0`` and ``x = L``, and ``u = 0`` at one node, for the
    follower and for the dead pressure"""
    prop = isotropic_plate(thickness=h, E=E, nu=NU)
    out = cylinder_model(kinematics, L, R, prop, nx, ny)
    mod, elements, points, weights, x, y, size = out
    p = -1.
    Nxx = 0. if prestate == 'lateral' else p*R/2
    Nyy = p*R
    KC0, KG, KCf, _ = matrices(mod, elements, points, weights, size, p,
                               Nxx=Nxx, Nyy=Nyy)
    bk = np.zeros(size, dtype=bool)
    ends = np.isclose(x, 0) | np.isclose(x, L)
    # v = w = 0 along the ends, hence also v,y = w,y = 0
    for d in (3, 5, 6, 8):
        bk[d::DOF] |= ends
    bk[0] = True
    bu = ~bk
    lam = {}
    waves = {}
    for load in ('follower', 'dead'):
        lam[load], mode = buckling(KC0, KG, KCf, bu, load)
        phi = np.zeros(size)
        phi[bu] = np.real(mode)
        # circumferential waves of w at the mid-length, nx odd
        w = phi[6::DOF][np.isclose(x, L/2)]
        waves[load] = int(np.argmax(np.abs(np.fft.rfft(w))))
    # closed form for each n
    ref = {load: {n: cylinder_closed_form(kinematics, load, 1, n, L, R,
                                          prop.ABD, Nxx, Nyy, p)
                  for n in range(2, 16)}
           for load in ('follower', 'dead')}
    return lam, waves, ref


@pytest.mark.parametrize('kinematics,prestate', [
    ('sanders', 'lateral'),
    ('sanders', 'hydrostatic'),
    ('donnell', 'lateral'),
    ])
def test_finite_cylinder_shear_diaphragm(kinematics, prestate):
    r"""Finite cylinder, ``L/R = 1.5``, ``R/h = 500``, under a lateral
    (``Nxx = 0``) or hydrostatic (``Nxx = p R/2``, the end caps as a dead
    axial load) external pressure, with the membrane prestate given by
    ``Nxx`` and ``Nyy = p R`` and the pressure as a follower or dead load.

    The load multipliers of the 12 x 96 mesh converge as `O(h^2)`, with
    relative errors of 8.5e-2, 1.5e-2 and 3.7e-3 for the meshes 6 x 48, 12 x
    96 and 24 x 192 (see the module ``__main__``), the Richardson
    extrapolation of the last two agreeing with the closed form within 7e-4;
    they are checked within 2.5e-2. The discretization error is the same for
    the follower and for the dead pressure, whose modes are the same, and
    the ratio of the two load multipliers, which isolates the effect of the
    follower load stiffness, is checked within 1e-4 against the ratio of
    the closed forms of the number of circumferential waves ``n`` of the
    finite element modes, the same for both loads. For the Sanders-type
    kinematics under the lateral pressure the critical ``n`` of the closed
    form is 10 for the follower and 11 for the dead pressure, the load
    multipliers of ``n = 10`` and ``11`` differing by 6e-4 and 1e-3, less
    than the discretization error, and the modes of the mesh have ``n =
    11`` for both loads."""
    lam, waves, ref = cylinder_buckling(kinematics, prestate)
    for load in ('follower', 'dead'):
        assert np.isclose(lam[load], min(ref[load].values()), rtol=2.5e-2), (
            load, lam, ref)
    n = waves['follower']
    assert waves['dead'] == n
    assert n in (10, 11)
    ratio = lam['follower']/lam['dead']
    ratio_ref = ref['follower'][n]/ref['dead'][n]
    assert ratio < 1
    assert np.isclose(ratio, ratio_ref, rtol=1e-4), (ratio, ratio_ref)


if __name__ == '__main__':
    R = 1.
    for kin in sorted(KINEMATICS):
        for load in ('follower', 'dead'):
            prop = isotropic_plate(thickness=0.01, E=E, nu=NU)
            ref = -ring_closed_form(kin, load, 2, prop.A[1, 1], prop.D[1, 1],
                                    R)
            for ny in (24, 48, 96, 192):
                lam, _ = ring_buckling_pressure(kin, load, ny=ny)
                print('ring %s %s ny=%d lam=%.12e ref=%.12e relerr=%.2e'
                      % (kin, load, ny, lam, ref, lam/ref - 1))
    for kin, prestate in (('sanders', 'lateral'), ('sanders', 'hydrostatic'),
                          ('donnell', 'lateral')):
        for nx, ny in ((7, 48), (13, 96), (25, 192)):
            lam, waves, ref = cylinder_buckling(kin, prestate, nx=nx,
                                                ny=ny)
            print('    waves of the modes', waves)
            ref = {k: min(v.values()) for k, v in ref.items()}
            for load in ('follower', 'dead'):
                print('cylinder %s %s %s %dx%d lam=%.8e ref=%.8e relerr=%.2e'
                      % (kin, load, prestate, nx - 1, ny, lam[load],
                         ref[load], lam[load]/ref[load] - 1))
            print('    ratio follower/dead %.8f ref %.8f relerr %.2e'
                  % (lam['follower']/lam['dead'], ref['follower']/ref['dead'],
                     lam['follower']/lam['dead']
                     /(ref['follower']/ref['dead']) - 1))
