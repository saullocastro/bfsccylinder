r"""The mass matrix against an independent quadrature of the kinetic energy

The kinetic energy of the Kirchhoff-Love field of each kinematics, for a
density ``rho`` uniform over the thickness ``h``, the mid-surface as the
reference surface and without the factor ``1 + z/R`` of the area (shallow
shell),

    u(z) = u - z w_x,   v(z) = v + z beta_y,   w(z) = w

with ``beta_y = -w_y`` for the Donnell-type and ``beta_y = -w_y + v/R`` for
the Sanders-type kinematics, gives

    M = int rho (h (Su'Su + Sv'Sv + Sw'Sw)
                 + h**3/12 (Sw_x'Sw_x + (Sw_y - k Sv/R)'(Sw_y - k Sv/R))) dA

with ``k = 0`` (Donnell) or ``1`` (Sanders). The rows are built here from
cubic Hermite polynomials written from scratch, and the integral with a
Gauss-Legendre rule of numpy, exact for the polynomial integrand.
"""
import sys
sys.path.append(r'..')

import numpy as np
import pytest

import bfsccylinder
import bfsccylinder.sanders

DOF = 10
N = 4*DOF
NODES = [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]


def hermite(xi, xii, le):
    r"""Cubic Hermite functions of the end ``xii`` of ``[-1, 1]``, and their
    derivative in the physical coordinate, the slope function scaled by the
    length ``le``"""
    t = (xi + 1)/2
    if xii < 0:
        f0 = (2*t**3 - 3*t**2 + 1, (6*t**2 - 6*t)/le)
        f1 = (le*(t**3 - 2*t**2 + t), 3*t**2 - 4*t + 1)
    else:
        f0 = (-2*t**3 + 3*t**2, (-6*t**2 + 6*t)/le)
        f1 = (le*(t**3 - t**2), 3*t**2 - 2*t)
    return f0, f1


def rows(xi, eta, lex, ley):
    r"""Rows of ``u``, ``v``, ``w`` and of their first derivatives"""
    out = {k: np.zeros(N) for k in ('u', 'u_x', 'u_y', 'v', 'v_x', 'v_y',
                                    'w', 'w_x', 'w_y')}
    for i, (xii, etai) in enumerate(NODES):
        F0, F1 = hermite(xi, xii, lex)
        G0, G1 = hermite(eta, etai, ley)
        c = DOF*i
        for off, fld in ((0, 'u'), (3, 'v')):
            for j, (F, G) in enumerate(((F0, G0), (F1, G0), (F0, G1))):
                out[fld][c + off + j] = F[0]*G[0]
                out[fld + '_x'][c + off + j] = F[1]*G[0]
                out[fld + '_y'][c + off + j] = F[0]*G[1]
        for j, (F, G) in enumerate(((F0, G0), (F1, G0), (F0, G1), (F1, G1))):
            out['w'][c + 6 + j] = F[0]*G[0]
            out['w_x'][c + 6 + j] = F[1]*G[0]
            out['w_y'][c + 6 + j] = F[0]*G[1]
    return out


def mass_reference(sanders, lex, ley, R, rho, h, nint=6):
    xs, ws = np.polynomial.legendre.leggauss(nint)
    M = np.zeros((N, N))
    for xi, wx in zip(xs, ws):
        for eta, wy in zip(xs, ws):
            S = rows(xi, eta, lex, ley)
            wgt = wx*wy*lex*ley/4
            bx = -S['w_x']
            by = -S['w_y'] + (S['v']/R if sanders else 0.)
            M += wgt*rho*(h*(np.outer(S['u'], S['u'])
                             + np.outer(S['v'], S['v'])
                             + np.outer(S['w'], S['w']))
                          + h**3/12*(np.outer(bx, bx) + np.outer(by, by)))
    return M


KINEMATICS = {
    'donnell': (bfsccylinder, 'BFSCCylinder'),
    'sanders': (bfsccylinder.sanders, 'BFSCCylinderSanders'),
}


def element_mass(kinematics, lex, ley, R, rho, h):
    mod, cls = KINEMATICS[kinematics]
    elem = getattr(mod, cls)(4)
    elem.c1, elem.c2, elem.c3, elem.c4 = 0, DOF, 2*DOF, 3*DOF
    elem.lex, elem.ley, elem.R, elem.rho, elem.h = lex, ley, R, rho, h
    elem.init_k_M = 0
    Mr = np.zeros(mod.M_SPARSE_SIZE, dtype=mod.INT)
    Mc = np.zeros(mod.M_SPARSE_SIZE, dtype=mod.INT)
    Mv = np.zeros(mod.M_SPARSE_SIZE, dtype=mod.DOUBLE)
    mod.update_M(elem, Mr, Mc, Mv)
    M = np.zeros((N, N))
    np.add.at(M, (Mr, Mc), Mv)
    # every entry written once
    assert len(set(zip(Mr.tolist(), Mc.tolist()))) == mod.M_SPARSE_SIZE
    return M


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
@pytest.mark.parametrize('lex, ley, R, h', [(0.3, 0.2, 0.7, 0.004),
                                            (0.05, 0.08, 0.1, 0.02)])
def test_mass_matrix_is_that_of_the_kinetic_energy(kinematics, lex, ley, R,
                                                   h):
    rho = 1500.
    M = element_mass(kinematics, lex, ley, R, rho, h)
    ref = mass_reference(kinematics == 'sanders', lex, ley, R, rho, h)
    err = np.linalg.norm(M - ref)/np.linalg.norm(ref)
    assert err < 1e-12, err
    # the v-w coupling of the rotary inertia, relative to the rest of M
    vw = np.ix_([i for i in range(N) if 3 <= i % DOF <= 5],
                [i for i in range(N) if i % DOF >= 6])
    coupling = np.abs(M[vw]).max()
    if kinematics == 'sanders':
        assert coupling > 0
        assert np.abs(M[vw] - ref[vw]).max() <= 1e-12*coupling
    else:
        assert coupling == 0
