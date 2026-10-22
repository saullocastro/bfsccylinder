r"""Follower pressure solved with structsolve.Analysis

A cylinder clamped at `x = 0` and free at `x = L`, under an external
follower pressure varying around the circumference, solved with the full Newton-Raphson method of
structsolve for both kinematics. The free end makes the load stiffness
matrix unsymmetric. The callables of ``FollowerCallables`` receive the load
factor ``inc``, which structsolve passes to the callables that declare it:

- ``calc_fext`` gives the dead part `\lambda \{F_p(0)\}` of the pressure;
- ``calc_fint`` subtracts the part that depends on the displacements,
  `\lambda (\{F_p(u)\} - \{F_p(0)\})`, of ``update_fext_follower`` with
  ``reference=0``, such that the residual of structsolve is that of the
  follower pressure, `\lambda \{F_p(u)\} - \{f_{int}(u)\}`;
- ``calc_kC`` adds the load stiffness matrix of the pressure `\lambda p`.

The checks: the solution is in equilibrium with the whole follower
pressure, the load stiffness matrix is unsymmetric, and the Newton-Raphson
iterations converge quadratically with it, which only happens when the
unsymmetric tangent stiffness matrix is the Jacobian of the residual.

Units: N, m, Pa.
"""
import sys
sys.path.append(r'..')

import numpy as np
import pytest
from numpy import isclose
from scipy.sparse import coo_matrix
from composites import isotropic_plate
from structsolve import Analysis
from structsolve.convergence import crisfield_weights, crisfield_zeta

from test_follower_pressure_validation import (KINEMATICS, E, NU,
        cylinder_model, assemble)


class FollowerCallables(object):
    r"""The callables of structsolve.Analysis for the ``elements`` of the
    module ``mod`` under the follower pressures ``pressures``, one array of
    shape ``(nint, nint)`` per element, times the load factor

    They loop over the elements with the update functions and assemble, and
    work on the unknown degrees of freedom ``bu`` only, the partition of the
    free dofs of structsolve: the solution vector of the solvers is ``c =
    u[bu]``, and the force vectors and matrices are restricted to ``bu``.
    ``quadratic`` is the truncation of the area vector. ``calc_fint`` also
    keeps the load factor and the internal forces of each iteration, for the
    residual history.
    """
    def __init__(self, mod, elements, points, weights, N, bu, pressures,
                 quadratic=0):
        self.mod = mod
        self.elements = elements
        self.points = points
        self.weights = weights
        self.N = N
        self.bu = bu
        self.pressures = pressures
        self.quadratic = quadratic
        self.iterations = []
        self.KC0 = assemble(mod, elements, mod.update_KC0, 'KC0', N,
                            (points, weights))
        num_elements = len(elements)
        for i, shell in enumerate(elements):
            shell.init_k_KCNL = i*mod.KCNL_SPARSE_SIZE
            shell.init_k_KG = i*mod.KG_SPARSE_SIZE
            shell.init_k_KCfollower = i*mod.KCfollower_SPARSE_SIZE
        size = mod.KCNL_SPARSE_SIZE*num_elements
        self.KCNLr = np.zeros(size, dtype=mod.INT)
        self.KCNLc = np.zeros(size, dtype=mod.INT)
        self.KCNLv = np.zeros(size, dtype=mod.DOUBLE)
        size = mod.KG_SPARSE_SIZE*num_elements
        self.KGr = np.zeros(size, dtype=mod.INT)
        self.KGc = np.zeros(size, dtype=mod.INT)
        self.KGv = np.zeros(size, dtype=mod.DOUBLE)
        size = mod.KCfollower_SPARSE_SIZE*num_elements
        self.KCfr = np.zeros(size, dtype=mod.INT)
        self.KCfc = np.zeros(size, dtype=mod.INT)
        self.KCfv = np.zeros(size, dtype=mod.DOUBLE)
        # the dead part of the pressure, the force vector at u = 0
        self.fp0 = np.zeros(N, dtype=mod.DOUBLE)
        for shell, p in zip(elements, pressures):
            mod.update_fext_follower(np.zeros(N), shell, points, weights, p,
                                     self.fp0, quadratic)

    def expand(self, c):
        """The displacements of the model from those of the unknown dofs."""
        u = np.zeros(self.N, dtype=self.mod.DOUBLE)
        u[self.bu] = c
        return u

    def calc_fext(self, inc=1., silent=False):
        return inc*self.fp0[self.bu]

    def calc_fint(self, c, inc=1., silent=False):
        mod = self.mod
        u = self.expand(c)
        fint = np.zeros(self.N, dtype=mod.DOUBLE)
        for shell, p in zip(self.elements, self.pressures):
            mod.update_fint(u, shell, self.points, self.weights, fint)
            # minus the part of the follower pressure that depends on u
            mod.update_fext_follower(u, shell, self.points, self.weights,
                                     -inc*p, fint, self.quadratic, 0)
        self.iterations.append((inc, fint[self.bu]))
        return fint[self.bu]

    def calc_kC(self, c=None, NLgeom=False, inc=1., silent=False):
        bu = self.bu
        if not NLgeom:
            return self.KC0[bu, :][:, bu]
        mod = self.mod
        u = self.expand(c)
        self.KCNLv *= 0
        self.KCfv *= 0
        for shell, p in zip(self.elements, self.pressures):
            mod.update_KCNL(u, shell, self.points, self.weights, self.KCNLr,
                            self.KCNLc, self.KCNLv)
            # the load stiffness matrix of the pressure inc*p
            mod.update_KCfollower(u, shell, self.points, self.weights,
                                  inc*p, self.KCfr, self.KCfc, self.KCfv,
                                  self.quadratic)
        KCNL = coo_matrix((self.KCNLv, (self.KCNLr, self.KCNLc)),
                          shape=(self.N, self.N))
        KCf = coo_matrix((self.KCfv, (self.KCfr, self.KCfc)),
                         shape=(self.N, self.N))
        return (self.KC0 + KCNL + KCf).tocsc()[bu, :][:, bu]

    def calc_kG(self, c=None, NLgeom=False, silent=False):
        mod = self.mod
        u = self.expand(c)
        self.KGv *= 0
        for shell in self.elements:
            mod.update_KG(u, shell, self.points, self.weights, self.KGr,
                          self.KGc, self.KGv)
        KG = coo_matrix((self.KGv, (self.KGr, self.KGc)),
                        shape=(self.N, self.N))
        return KG.tocsc()[self.bu, :][:, self.bu]


@pytest.mark.parametrize('kinematics', sorted(KINEMATICS))
def test_clamped_free_cylinder(kinematics):
    r"""Cylinder of radius 1 m, length 1 m and thickness 10 mm, clamped at
    `x = 0` and free at `x = L`, under the external follower pressure `p =
    -2 (1 + \cos(3 y/R)/2)` bar, in two equal load increments"""
    R, L, h = 1., 1., 0.01
    prop = isotropic_plate(thickness=h, E=E, nu=NU)
    mod, elements, points, weights, x, y, N = cylinder_model(kinematics, L,
            R, prop, nx=7, ny=24)
    DOF = mod.DOF
    bk = np.zeros(N, dtype=bool)
    for i in range(DOF):
        bk[i::DOF] |= isclose(x, 0)
    bu = ~bk
    nint = len(points)
    # p[i, j] at xi = points[i] and eta = points[j], from the circumferential
    # coordinate of node 1 of each element
    pressures = []
    for shell in elements:
        yj = y[shell.n1] + shell.ley*(points + 1)/2
        p = -2.e5*(1 + 0.5*np.cos(3*yj/R))
        pressures.append(np.ascontiguousarray(np.ones((nint, 1))*p[None, :]))

    callables = FollowerCallables(mod, elements, points, weights, N, bu,
                                  pressures)
    an = Analysis(calc_fext=callables.calc_fext,
                  calc_fint=callables.calc_fint,
                  calc_kC=callables.calc_kC,
                  calc_kG=callables.calc_kG)
    an.initialInc = 0.5
    an.maxInc = 0.5
    increments, cs = an.static(NLgeom=True)
    # every increment converged, none was cut
    assert np.allclose(increments, [0.5, 1.]), increments
    u = callables.expand(cs[-1])
    print('max |w|/h', np.abs(u[6::DOF]).max()/h)

    # equilibrium with the whole follower pressure, F_p(u) of the pressure p
    fint = np.zeros(N)
    fp = np.zeros(N)
    for shell, p in zip(elements, pressures):
        mod.update_fint(u, shell, points, weights, fint)
        mod.update_fext_follower(u, shell, points, weights, p, fp)
    residual = np.linalg.norm((fint - fp)[bu])/np.linalg.norm(fp[bu])
    print('relative residual', residual)
    assert residual < 1.e-8
    # the follower part of the load is not negligible
    assert np.linalg.norm((fp - callables.fp0)[bu]) > 1.e-3*np.linalg.norm(
            callables.fp0[bu])

    # the free end makes the load stiffness matrix unsymmetric
    KC = callables.calc_kC(cs[-1], NLgeom=True, inc=1.)
    KCnl = callables.calc_kC(cs[-1], NLgeom=True, inc=0.)
    KCf = (KC - KCnl).toarray()
    asym = np.abs(KCf - KCf.T).max()/np.abs(KCf).max()
    print('asymmetry of the load stiffness matrix', asym)
    assert asym > 1.e-3

    # quadratic convergence in each increment: Crisfield's zeta, the
    # criterion of structsolve, is squared by the last iteration, down to
    # the round-off
    w = crisfield_weights(callables.KC0[bu, :][:, bu])
    f = callables.fp0[bu]
    for lbd in increments:
        history = [crisfield_zeta(inc*f - fi, inc*f, fi, w)
                   for inc, fi in callables.iterations if inc == lbd]
        print('load factor', lbd, 'zeta', history)
        assert history[-1] < 1.e-9
        assert len(history) <= 6
        prev, curr = history[-2], history[-1]
        assert curr < max(10.*prev**2, 1.e-13), history


if __name__ == '__main__':
    for kinematics in sorted(KINEMATICS):
        test_clamped_free_cylinder(kinematics)
