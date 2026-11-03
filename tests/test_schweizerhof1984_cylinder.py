r"""Cantilever cylinder under a follower pressure, Schweizerhof and Ramm (1984)

Shell 1 of Section 6.4 of Schweizerhof and Ramm (1984), ``L/R = 6``, ``R/t =
100``, lower edge hinged (B1), upper edge free, under a uniform external
pressure, Table 4 on p. 1111 and Fig. 13 on p. 1112, with the 60 deg sector
of the paper meshed with the Sanders-type element, see the module
``doc/source/tutorials/schweizerhof1984_cylinder.py`` and the tutorial
``schweizerhof1984_cylinder_follower.ipynb``. About 20 s.

The checks:

- the vectorized tangent stiffness matrix of the module is that of the
  kernels ``update_KC0 + update_KCNL + update_KG`` to the round-off; the
  callables of the imperfect shell under the follower pressure have the
  Jacobian of their residual as tangent stiffness matrix, and the internal
  forces are the gradient of a strain energy computed independently of the
  kernels (central differences, errors of order ``h**2``); the load
  stiffness matrix is unsymmetric, the free edge making the pressure
  nonconservative;
- the linear buckling factors of the dead (``k_c``) and of the follower
  (``k_f``) pressure, with the full unsymmetric load stiffness matrix, are
  those of Table 4, 10.88 and 9.70, within 0.5 %, with ``n = 3`` waves. The
  values of 64 x 8 elements, 10.901 and 9.705, are within 0.2 % of those
  of 128 x 8 elements, 10.887 and 9.693, and the paper gives its 24
  elements as converged, so that the tolerance covers both
  discretizations;
- the Riks method with ``inertia`` and ``inertia_num_eigvals`` (the tangent
  stiffness matrix is unsymmetric, the count comes from ARPACK on its LU
  factorization) follows the imperfect shell, imperfection of 1 % of
  ``t``, through the flat maximum of the load, at a largest ``|w|`` of
  about ``4 t``: the count of negative eigenvalues is zero up to the
  maximum and one after it, with the sign of the determinant, and the
  maximum is below the linear ``k_f`` of the same mesh by less than 1 %
  (0.4 % with 32 x 8 elements, whose ``k_f`` is 0.5 % above the converged
  one). The residual of every converged increment is that of the whole
  follower pressure. Fig. 13 of the paper, labelled B1, has a plateau at
  ``K = 8.0``, which is that of B2 (see the tutorial), and is not compared
  here.

"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'doc', 'source', 'tutorials'))
import schweizerhof1984_cylinder as sr # noqa: E402


def test_identities():
    rng = np.random.default_rng(1)
    for kinematics in ('sanders', 'donnell'):
        model = sr.CantileverModel(1, 'B1', 6, 3, kinematics)
        n = model.bu.sum()
        u = model.expand(rng.standard_normal(n)*0.1)
        KT = model.calc_KT(u)
        KTk = model.KC0 + model.calc_KCNL(u) + model.calc_KG(u)
        assert abs(KT - KTk).max() < 1e-13*abs(KTk).max()
        KG = model.calc_KG(u).toarray()
        KGN = model.calc_kG_N(model.membrane_resultants(u, nonlinear=False))
        assert np.abs(KGN.toarray() - KG).max() < 1e-12*np.abs(KG).max()

        eigvals, modes, _ = sr.linear_buckling(model)
        u0 = sr.imperfection(model, modes[:, 0], 0.3)
        assert np.isclose(np.abs(u0[6::10]).max(), 0.3*model.t)
        cb = sr.FollowerCylinder(model, u0)
        dead = sr.FollowerCylinder(model, u0, 'dead')
        assert np.linalg.norm(cb.calc_fint(np.zeros(n), inc=5.)) == 0
        u = np.zeros(model.N)
        u[6::10] = rng.standard_normal(model.N//10)
        u *= 0.3*model.t/np.abs(u).max()
        u[0::10] = 1e-3*rng.standard_normal(model.N//10)
        u[3::10] = 1e-3*rng.standard_normal(model.N//10)
        c = u[model.bu]
        v = rng.standard_normal(n)
        v /= np.abs(v).max()
        h = 1e-6
        inc = 5.
        KT = cb.calc_kC(c, NLgeom=True, inc=inc) + cb.calc_kG(c, NLgeom=True)
        KTk = (sr.FollowerCylinder(model, u0, vectorized=False).calc_kC(
               c, NLgeom=True, inc=inc) + cb.calc_kG(c, NLgeom=True))
        assert abs(KT - KTk).max() < 1e-13*abs(KTk).max()
        dfd = (cb.calc_fint(c + h*v, inc=inc)
               - cb.calc_fint(c - h*v, inc=inc))/(2*h)
        assert np.linalg.norm(dfd - KT @ v) < 1e-8*np.linalg.norm(KT @ v)
        dU = (model.strain_energy(model.expand(c + h*v), u0)
              - model.strain_energy(model.expand(c - h*v), u0))/(2*h)
        fv = dead.calc_fint(c) @ v
        assert abs(dU - fv) < 1e-7*abs(fv)
        # the residual of structsolve is that of the follower pressure on
        # the imperfect shell
        r = cb.calc_fint(c, inc=inc) - cb.calc_fext(inc)
        rf = dead.calc_fint(c) - inc*model.calc_fp(model.expand(c) + u0)[
            model.bu]
        assert np.linalg.norm(r - rf) < 1e-12*np.linalg.norm(rf)
        # nonconservative: the free edge makes KF unsymmetric
        KF = model.restrict(model.KF).toarray()
        assert np.abs(KF - KF.T).max() > 1e-3*np.abs(KF).max()


def test_linear_buckling_table4():
    model = sr.CantileverModel(1, 'B1', 64, 8)
    kc, modes_c, _ = sr.linear_buckling(model, 'dead')
    kf, modes_f, _ = sr.linear_buckling(model, 'follower')
    k_c, (k_f,) = sr.TABLE4[1, 'B1']
    print('k_c %.4f, k_f %.4f' % (kc[0], kf[0]))
    assert abs(kc[0]/k_c - 1) < 0.005
    assert abs(kf[0]/k_f - 1) < 0.005
    assert sr.num_waves(model, modes_c[:, 0]) == 3
    assert sr.num_waves(model, modes_f[:, 0]) == 3


def test_riks_follower_beyond_maximum():
    r = sr.run_riks(1, 'B1', 32, 8, wmax=4.6)
    an = r['an']
    ks, neg, det = r['ks'], np.array(r['neg']), np.array(r['det'])
    fext = an.calc_fext()
    for k, c in zip(an.increments, an.cs):
        # the residual of the follower pressure, relative to the load
        assert (np.linalg.norm(an.calc_fint(c, inc=k) - k*fext)
                < 1e-7*np.linalg.norm(k*fext))
    imax = np.argmax(ks)
    print('k_lin %.4f, maximum %.4f at w/t %.3f, %d increments, %.1f s'
          % (r['k_lin'], ks[imax], r['wt'][imax], len(ks) - 1, r['elapsed']))
    # a maximum inside the path, below the linear k_f
    assert 0 < imax < len(ks) - 1
    assert 0.99*r['k_lin'] < ks[imax] < r['k_lin']
    # increment i of an is ks[i + 1]
    assert np.all(neg[:imax] == 0) and np.all(det[:imax] == 1)
    assert neg[-1] == 1 and det[-1] == -1
    assert 3.5 < r['wt'][imax] < 4.5
    assert r['wt'][-1] >= 4.6
