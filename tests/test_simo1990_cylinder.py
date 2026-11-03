r"""Post-buckling of an axially compressed cylinder with an imperfection

The ``L/10 x 15 deg`` section of the cylinder of Simo, Fox and Rifai (1990),
Section 6.3.4, pp. 54-57, Fig. 6.3.4 on p. 56, meshed with the Donnell-type
element, with an initial geometric imperfection added exactly by the model
of ``doc/source/tutorials/simo1990_cylinder.py``, see the tutorial
``simo1990_cylinder_postbuckling.ipynb``. Coarse meshes, about 10 s.

The checks:

- the identities of the imperfect shell: the internal forces vanish at the
  imperfect state, the tangent stiffness matrix is their Jacobian, and the
  internal forces are the gradient of a strain energy of the strains
  ``eps(u + u0) - eps(u0)`` computed independently of the kernels of the
  element, which fails with the geometric stiffness matrix of
  :func:`.update_KG` alone, of the resultants of the linear strains;
- the Riks method of structsolve passes the limit point of the cylinder
  with an imperfection of 10 % of the thickness, at the limit load of Fig.
  6.3.4(a). The figure is read to about +-0.01 in the load and +-0.0003 in
  the axial deflection; the converged limit loads of this model, 0.727 and
  0.391 for 10 % and 50 %, differ from the curves of Simo et al., 0.715 and
  0.411, by up to 0.02, more than the reading precision, which is the
  difference between the Donnell-type kinematics and the geometrically
  exact shell of the paper with its 10 x 10 mesh. The tolerances are 0.025
  in the load and 0.001 in the deflection.

"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'doc', 'source', 'tutorials'))
import simo1990_cylinder as simo # noqa: E402


def test_imperfection_identities():
    rng = np.random.default_rng(1)
    model, mode, lbd_cr = simo.normalized_model(4, 4)
    assert 1.0 < lbd_cr < 1.02
    # the geometric stiffness matrix of the resultants given at the
    # integration points reproduces update_KG for the linear resultants
    u = model.expand(rng.standard_normal(model.bu.sum())*1e-2)
    KG = model.calc_KG(u).toarray()
    KGN = model.calc_kG_N(model.membrane_resultants(u, nonlinear=False))
    assert np.abs(KGN.toarray() - KG).max() < 1e-12*np.abs(KG).max()
    for xi in (0.1, 0.5):
        u0 = simo.imperfection(model, mode, xi)
        assert np.isclose(np.abs(u0[6::10]).max(), xi*simo.t)
        imp = simo.ImperfectCylinder(model, u0)
        n = model.bu.sum()
        assert np.linalg.norm(imp.calc_fint(np.zeros(n))) == 0
        u = np.zeros(model.N)
        u[6::10] = rng.standard_normal(model.N//10)
        u *= 0.3*simo.t/np.abs(u).max()
        u[0::10] = 1e-3*rng.standard_normal(model.N//10)
        u[3::10] = 1e-3*rng.standard_normal(model.N//10)
        c = u[model.bu]
        v = rng.standard_normal(n)
        h = 1e-5
        KT = imp.calc_kC(c, NLgeom=True) + imp.calc_kG(c, NLgeom=True)
        dfd = (imp.calc_fint(c + h*v) - imp.calc_fint(c - h*v))/(2*h)
        assert np.linalg.norm(dfd - KT @ v) < 1e-8*np.linalg.norm(KT @ v)
        dU = (model.strain_energy(model.expand(c + h*v), u0)
              - model.strain_energy(model.expand(c - h*v), u0))/(2*h)
        fv = imp.calc_fint(c) @ v
        assert abs(dU - fv) < 1e-6*abs(fv)
        # with the resultants of the linear strains of u0 only
        U = model.expand(c)
        fv_L = (model.calc_fint(U + u0) - imp.fint0 - imp.KG0 @ U)[model.bu] @ v
        assert abs(dU - fv_L) > 1e-5*abs(fv)
    # without imperfection, the perfect shell
    imp = simo.ImperfectCylinder(model)
    c = u[model.bu]
    f = model.calc_fint(model.expand(c))[model.bu]
    assert np.allclose(imp.calc_fint(c), f, rtol=0, atol=1e-12*np.abs(f).max())


def test_riks_limit_point_imperfection_10():
    r = simo.run_riks(0.1, 6, 6, umax=0.022)
    an = r['an']
    lbds, umax = r['lbds'], r['umax']
    fext = an.calc_fext()
    for lbd, c in zip(an.increments, an.cs):
        assert (np.linalg.norm(an.calc_fint(c) - lbd*fext)
                < 1e-8*np.linalg.norm(fext))
    lim = simo.limit_point(lbds, umax)
    assert lim is not None
    lbd_lim, u_lim = lim
    u_fig, lbd_fig = simo.FIG_A_LIMIT[0.1]
    assert abs(lbd_lim - lbd_fig) < 0.025
    assert abs(u_lim - u_fig) < 0.001
    # deep into the post-buckling regime: the load drops to less than half
    # of the limit load, Fig. 6.3.4(a), and the shortening still grows
    assert lbds[-1] < 0.5*lbd_lim
    assert umax[-1] >= 0.022
