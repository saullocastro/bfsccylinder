r"""Axially compressed cylinder of Simo, Fox and Rifai (1990), with imperfections

Model of the tutorial ``simo1990_cylinder_postbuckling.ipynb`` and of
``tests/test_simo1990_cylinder.py``: Section 6.3.4 of Simo, Fox and Rifai
(1990), "Post-buckling analysis of a simply-supported cylinder; effect of
initial imperfections", pp. 54-57, Fig. 6.3.4 on p. 56. A cylinder of radius
``R = 36``, length ``L = 100`` and thickness ``t = 0.125``, of an isotropic
material with ``E = 1e7`` and ``nu = 0.3``, under axial compression. As in
the paper, after Stanley (1985), an ``L/10 x 15 deg`` section is modelled,
with symmetry conditions on its interior edges and a simple support on the
loaded edge, such that the diamond mode of 5 axial half-waves and 12
circumferential waves is the lowest buckling mode.

The section is meshed with the Donnell-type element :class:`.BFSCCylinder`,
with ``x`` from ``0``, the loaded edge, to ``a = L/10``, and ``y`` from ``0``
to ``b = 15 deg x R``:

- ``x = 0``, simply supported and loaded: ``v = v_y = w = w_y = 0``, ``u``
  free, under the compressive force per unit length ``lbd*N_ref``;
- ``x = a``, symmetry plane of the axial half-wave: ``u = u_y = v_x = w_x =
  w_xy = 0``;
- ``y = 0`` and ``y = b``, symmetry planes of the circumferential waves:
  ``u_y = v = v_x = w_y = w_xy = 0``.

The elements take no initial geometric imperfection. For the Donnell-type
kinematics it is added exactly by :class:`ImperfectCylinder`, from the
vector ``u0`` of the degrees of freedom of the imperfection ``w0``. The
membrane strains of the imperfect shell are ``eps(u + u0) - eps(u0)`` and
its curvatures ``kappa(u)``, see the tutorial, which gives

``fint_imp(u) = fint(u + u0) - fint(u0) - KG(N0) u``

``KT_imp(u) = KT(u + u0) - KG(N0)``

with ``KG(N0)`` the geometric stiffness matrix of the membrane resultants
``N0`` of the whole non-linear strains of ``u0``. :func:`.update_KG` builds
its matrix from the resultants of the linear strains only, ``KG(u0)``, so
the non-linear part ``KGNL0`` is added by :meth:`SectionModel.calc_kG_N`,
with the rows of ``w_x`` and ``w_y`` of the Donnell-type kinematics. The model
is therefore restricted to the Donnell-type element: with the Sanders-type
element of :mod:`bfsccylinder.bfsccylinder_sanders` the rotation is ``-w_y +
v/R``, ``KG(N0)`` has rows in ``v``, and the construction, not verified here,
would need that matrix.

"""
import time

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix, csc_matrix

from composites import isotropic_plate
from structsolve import Analysis, lb, solve
from structsolve.stopping import value_reached

from bfsccylinder import (BFSCCylinder, update_KC0, update_KCNL, update_KG,
        update_fint, DOF, DOUBLE, INT, KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE,
        KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD

R = 36.
L = 100.
t = 0.125
E = 1.e7
nu = 0.3
#: axial length of the section
A_SEC = L/10.
#: arc length of the 15 deg section
B_SEC = R*np.pi/12.
#: classical buckling load per unit length of the circumference
N_CL = E*t**2/(R*np.sqrt(3*(1 - nu**2)))

# Values read from Fig. 6.3.4 of Simo, Fox and Rifai (1990), p. 56, on the
# page rendered at 300 dpi with a grid of 0.001 x 0.05 drawn over the axes
# (readings of the benchmark simo1990_cylinder.py of structsolve, checked
# against the figure). Reading precision about +-0.0003 in the axial
# deflection and +-0.01 in the load normalized by the linear buckling load.
# Pairs (maximum axial deflection, load).
#: Fig. (a), solid line, Simo et al., 10 % imperfection
FIG_A_SIMO_10 = [
    (0.0125, 0.575), (0.0150, 0.675), (0.0168, 0.715), (0.0158, 0.642),
    (0.0151, 0.580), (0.0146, 0.512), (0.0142, 0.445), (0.0143, 0.399),
    (0.0148, 0.366), (0.0158, 0.343), (0.0184, 0.326), (0.0200, 0.330),
    (0.0250, 0.344), (0.0300, 0.372), (0.0350, 0.406), (0.0400, 0.459)]
#: Fig. (a), solid line, Simo et al., 50 % imperfection
FIG_A_SIMO_50 = [
    (0.0108, 0.411), (0.0116, 0.366), (0.0126, 0.335), (0.0141, 0.319),
    (0.0150, 0.315), (0.0200, 0.315), (0.0250, 0.344), (0.0300, 0.370),
    (0.0350, 0.401), (0.0400, 0.449)]
#: Fig. (a), triangles, 9-ANS element of Stanley (1985), 10 % imperfection
FIG_A_ANS_10 = [
    (0.0071, 0.331), (0.0095, 0.445), (0.0125, 0.570), (0.0148, 0.666),
    (0.0159, 0.717), (0.0166, 0.713), (0.0163, 0.693), (0.0157, 0.653),
    (0.0152, 0.619), (0.0148, 0.580), (0.0144, 0.532), (0.0141, 0.487),
    (0.0140, 0.430), (0.0143, 0.382), (0.0157, 0.346), (0.0184, 0.319),
    (0.0218, 0.324), (0.0296, 0.349), (0.0353, 0.395)]
#: Fig. (a), diamonds, 9-ANS element of Stanley (1985), 50 % imperfection
FIG_A_ANS_50 = [
    (0.0055, 0.229), (0.0066, 0.273), (0.0077, 0.310), (0.0084, 0.342),
    (0.0089, 0.366), (0.0094, 0.383), (0.0098, 0.397), (0.0106, 0.402),
    (0.0109, 0.385), (0.0112, 0.367), (0.0118, 0.354), (0.0122, 0.332),
    (0.0132, 0.312), (0.0151, 0.300), (0.0175, 0.297), (0.0204, 0.300),
    (0.0237, 0.318), (0.0265, 0.349), (0.0363, 0.420)]
#: Fig. (d), solid line, Simo et al., perfect cylinder, branch switching
FIG_D_SIMO = [
    (0.0183, 0.833), (0.0182, 0.742), (0.0171, 0.695), (0.0160, 0.599),
    (0.0151, 0.530), (0.0149, 0.457), (0.0160, 0.360), (0.0190, 0.335),
    (0.0200, 0.334), (0.0250, 0.355), (0.0300, 0.397), (0.0350, 0.436),
    (0.0400, 0.480)]
#: limit loads read from Fig. 6.3.4(a), (maximum axial deflection, load)
FIG_A_LIMIT = {0.1: (0.0168, 0.715), 0.5: (0.0108, 0.411)}
#: slope of the straight line of the figures: axial deflection at load 1
FIG_LINEAR_U1 = 0.0205


class SectionModel(object):
    r"""Mesh, boundary conditions and reference load of the section

    Parameters
    ----------
    nx, ny : int
        Number of elements along ``x`` (axial) and ``y`` (circumferential).
    nint : int
        Number of Gauss-Legendre points in each direction of an element.
    load : float
        Compressive force per unit length applied at ``x = 0``.

    Attributes
    ----------
    elements : list of :class:`.BFSCCylinder`
    N : int
        Number of degrees of freedom of the model.
    bu : array of bool
        The unknown (free) degrees of freedom.
    KC0 : csc_matrix
        Linear constitutive stiffness matrix, all degrees of freedom.
    fext : array
        Reference external force vector, all degrees of freedom.
    edofs : array of int, shape ``(num_elements, 40)``
        Global degrees of freedom of each element.
    loaded : array of int
        Node indices of the loaded edge ``x = 0``, in increasing ``y``.

    """
    def __init__(self, nx=8, ny=8, nint=4, load=N_CL):
        self.nx, self.ny, self.nint = nx, ny, nint
        self.points, self.weights = get_points_weights(nint=nint)
        self.prop = isotropic_plate(thickness=t, E=E, nu=nu)
        xlin = np.linspace(0, A_SEC, nx + 1)
        ylin = np.linspace(0, B_SEC, ny + 1)
        xmesh, ymesh = np.meshgrid(xlin, ylin, indexing='ij')
        self.x = x = xmesh.ravel()
        self.y = y = ymesh.ravel()
        nids = np.arange(x.size).reshape(nx + 1, ny + 1)
        n1s = nids[:-1, :-1].ravel()
        n2s = nids[1:, :-1].ravel()
        n3s = nids[1:, 1:].ravel()
        n4s = nids[:-1, 1:].ravel()
        self.N = N = DOF*x.size
        self.lex = A_SEC/nx
        self.ley = B_SEC/ny
        self.elements = []
        for i, (n1, n2, n3, n4) in enumerate(zip(n1s, n2s, n3s, n4s)):
            shell = BFSCCylinder(nint)
            shell.n1, shell.n2, shell.n3, shell.n4 = n1, n2, n3, n4
            shell.c1, shell.c2 = DOF*n1, DOF*n2
            shell.c3, shell.c4 = DOF*n3, DOF*n4
            shell.R = R
            shell.lex = self.lex
            shell.ley = self.ley
            assign_constant_ABD(shell, self.prop)
            shell.init_k_KC0 = i*KC0_SPARSE_SIZE
            shell.init_k_KCNL = i*KCNL_SPARSE_SIZE
            shell.init_k_KG = i*KG_SPARSE_SIZE
            self.elements.append(shell)
        ne = len(self.elements)
        self.edofs = np.array([np.concatenate([DOF*n + np.arange(DOF)
                               for n in nodes])
                               for nodes in zip(n1s, n2s, n3s, n4s)])
        KC0r = np.zeros(KC0_SPARSE_SIZE*ne, dtype=INT)
        KC0c = np.zeros(KC0_SPARSE_SIZE*ne, dtype=INT)
        KC0v = np.zeros(KC0_SPARSE_SIZE*ne, dtype=DOUBLE)
        for shell in self.elements:
            update_KC0(shell, self.points, self.weights, KC0r, KC0c, KC0v)
        self.KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
        self.KCNLr = np.zeros(KCNL_SPARSE_SIZE*ne, dtype=INT)
        self.KCNLc = np.zeros(KCNL_SPARSE_SIZE*ne, dtype=INT)
        self.KCNLv = np.zeros(KCNL_SPARSE_SIZE*ne, dtype=DOUBLE)
        self.KGr = np.zeros(KG_SPARSE_SIZE*ne, dtype=INT)
        self.KGc = np.zeros(KG_SPARSE_SIZE*ne, dtype=INT)
        self.KGv = np.zeros(KG_SPARSE_SIZE*ne, dtype=DOUBLE)

        # boundary conditions, dofs u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xy
        bk = np.zeros(N, dtype=bool)
        x0 = isclose(x, 0)
        xa = isclose(x, A_SEC)
        ye = isclose(y, 0) | isclose(y, B_SEC)
        for dof in (3, 5, 6, 8):
            bk[dof::DOF] |= x0
        for dof in (0, 2, 4, 7, 9):
            bk[dof::DOF] |= xa
        for dof in (2, 3, 4, 8, 9):
            bk[dof::DOF] |= ye
        self.bk = bk
        self.bu = ~bk
        self.loaded = np.flatnonzero(x0)
        self.loaded = self.loaded[np.argsort(y[self.loaded])]

        # consistent nodal forces of the uniform force per unit length
        # ``load`` along the edge x = 0, compressive, pointing to +x: the
        # integrals of the Hermite functions along an element side of length
        # ley are ley/2 for the values and +-ley**2/12 for the derivatives
        # u_y, which cancel between neighbours and fall on the constrained
        # u_y of the corners
        self.fext = np.zeros(N, dtype=DOUBLE)
        for na, nb in zip(self.loaded[:-1], self.loaded[1:]):
            self.fext[DOF*na + 0] += load*self.ley/2
            self.fext[DOF*nb + 0] += load*self.ley/2
            self.fext[DOF*na + 2] += load*self.ley**2/12
            self.fext[DOF*nb + 2] -= load*self.ley**2/12
        self.fext[self.bk] = 0
        self.load = load

        # rows of the interpolation at the integration points, the same for
        # every element of this regular mesh: Bm, Bb, Sw_x and Sw_y of shape
        # (num_points, ..., 40), and the weights times the Jacobian
        shell = self.elements[0]
        Bm, Bb, Gx, Gy, wp = [], [], [], [], []
        for i, xi in enumerate(self.points):
            for j, eta in enumerate(self.points):
                shell.update_Bm(xi, eta)
                shell.update_Bb(xi, eta)
                shell.update_Sw_x(xi, eta)
                shell.update_Sw_y(xi, eta)
                Bm.append(np.array(shell.Bm))
                Bb.append(np.array(shell.Bb))
                Gx.append(np.array(shell.Sw_x))
                Gy.append(np.array(shell.Sw_y))
                wp.append(self.weights[i]*self.weights[j]
                          *self.lex*self.ley/4)
        self.Bm, self.Bb = np.array(Bm), np.array(Bb)
        self.Gx, self.Gy = np.array(Gx), np.array(Gy)
        self.wp = np.array(wp)
        p = self.prop
        self.A = np.array([[p.A11, p.A12, p.A16], [p.A12, p.A22, p.A26],
                           [p.A16, p.A26, p.A66]])
        self.B = np.array([[p.B11, p.B12, p.B16], [p.B12, p.B22, p.B26],
                           [p.B16, p.B26, p.B66]])
        self.D = np.array([[p.D11, p.D12, p.D16], [p.D12, p.D22, p.D26],
                           [p.D16, p.D26, p.D66]])

    def expand(self, c):
        """The displacements of all dofs from those of the unknown dofs"""
        u = np.zeros(self.N, dtype=DOUBLE)
        u[self.bu] = c
        return u

    def restrict(self, K):
        """``K`` on the unknown dofs"""
        return csc_matrix(K)[self.bu, :][:, self.bu]

    def calc_fint(self, u):
        """Internal forces of the displacements ``u``, all dofs"""
        fint = np.zeros(self.N, dtype=DOUBLE)
        for shell in self.elements:
            update_fint(u, shell, self.points, self.weights, fint)
        return fint

    def calc_KCNL(self, u):
        """``KCNL`` of :func:`.update_KCNL` at ``u``, all dofs"""
        self.KCNLv *= 0
        for shell in self.elements:
            update_KCNL(u, shell, self.points, self.weights, self.KCNLr,
                        self.KCNLc, self.KCNLv)
        return coo_matrix((self.KCNLv, (self.KCNLr, self.KCNLc)),
                          shape=(self.N, self.N)).tocsc()

    def calc_KG(self, u):
        """``KG`` of :func:`.update_KG` at ``u``, all dofs: the membrane
        resultants of the linear strains only"""
        self.KGv *= 0
        for shell in self.elements:
            update_KG(u, shell, self.points, self.weights, self.KGr,
                      self.KGc, self.KGv)
        return coo_matrix((self.KGv, (self.KGr, self.KGc)),
                          shape=(self.N, self.N)).tocsc()

    def generalized_strains(self, u):
        """Linear membrane strains, non-linear membrane strains and
        curvatures of ``u`` at the integration points, each of shape
        ``(num_elements, num_points, 3)``"""
        ue = u[self.edofs]
        epsL = np.einsum('pkj,ej->epk', self.Bm, ue)
        kappa = np.einsum('pkj,ej->epk', self.Bb, ue)
        wx = np.einsum('pj,ej->ep', self.Gx, ue)
        wy = np.einsum('pj,ej->ep', self.Gy, ue)
        epsNL = np.stack((wx**2/2, wy**2/2, wx*wy), axis=-1)
        return epsL, epsNL, kappa

    def membrane_resultants(self, u, nonlinear=True):
        """``(Nxx, Nyy, Nxy)`` of ``u`` at the integration points, of shape
        ``(num_elements, num_points, 3)``, with or without the non-linear
        strains"""
        epsL, epsNL, kappa = self.generalized_strains(u)
        eps = epsL + epsNL if nonlinear else epsL
        return eps @ self.A.T + kappa @ self.B.T

    def calc_kG_N(self, N):
        r"""Geometric stiffness matrix of the membrane resultants ``N`` given
        at the integration points, shape ``(num_elements, num_points, 3)``:
        `\int_A \boldsymbol{G}^\top [N_{xx}\ N_{xy};\ N_{xy}\ N_{yy}]
        \boldsymbol{G} dA`, all dofs, with the quadrature of the elements"""
        Nw = N*self.wp[None, :, None]
        Gx, Gy = self.Gx, self.Gy
        ke = (np.einsum('ep,pi,pj->eij', Nw[..., 0], Gx, Gx)
              + np.einsum('ep,pi,pj->eij', Nw[..., 1], Gy, Gy)
              + np.einsum('ep,pi,pj->eij', Nw[..., 2], Gx, Gy)
              + np.einsum('ep,pi,pj->eij', Nw[..., 2], Gy, Gx))
        rows = np.repeat(self.edofs[:, :, None], 40, axis=2)
        cols = np.repeat(self.edofs[:, None, :], 40, axis=1)
        return coo_matrix((ke.ravel(), (rows.ravel(), cols.ravel())),
                          shape=(self.N, self.N)).tocsc()

    def strain_energy(self, u, u0=None):
        """Strain energy of the strains ``eps(u + u0) - eps(u0)`` and the
        curvatures ``kappa(u)``, by the quadrature of the elements,
        independently of the kernels of the elements"""
        if u0 is None:
            u0 = np.zeros_like(u)
        eL1, eN1, k1 = self.generalized_strains(u + u0)
        eL0, eN0, k0 = self.generalized_strains(u0)
        e = (eL1 + eN1) - (eL0 + eN0)
        k = k1 - k0
        dens = (np.einsum('epi,ij,epj->ep', e, self.A, e)
                + 2*np.einsum('epi,ij,epj->ep', e, self.B, k)
                + np.einsum('epi,ij,epj->ep', k, self.D, k))
        return 0.5*np.sum(dens*self.wp[None, :])

    def shortening(self, u):
        """Maximum and average axial displacement of the loaded edge ``x =
        0``, the shortening of the section: the nodal maximum and the
        integral of the Hermite interpolation of ``u`` over the edge"""
        n = self.loaded
        ui = u[DOF*n + 0]
        uyi = u[DOF*n + 2]
        l = self.ley
        integral = np.sum(l/2*(ui[:-1] + ui[1:])
                          + l**2/12*(uyi[:-1] - uyi[1:]))
        return ui.max(), integral/B_SEC

    def w_max(self, u):
        """Largest nodal ``|w|``"""
        return np.abs(u[6::DOF]).max()


class ImperfectCylinder(object):
    r"""Callables of structsolve.Analysis for the section with the initial
    imperfection ``u0``

    ``u0`` has all dofs of the model, with only the dofs of `w` non-zero,
    ``u0 = 0`` gives the perfect shell. The callables work on the unknown
    dofs ``c = u[bu]``:

    - ``calc_fint(c) = fint(u + u0) - fint(u0) - KG(u0) u - KGNL0 u``
    - ``calc_kC(c, NLgeom=True) = KC0 + KCNL(u + u0) - KGNL0``
    - ``calc_kG(c, NLgeom=True) = KG(u + u0) - KG(u0) = KG(u)``

    with ``KG`` of :func:`.update_KG`, which is linear in ``u``, and
    ``KGNL0`` the geometric stiffness matrix of the membrane resultants of
    the non-linear strains of ``u0``, such that ``KG(u0) + KGNL0 = KG(N0)``.
    ``calc_kC + calc_kG`` is the Jacobian of ``calc_fint``, and ``calc_kG``
    keeps the meaning of the geometric stiffness matrix of the perfect
    shell. ``calc_kC()``, without ``NLgeom``, gives the linear stiffness
    matrix of the imperfect shell, ``KC0 + KCNL(u0) - KGNL0 + KG(u0)``.

    """
    def __init__(self, model, u0=None):
        self.model = m = model
        self.u0 = np.zeros(m.N) if u0 is None else np.asarray(u0, float)
        self.fint0 = m.calc_fint(self.u0)
        N0NL = (m.membrane_resultants(self.u0, nonlinear=True)
                - m.membrane_resultants(self.u0, nonlinear=False))
        self.KGNL0 = m.calc_kG_N(N0NL)
        self.KG0 = m.calc_KG(self.u0)
        self.KC0uu = m.restrict(m.KC0)
        self.K0uu = m.restrict(m.KC0 + m.calc_KCNL(self.u0) - self.KGNL0
                               + self.KG0)
        self.fextuu = m.fext[m.bu]

    def calc_fext(self, inc=1., silent=True):
        return inc*self.fextuu

    def calc_fint(self, c, silent=True):
        m = self.model
        u = m.expand(c)
        f = (m.calc_fint(u + self.u0) - self.fint0 - self.KG0 @ u
             - self.KGNL0 @ u)
        return f[m.bu]

    def calc_kC(self, c=None, NLgeom=False, silent=True):
        if c is None or not NLgeom:
            return self.K0uu
        m = self.model
        u = m.expand(c)
        return m.restrict(m.KC0 + m.calc_KCNL(u + self.u0) - self.KGNL0)

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        m = self.model
        u = m.expand(c)
        return m.restrict(m.calc_KG(u))


def linear_buckling(model, num_eigvalues=4):
    """Linear buckling load factors of the reference load of ``model`` and
    the modes, all dofs, from the linear static solution"""
    KC0uu = model.restrict(model.KC0)
    clin = solve(KC0uu, model.fext[model.bu], silent=True)
    KGuu = model.restrict(model.calc_KG(model.expand(clin)))
    eigvals, eigvecs = lb(KC0uu, KGuu, num_eigvalues=num_eigvalues,
                          silent=True)
    modes = np.zeros((model.N, eigvecs.shape[1]))
    modes[model.bu] = np.real(eigvecs)
    return np.real(eigvals), modes, model.expand(clin)


def imperfection(model, mode, xi):
    """The dofs of `w` of ``mode``, scaled to a largest nodal ``|w|`` of
    ``xi*t``"""
    u0 = np.zeros_like(mode)
    for dof in (6, 7, 8, 9):
        u0[dof::DOF] = mode[dof::DOF]
    w = u0[6::DOF]
    return u0*xi*t/w[np.argmax(np.abs(w))]


def normalized_model(nx, ny, nint=4):
    """Model whose reference load is its linear buckling load, the lowest
    buckling mode and the linear buckling load in units of ``N_CL``"""
    model = SectionModel(nx, ny, nint)
    eigvals, modes, _ = linear_buckling(model)
    lbd_cr = eigvals[0]
    model = SectionModel(nx, ny, nint, load=lbd_cr*N_CL)
    return model, modes[:, 0], lbd_cr


def run_riks(xi, nx=8, ny=8, nint=4, umax=0.04, branch_switching=0,
             silent=True, **kwargs):
    r"""Riks analysis of the section until the maximum shortening ``umax``

    ``xi > 0``: imperfection of amplitude ``xi*t`` in the shape of the lowest
    linear buckling mode; ``xi = 0``: the perfect shell, with ``inertia``,
    SuperLU, and ``branch_switching``. ``kwargs`` are attributes of the
    :class:`structsolve.Analysis`. Returns a dict with the load factors
    ``lbds`` (in units of the linear buckling load), the maximum and
    average shortenings ``umax`` and ``uavg``, the largest ``|w|``
    ``wmax``, starting from the unloaded state, the linear buckling load
    ``lbd_cr`` in units of ``N_CL``, the run time ``elapsed`` and the
    analysis ``an``.

    """
    model, mode, lbd_cr = normalized_model(nx, ny, nint)
    u0 = imperfection(model, mode, xi) if xi > 0 else np.zeros(model.N)
    imp = ImperfectCylinder(model, u0)
    an = Analysis(imp.calc_fext, imp.calc_fint, imp.calc_kC, imp.calc_kG)
    an.NL_method = 'arc_length_riks'
    an.final_load_factor = None
    an.initialInc = 0.1
    an.maxInc = 0.1
    an.minInc = 1e-5
    an.maxArcLength = 1e4
    if xi == 0:
        an.inertia = True
        an.branch_switching = branch_switching
        an.linear_solver = 'splu'
        an.initialInc = an.maxInc = 0.03
    an.stop_criterion = value_reached(
        lambda c: model.shortening(model.expand(c))[0], umax)
    for k, v in kwargs.items():
        setattr(an, k, v)
    t0 = time.perf_counter()
    an.static(NLgeom=True, silent=silent)
    elapsed = time.perf_counter() - t0
    us = [model.expand(c) for c in an.cs]
    sh = np.array([[0., 0.]] + [model.shortening(u) for u in us])
    return dict(lbds=np.concatenate(([0.], an.increments)),
                umax=sh[:, 0], uavg=sh[:, 1],
                wmax=np.array([0.] + [model.w_max(u) for u in us]),
                lbd_cr=lbd_cr, elapsed=elapsed, an=an, model=model, u0=u0)


def limit_point(lbds, u):
    """Load and shortening at the first limit point, the first decrease of
    the load factor, ``None`` if there is none"""
    drops = np.flatnonzero(np.diff(lbds) < 0)
    if drops.size == 0:
        return None
    i = drops[0]
    return lbds[i], u[i]
