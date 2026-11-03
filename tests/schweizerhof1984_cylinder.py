r"""Cantilever cylinder under a follower pressure, Schweizerhof and Ramm (1984)

Model of ``tests/test_schweizerhof1984_cylinder.py``: Section 6.4 of
Schweizerhof and Ramm (1984), "Cylindrical cantilever shell under uniform
pressure", Fig. 12 on p. 1110, Table 4 on p. 1111 and Fig. 13 on p. 1112. A
cylinder of radius
``R``, length ``L`` and thickness ``t``, of an isotropic material with ``E =
2.1e6`` and ``mu = 0.3``, upper edge free, lower edge hinged, under a
uniform external pressure, either of constant direction (dead, index ``c``
of the paper) or normal to the deformed surface (follower, index ``f``).
The three shells of the paper are given by :data:`SHELLS`, with the number
``n`` of circumferential waves of their buckling mode. The reference
pressure of the models is that of ``k = 1`` in the normalization of eq.
(60) of the paper, :func:`p_ref`, such that the load multipliers are the
factors ``k`` of Table 4 and the load ``K`` of Fig. 13. Here ``R = 100``,
the factors do not depend on it.

As in the paper, a sector of the angle ``180 deg/n`` (60, 45 and 30 deg for
the shells 1, 2 and 3) is modelled, with ``x`` from ``0``, the lower edge, to
``L``, the free upper edge, and ``y`` from ``0`` to ``b = pi R/n``, with
symmetry planes at ``y = 0`` and ``y = b``, which keep the modes with
``n``, ``2n``, ... circumferential waves; ``full=True`` gives the closed
cylinder instead, with every ``n``. The degrees of freedom of a node are
``u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xy``:

- ``x = 0``, B1, "warping completely restrained", ``v = w = u = m_x = 0``:
  ``u = u_y = v = v_y = w = w_y = 0``;
- ``x = 0``, B2, "warping partially restrained", ``v = w = u0 = m_x = 0``:
  ``v = v_y = w = w_y = 0`` and ``u = 0`` at the node ``x = y = 0`` only,
  which removes the axial translation of the sector (sector only);
- ``y = 0`` and ``y = b``, symmetry planes: ``u_y = v = v_x = w_y = w_xy =
  0``;
- ``x = L``: free.

The element is the Sanders-type :class:`.BFSCCylinderSanders` by default,
``kinematics='donnell'`` gives the Donnell-type :class:`.BFSCCylinder`. The
follower pressure is that of :func:`.update_fext_follower` with the
first-order area vector (``quadratic=0``), with its full, unsymmetric, load
stiffness matrix of :func:`.update_KCfollower`.

The initial geometric imperfection ``u0``, of the dofs of `w` only, is added
by :class:`FollowerCylinder` as in ``simo1990_cylinder.py`` of this folder,
for both kinematics; the pressure acts on the imperfect surface.
:func:`run_riks` traces the path with the Riks method of structsolve,
recording the count of negative eigenvalues of the unsymmetric tangent
stiffness matrix. The constants ``TABLE4``, ``TABLE4_CLAMPED`` and
``FIG13_*`` are values of Table 4 and Fig. 13 of the paper.

Notes
-----
The pressure is normalized by eq. (60) of [1]_,

.. math::
    p = k \frac{E}{12 (1 - \mu^2)} \left(\frac{t}{R}\right)^3

On the free edge the axial displacement is not restrained, which makes the
pressure nonconservative and its load stiffness matrix unsymmetric. With
``n = 3`` circumferential waves over a length of six radii the Donnell-type
kinematics are not accurate: they underestimate the B1 factors of shell 1
by 3.4 % (``k_c``) and 2.2 % (``k_f``), the terms in `1/R` of the
Sanders-type curvatures and rotations being significant.

The imperfect shell has the membrane strains `\boldsymbol{\varepsilon}
(\boldsymbol{u} + \boldsymbol{u}_0) - \boldsymbol{\varepsilon}
(\boldsymbol{u}_0)` and the curvatures `\boldsymbol{\kappa}
(\boldsymbol{u})`, which gives

.. math::
    \boldsymbol{F}_{int,imp}(\boldsymbol{u}) =
    \boldsymbol{F}_{int}(\boldsymbol{u} + \boldsymbol{u}_0) -
    \boldsymbol{F}_{int}(\boldsymbol{u}_0) -
    \boldsymbol{K}_G(\boldsymbol{N}_0) \boldsymbol{u}

with `\boldsymbol{K}_G(\boldsymbol{N}_0)` the geometric stiffness matrix of
the membrane resultants `\boldsymbol{N}_0` of the strains of
`\boldsymbol{u}_0`. The derivation holds for both kinematics, whose
non-linear membrane strains are quadratic forms of the rotations
`\boldsymbol{G} \boldsymbol{u}`, with the rows `\boldsymbol{S}^w_{,x}` and
`\boldsymbol{S}^w_{,y}` (Donnell) or `\boldsymbol{S}^w_{,x}` and
`\boldsymbol{S}^w_{,y} - \boldsymbol{S}^v/R` (Sanders) of `\boldsymbol{G}`,
see :meth:`CantileverModel.calc_kG_N`. The follower force vector is that of
the displacements `\boldsymbol{u} + \boldsymbol{u}_0`, and the residual is

.. math::
    \boldsymbol{R} = \boldsymbol{F}_{int,imp}(\boldsymbol{u}) - k
    \boldsymbol{F}_p(\boldsymbol{u} + \boldsymbol{u}_0)

Table 4 was read on the page image of [1]_, p. 1111; the `k_f` of B2 are
those of the load stiffness matrix symmetrized by eqs. (63), (64) and (65),
and the paper gives a single `k_f` for B1 (none for shell 3). Fig. 13, p.
1112, imperfect shell 1 under the follower pressure, imperfection of 1 % of
`t` in the shape of the linear buckling mode, was read on the page rendered
at 300 dpi, from the centre line of the thick curve, with the axes
calibrated on their ticks: 87.0 px per unit of `K` and 268 px per unit of
`w/t`. The curve is about 6 px thick, the reading precision is about `\pm
0.05` in `K` and `\pm 0.01` in `w/t` (`\pm 0.003` below `w/t = 0.2`, where
the curve is steep). The dashed curve of the current stiffness parameter of
Bergan [2]_ is read on the dashes to about `\pm 0.02` in the parameter
and `\pm 0.05` in `K`.

References
----------
.. [1] Schweizerhof, K., and Ramm, E., Displacement dependent pressure
   loads in nonlinear finite element analyses, Computers & Structures,
   18(6), 1099-1114, 1984, doi:10.1016/0045-7949(84)90154-8.
.. [2] Bergan, P. G., Automated incremental-iterative solution schemes, in
   Numerical Methods for Nonlinear Problems, edited by C. Taylor, E. Hinton
   and D. R. J. Owen, Pineridge Press, Swansea, 1980 (ref. [7] of [1]_).

"""
import time

import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix, csc_matrix

from composites import isotropic_plate
from structsolve import Analysis, lb, solve

import bfsccylinder
import bfsccylinder.sanders
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD

DOF = 10
E = 2.1e6
MU = 0.3
R = 100.
#: shells 1, 2 and 3 of Section 6.4: ``(L/R, R/t, n)``
SHELLS = {1: (6., 100., 3), 2: (3., 100., 4), 3: (3., 500., 6)}
KINEMATICS = {
    'donnell': (bfsccylinder, 'BFSCCylinder'),
    'sanders': (bfsccylinder.sanders, 'BFSCCylinderSanders'),
}

# Table 4, k_f of B2 with the symmetrized load stiffness matrices
#: ``TABLE4[(shell, bc)] = (k_c, k_f)``, ``k_f`` a tuple of (63), (64), (65)
TABLE4 = {
    (1, 'B2'): (9.11, (8.10, 8.09, 8.12)),
    (1, 'B1'): (10.88, (9.70,)),
    (2, 'B2'): (16.43, (15.42, 15.38, 15.46)),
    (2, 'B1'): (20.85, (19.58,)),
    (3, 'B2'): (36.48, (35.48, 35.46, 35.50)),
    (3, 'B1'): (45.83, ()),
}
#: Table 4, clamped-free shells, k_f of refs [36] and [15] of the paper
TABLE4_CLAMPED = {1: (9.85, 10.51), 2: (19.97, 21.01), 3: (None, 46.98)}

# Fig. 13, imperfect shell 1, follower pressure, pairs (w/t, K)
#: Fig. 13, imperfect shell, maximum displacement against K
FIG13_IMPERFECT = [
    (0.058, 6.72), (0.095, 7.19), (0.133, 7.41), (0.170, 7.54),
    (0.245, 7.71), (0.319, 7.77), (0.506, 7.85), (0.693, 7.91),
    (1.002, 7.93), (1.439, 7.95), (2.007, 7.99), (2.559, 8.00),
    (2.970, 8.01)]
#: Fig. 13, horizontal line of the perfect shell, K
FIG13_PERFECT = 8.06
#: Fig. 13, dashed curve, current stiffness parameter of Bergan, pairs
#: (CS, K)
FIG13_CS = [
    (0.97, 0.98), (0.948, 1.44), (0.93, 1.90), (0.88, 2.36), (0.81, 2.82),
    (0.747, 3.28), (0.667, 3.74), (0.57, 4.20), (0.475, 4.66),
    (0.37, 5.11), (0.26, 5.57), (0.18, 6.03), (0.10, 6.49), (0.047, 6.95),
    (0.02, 7.41)]


def shell_geometry(shell):
    """``(L, t, n)`` of the shell ``1``, ``2`` or ``3``, with ``R = 100``"""
    LR, Rt, n = SHELLS[shell]
    return LR*R, R/Rt, n


def p_ref(t):
    """Pressure of ``k = 1`` in eq. (60) of the paper, positive, for the
    thickness ``t``"""
    return E/(12*(1 - MU**2))*(t/R)**3


class CantileverModel(object):
    r"""Mesh, boundary conditions and matrices of the cantilever cylinder

    Parameters
    ----------
    shell : int
        ``1``, ``2`` or ``3``, see :data:`SHELLS`.
    bc : str
        ``'B1'`` or ``'B2'``, the boundary condition of the lower edge.
    nx, ny : int
        Number of elements along ``x`` (axial) and ``y`` (circumferential).
    kinematics : str
        ``'sanders'`` or ``'donnell'``.
    n : int or None
        Number of circumferential waves of the sector of ``180 deg/n``,
        ``None`` for that of :data:`SHELLS`.
    full : bool
        The closed cylinder instead of the sector, only with ``'B1'``.
    nint : int
        Number of Gauss-Legendre points in each direction of an element.

    Attributes
    ----------
    L, t, b : float
        Length, thickness and arc length of the model.
    N : int
        Number of degrees of freedom.
    bu : array of bool
        The unknown (free) degrees of freedom.
    KC0, KF : csc_matrix
        Linear constitutive stiffness matrix and load stiffness matrix of
        the reference pressure (external, ``k = 1``), all dofs; ``KF`` is
        constant with the first-order area vector.
    fp0 : array
        Force vector of the reference pressure on the undeformed shell, the
        dead load, all dofs.

    """
    def __init__(self, shell=1, bc='B1', nx=16, ny=4, kinematics='sanders',
                 n=None, full=False, nint=4):
        self.shell, self.bc, self.kinematics = shell, bc, kinematics
        self.nx, self.ny, self.nint, self.full = nx, ny, nint, full
        self.L, self.t, n0 = shell_geometry(shell)
        self.n = n = n0 if n is None else n
        self.mod = mod = KINEMATICS[kinematics][0]
        cls = getattr(mod, KINEMATICS[kinematics][1])
        self.points, self.weights = get_points_weights(nint=nint)
        self.prop = isotropic_plate(thickness=self.t, E=E, nu=MU)
        self.b = b = 2*pi*R if full else pi*R/n
        nyn = ny if full else ny + 1
        xlin = np.linspace(0, self.L, nx + 1)
        ylin = np.arange(nyn)*b/ny
        xmesh, ymesh = np.meshgrid(xlin, ylin, indexing='ij')
        self.x = x = xmesh.ravel()
        self.y = y = ymesh.ravel()
        nids = np.arange(x.size).reshape(nx + 1, nyn)
        if full:
            nids = np.hstack((nids, nids[:, :1]))
        n1s = nids[:-1, :-1].ravel()
        n2s = nids[1:, :-1].ravel()
        n3s = nids[1:, 1:].ravel()
        n4s = nids[:-1, 1:].ravel()
        self.N = N = DOF*x.size
        self.lex, self.ley = self.L/nx, b/ny
        self.elements = []
        for i, (n1, n2, n3, n4) in enumerate(zip(n1s, n2s, n3s, n4s)):
            elem = cls(nint)
            elem.n1, elem.n2, elem.n3, elem.n4 = n1, n2, n3, n4
            elem.c1, elem.c2 = DOF*n1, DOF*n2
            elem.c3, elem.c4 = DOF*n3, DOF*n4
            elem.R = R
            elem.lex = self.lex
            elem.ley = self.ley
            assign_constant_ABD(elem, self.prop)
            elem.init_k_KC0 = i*mod.KC0_SPARSE_SIZE
            elem.init_k_KCNL = i*mod.KCNL_SPARSE_SIZE
            elem.init_k_KG = i*mod.KG_SPARSE_SIZE
            elem.init_k_KCfollower = i*mod.KCfollower_SPARSE_SIZE
            self.elements.append(elem)
        ne = len(self.elements)
        self.edofs = np.array([np.concatenate([DOF*nd + np.arange(DOF)
                               for nd in nodes])
                               for nodes in zip(n1s, n2s, n3s, n4s)])
        INT, DOUBLE = mod.INT, mod.DOUBLE
        pts, wts = self.points, self.weights

        def arrays(size):
            return (np.zeros(size*ne, dtype=INT), np.zeros(size*ne, dtype=INT),
                    np.zeros(size*ne, dtype=DOUBLE))

        r, c, v = arrays(mod.KC0_SPARSE_SIZE)
        for elem in self.elements:
            mod.update_KC0(elem, pts, wts, r, c, v)
        self.KC0 = coo_matrix((v, (r, c)), shape=(N, N)).tocsc()
        self.KCNLr, self.KCNLc, self.KCNLv = arrays(mod.KCNL_SPARSE_SIZE)
        self.KGr, self.KGc, self.KGv = arrays(mod.KG_SPARSE_SIZE)
        # reference pressure, external: negative
        self.p = -p_ref(self.t)
        self.pp = np.full((nint, nint), self.p)
        zero = np.zeros(N)
        r, c, v = arrays(mod.KCfollower_SPARSE_SIZE)
        self.fp0 = np.zeros(N)
        for elem in self.elements:
            mod.update_KCfollower(zero, elem, pts, wts, self.pp, r, c, v)
            mod.update_fext_follower(zero, elem, pts, wts, self.pp, self.fp0)
        self.KF = coo_matrix((v, (r, c)), shape=(N, N)).tocsc()

        # boundary conditions
        bk = np.zeros(N, dtype=bool)
        x0 = isclose(x, 0)
        if bc == 'B1':
            dofs = (0, 2, 3, 5, 6, 8)
        elif bc == 'B2':
            if full:
                raise NotImplementedError('B2 only for the sector')
            dofs = (3, 5, 6, 8)
            bk[0] = True # u of the node x = y = 0
        else:
            raise ValueError(bc)
        for dof in dofs:
            bk[dof::DOF] |= x0
        if not full:
            ye = isclose(y, 0) | isclose(y, b)
            for dof in (2, 3, 4, 8, 9):
                bk[dof::DOF] |= ye
        self.bk = bk
        self.bu = ~bk

        # rows of the interpolation at the integration points, the same for
        # every element of this regular mesh, for calc_kG_N and the strains
        elem = self.elements[0]
        Bm, Bb, Gx, Gy, wp = [], [], [], [], []
        for i, xi in enumerate(pts):
            for j, eta in enumerate(pts):
                elem.update_Bm(xi, eta)
                elem.update_Bb(xi, eta)
                elem.update_Sw_x(xi, eta)
                elem.update_Sw_y(xi, eta)
                elem.update_Sv(xi, eta)
                Bm.append(np.array(elem.Bm))
                Bb.append(np.array(elem.Bb))
                Gx.append(np.array(elem.Sw_x))
                gy = np.array(elem.Sw_y)
                if kinematics == 'sanders':
                    gy = gy - np.array(elem.Sv)/R
                Gy.append(gy)
                wp.append(wts[i]*wts[j]*self.lex*self.ley/4)
        self.Bm, self.Bb = np.array(Bm), np.array(Bb)
        self.Gx, self.Gy = np.array(Gx), np.array(Gy)
        self.wp = np.array(wp)
        self._rows = np.repeat(self.edofs[:, :, None], 40, axis=2).ravel()
        self._cols = np.repeat(self.edofs[:, None, :], 40, axis=1).ravel()
        p = self.prop
        self.A = np.array([[p.A11, p.A12, p.A16], [p.A12, p.A22, p.A26],
                           [p.A16, p.A26, p.A66]])
        self.B = np.array([[p.B11, p.B12, p.B16], [p.B12, p.B22, p.B26],
                           [p.B16, p.B26, p.B66]])
        self.D = np.array([[p.D11, p.D12, p.D16], [p.D12, p.D22, p.D26],
                           [p.D16, p.D26, p.D66]])

    def expand(self, c):
        """The displacements of all dofs from those of the unknown dofs"""
        u = np.zeros(self.N)
        u[self.bu] = c
        return u

    def restrict(self, K):
        """``K`` on the unknown dofs"""
        return csc_matrix(K)[self.bu, :][:, self.bu]

    def calc_fint(self, u):
        """Internal forces of the displacements ``u``, all dofs"""
        fint = np.zeros(self.N)
        for elem in self.elements:
            self.mod.update_fint(u, elem, self.points, self.weights, fint)
        return fint

    def calc_fp(self, u, reference=1):
        """Follower force vector of the reference pressure at ``u``, all
        dofs; ``reference=0`` leaves out the dead load ``fp0``"""
        f = np.zeros(self.N)
        for elem in self.elements:
            self.mod.update_fext_follower(u, elem, self.points, self.weights,
                                          self.pp, f, 0, reference)
        return f

    def calc_KCNL(self, u):
        """``KCNL`` of :func:`.update_KCNL` at ``u``, all dofs"""
        self.KCNLv *= 0
        for elem in self.elements:
            self.mod.update_KCNL(u, elem, self.points, self.weights,
                                 self.KCNLr, self.KCNLc, self.KCNLv)
        return coo_matrix((self.KCNLv, (self.KCNLr, self.KCNLc)),
                          shape=(self.N, self.N)).tocsc()

    def calc_KG(self, u):
        """``KG`` of :func:`.update_KG` at ``u``, all dofs: the membrane
        resultants of the linear strains only"""
        self.KGv *= 0
        for elem in self.elements:
            self.mod.update_KG(u, elem, self.points, self.weights, self.KGr,
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
        gx = np.einsum('pj,ej->ep', self.Gx, ue)
        gy = np.einsum('pj,ej->ep', self.Gy, ue)
        epsNL = np.stack((gx**2/2, gy**2/2, gx*gy), axis=-1)
        return epsL, epsNL, kappa

    def membrane_resultants(self, u, nonlinear=True):
        """``(Nxx, Nyy, Nxy)`` of ``u`` at the integration points, with or
        without the non-linear strains"""
        epsL, epsNL, kappa = self.generalized_strains(u)
        eps = epsL + epsNL if nonlinear else epsL
        return eps @ self.A.T + kappa @ self.B.T

    def calc_kG_N(self, N):
        r"""Geometric stiffness matrix of the membrane resultants ``N`` given
        at the integration points, shape ``(num_elements, num_points, 3)``,
        all dofs, with ``G`` of the kinematics

        Notes
        -----
        The matrix is

        .. math::
            \int_A \boldsymbol{G}^\top \begin{bmatrix} N_{xx} & N_{xy} \\
            N_{xy} & N_{yy} \end{bmatrix} \boldsymbol{G} \, dA

        with the rows of `\boldsymbol{G}` of the module Notes, evaluated
        with the weights and points of the elements.

        """
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

    def calc_KT(self, u):
        r"""Tangent stiffness matrix at ``u``, all dofs, ``KC0 + KCNL(u) +
        KG(u)``, vectorized over the elements with NumPy

        It gives the matrix of the kernels :func:`.update_KC0`,
        :func:`.update_KCNL` and :func:`.update_KG` to the round-off, as
        checked by ``tests/test_schweizerhof1984_cylinder.py``, in a fraction
        of the time of ``update_KCNL`` of the Sanders-type element.

        Notes
        -----
        The matrix is the Hessian of :meth:`strain_energy` with
        `\boldsymbol{u}_0 = \boldsymbol{0}`,

        .. math::
            \int_A \left( \boldsymbol{B}^\top \boldsymbol{A} \boldsymbol{B}
            + \boldsymbol{B}^\top \boldsymbol{B}_c \boldsymbol{B}_b
            + \boldsymbol{B}_b^\top \boldsymbol{B}_c \boldsymbol{B}
            + \boldsymbol{B}_b^\top \boldsymbol{D} \boldsymbol{B}_b
            + \boldsymbol{G}^\top \boldsymbol{N} \boldsymbol{G} \right) dA,
            \qquad \boldsymbol{B} = \boldsymbol{B}_m +
            \boldsymbol{B}_{NL}(\boldsymbol{u})

        with `\boldsymbol{B}_c` the coupling matrix ``B`` of the laminate,
        integrated with the quadrature of the elements.

        """
        ue = u[self.edofs]
        gx = np.einsum('pj,ej->ep', self.Gx, ue)
        gy = np.einsum('pj,ej->ep', self.Gy, ue)
        Bm = self.Bm[None] + np.stack((
            gx[..., None]*self.Gx[None],
            gy[..., None]*self.Gy[None],
            gx[..., None]*self.Gy[None] + gy[..., None]*self.Gx[None]),
            axis=2)
        Bb = self.Bb
        Nr = self.membrane_resultants(u)*self.wp[None, :, None]
        w = self.wp
        ke = np.einsum('p,epki,kl,eplj->eij', w, Bm, self.A, Bm,
                       optimize=True)
        ke += np.einsum('p,pki,kl,plj->ij', w, Bb, self.D, Bb,
                        optimize=True)[None]
        if np.any(self.B):
            kb = np.einsum('p,epki,kl,plj->eij', w, Bm, self.B, Bb,
                           optimize=True)
            ke += kb + kb.transpose(0, 2, 1)
        Gx, Gy = self.Gx, self.Gy
        ke += (np.einsum('ep,pi,pj->eij', Nr[..., 0], Gx, Gx)
               + np.einsum('ep,pi,pj->eij', Nr[..., 1], Gy, Gy)
               + np.einsum('ep,pi,pj->eij', Nr[..., 2], Gx, Gy)
               + np.einsum('ep,pi,pj->eij', Nr[..., 2], Gy, Gx))
        return coo_matrix((ke.ravel(), (self._rows, self._cols)),
                          shape=(self.N, self.N)).tocsc()

    def strain_energy(self, u, u0=None):
        """Strain energy of the strains ``eps(u + u0) - eps(u0)`` and the
        curvatures ``kappa(u)``, independently of the kernels of the
        elements"""
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

    def w_max(self, u):
        """Largest nodal ``|w|``"""
        return np.abs(u[6::DOF]).max()


def symmetrized_KF(model, variant):
    r"""Load stiffness matrix symmetrized by eq. (63), (64) or (65) of the
    paper, ``variant`` ``63``, ``64`` or ``65``, all dofs

    ``63`` returns the symmetric part of ``model.KF``, ``64`` and ``65`` add
    and subtract the substitute of its skew part.

    Notes
    -----
    The skew part `\boldsymbol{S} = (\boldsymbol{K}_F -
    \boldsymbol{K}_F^\top)/2` couples the in-plane dofs (`u, ..., v_{,y}`,
    block `a`) with those of `w` (block `b`), `\boldsymbol{S} =
    [\boldsymbol{0}\ \hat{\boldsymbol{K}};\ -\hat{\boldsymbol{K}}^\top\
    \boldsymbol{0}]`, eq. (62a) of [1]_; the substitute
    `[\boldsymbol{0}\ \hat{\boldsymbol{K}};\ \hat{\boldsymbol{K}}^\top\
    \boldsymbol{0}]`, eq. (62b), is added (64) or subtracted (65) to the
    symmetric part `(\boldsymbol{K}_F + \boldsymbol{K}_F^\top)/2` (63).

    References
    ----------
    .. [1] Schweizerhof, K., and Ramm, E., Displacement dependent pressure
       loads in nonlinear finite element analyses, Computers & Structures,
       18(6), 1099-1114, 1984, doi:10.1016/0045-7949(84)90154-8.

    """
    KF = csc_matrix(model.KF)
    Ksym = (KF + KF.T)/2
    if variant == 63:
        return Ksym.tocsc()
    S = ((KF - KF.T)/2).tocoo()
    inplane = (np.arange(model.N) % DOF) < 6
    ab = inplane[S.row] & ~inplane[S.col]
    Kh = coo_matrix((S.data[ab], (S.row[ab], S.col[ab])),
                    shape=S.shape).tocsc()
    Ksub = Kh + Kh.T
    sign = {64: 1., 65: -1.}[variant]
    return (Ksym + sign*Ksub).tocsc()


def linear_buckling(model, load='follower', num_eigvalues=4, KF=None):
    r"""Buckling factors ``k`` and modes, all dofs, of the linear buckling
    problem of the dead or of the follower pressure

    ``load='dead'`` leaves out the load stiffness matrix, ``'follower'``
    adds the unsymmetric load stiffness matrix of the model or the matrix
    ``KF`` given, e.g. of :func:`symmetrized_KF`. Returns the real parts of
    the load multipliers and of the modes, and the linear static solution
    ``u_lin`` of the dead load of ``k = 1``.

    Notes
    -----
    The eigenvalue problem, solved by :func:`structsolve.lb`, is

    .. math::
        \left(\boldsymbol{K}_{C0} + k \left(\boldsymbol{K}_G
        (\boldsymbol{u}_{lin}) + \boldsymbol{K}_F\right)\right)
        \boldsymbol{\phi} = \boldsymbol{0}

    on the unknown dofs, unsymmetric with the load stiffness matrix of the
    model.

    """
    KC0uu = model.restrict(model.KC0)
    clin = solve(KC0uu, model.fp0[model.bu], silent=True)
    ulin = model.expand(clin)
    KG = model.calc_KG(ulin)
    symmetric = None
    if load == 'follower':
        KF = model.KF if KF is None else KF
        KG = KG + KF
        symmetric = None if KF is not model.KF else False
    eigvals, eigvecs = lb(KC0uu, model.restrict(KG),
                          num_eigvalues=num_eigvalues, silent=True,
                          symmetric=symmetric)
    modes = np.zeros((model.N, eigvecs.shape[1]))
    modes[model.bu] = np.real(eigvecs)
    return np.real(eigvals), modes, ulin


def num_waves(model, mode):
    """Number of circumferential waves of ``mode``, the dominant harmonic
    of `w` along the free edge: ``cos(k y/R)`` and ``sin(k y/R)`` of any
    ``k`` for the closed cylinder, ``cos(j n y/R)`` for the sector"""
    top = np.flatnonzero(isclose(model.x, model.L))
    top = top[np.argsort(model.y[top])]
    w = mode[DOF*top + 6]
    y = model.y[top]
    if model.full:
        w, y = np.append(w, w[0]), np.append(y, 2*pi*R)
        ns = np.arange(0, model.ny//2)
        amp = [np.hypot(np.trapezoid(w*np.cos(k*y/R), y),
                        np.trapezoid(w*np.sin(k*y/R), y)) for k in ns]
    else:
        ns = model.n*np.arange(0, 4)
        amp = [abs(np.trapezoid(w*np.cos(k*y/R), y)) for k in ns]
    return int(ns[np.argmax(amp)])


def imperfection(model, mode, xi):
    """The dofs of `w` of ``mode``, scaled to a largest nodal ``|w|`` of
    ``xi*t``, outwards at the free edge"""
    u0 = np.zeros_like(mode)
    for dof in (6, 7, 8, 9):
        u0[dof::DOF] = mode[dof::DOF]
    w = u0[6::DOF]
    return u0*xi*model.t/w[np.argmax(np.abs(w))]


class FollowerCylinder(object):
    r"""Callables of structsolve.Analysis for the cantilever with the
    initial imperfection ``u0`` under the pressure of the load factor ``k``

    The callables work on the unknown dofs ``c = u[bu]`` and receive the load
    factor ``inc`` from structsolve:

    - ``calc_fext(inc) = inc*fp(u0)``, the force vector of the pressure on the
      imperfect shell, and ``calc_fext(inc, c) = inc*fp(u + u0)``, the
      load vector of the current configuration, ``-dR/dk``, which
      structsolve uses in the arc-length methods when ``calc_fext``
      declares ``c``;
    - ``calc_fint(c, inc) = fint(u + u0) - fint(u0) - KG(N0) u - inc*(fp(u +
      u0) - fp(u0))``, such that the residual of structsolve is that of the
      follower pressure, ``fint_imp(u) - inc*fp(u + u0)``;
    - ``calc_kC(c, NLgeom=True, inc) = KC0 + KCNL(u + u0) - KGNL0 + inc*KF``,
      and ``calc_kC()`` the linear stiffness matrix of the imperfect shell,
      ``KC0 + KCNL(u0) - KGNL0 + KG(u0)``, without ``KF``;
    - ``calc_kG(c) = KG(u)``,

    with the terms of the imperfection of ``ImperfectCylinder`` of
    ``simo1990_cylinder.py``. ``load='dead'`` leaves out the terms of
    ``fp(u)`` and ``KF``. With ``vectorized=True``, the default, ``KC0 +
    KCNL(u + u0)`` is computed as ``KT(u + u0) - KG(u + u0)`` with
    :meth:`CantileverModel.calc_KT`, the same matrix in a fraction of the
    time of ``update_KCNL``; ``False`` calls the kernel.

    Parameters
    ----------
    model : :class:`CantileverModel`
    u0 : array or None
        Degrees of freedom of the imperfection, all dofs; ``None`` for the
        perfect shell.
    load : str
        ``'follower'`` or ``'dead'``.
    vectorized : bool
        Computes the constitutive matrices with
        :meth:`CantileverModel.calc_KT`.

    Notes
    -----
    The residual of structsolve, ``calc_fint(c, inc) - calc_fext(inc)``, is
    that of the follower pressure on the imperfect shell, `\boldsymbol{R} =
    \boldsymbol{F}_{int,imp}(\boldsymbol{u}) - k \boldsymbol{F}_p
    (\boldsymbol{u} + \boldsymbol{u}_0)`, and ``calc_kC + calc_kG`` is its
    Jacobian. With the first-order area vector, `\boldsymbol{F}_p` is affine
    in the displacements and `\boldsymbol{K}_F` constant. The load vector
    of the current configuration, `\boldsymbol{F}_p(\boldsymbol{u} +
    \boldsymbol{u}_0) = -\partial \boldsymbol{R}/\partial k`, returned by
    ``calc_fext`` when it receives ``c``, gives the bordered system of the
    arc-length methods the derivative with respect to the load factor; with
    the constant reference vector instead the iterations converge linearly.

    """
    def __init__(self, model, u0=None, load='follower', vectorized=True):
        self.model = m = model
        self.vectorized = vectorized
        self.load = load
        self.u0 = np.zeros(m.N) if u0 is None else np.asarray(u0, float)
        self.fint0 = m.calc_fint(self.u0)
        N0NL = (m.membrane_resultants(self.u0, nonlinear=True)
                - m.membrane_resultants(self.u0, nonlinear=False))
        self.KGNL0 = m.calc_kG_N(N0NL)
        self.KG0 = m.calc_KG(self.u0)
        self.follower = load == 'follower'
        f = m.calc_fp(self.u0) if self.follower else m.fp0
        self.fextuu = f[m.bu]
        self.KFuu = m.restrict(m.KF)
        self.K0uu = m.restrict(m.KC0 + m.calc_KCNL(self.u0) - self.KGNL0
                               + self.KG0)

    def calc_fext(self, inc=1., c=None, silent=True):
        """Force vector of the pressure of the load factor ``inc``, unknown
        dofs, on the imperfect shell or, with ``c``, at the current
        configuration"""
        if c is None or not self.follower:
            return inc*self.fextuu
        # the load vector of the current configuration, -dR/dk, for the
        # predictors and the constraint of the arc-length methods
        m = self.model
        return inc*m.calc_fp(m.expand(c) + self.u0)[m.bu]

    def calc_fint(self, c, inc=1., silent=True):
        """Internal forces of the imperfect shell minus the part of the
        follower force that depends on ``c``, unknown dofs"""
        m = self.model
        u = m.expand(c)
        f = (m.calc_fint(u + self.u0) - self.fint0 - self.KG0 @ u
             - self.KGNL0 @ u)
        if self.follower:
            # fp(u + u0) - fp(u0), affine in u for quadratic=0
            f -= inc*m.calc_fp(u, reference=0)
        return f[m.bu]

    def calc_kC(self, c=None, NLgeom=False, inc=1., silent=True):
        """Constitutive part of the tangent stiffness matrix with the load
        stiffness matrix of the load factor ``inc``, unknown dofs"""
        if c is None or not NLgeom:
            # the linear stiffness matrix of the imperfect shell, without
            # load stiffness: structsolve uses it for the linear reference
            # solution, the convergence weights and the scaling of the
            # count of negative eigenvalues
            return self.K0uu
        else:
            m = self.model
            u = m.expand(c)
            if self.vectorized:
                # KG(u + u0) = KG(u) + KG0, KG linear in u
                K = m.restrict(m.calc_KT(u + self.u0) - m.calc_KG(u)
                               - self.KG0 - self.KGNL0)
            else:
                K = m.restrict(m.KC0 + m.calc_KCNL(u + self.u0)
                               - self.KGNL0)
        if self.follower:
            K = K + inc*self.KFuu
        return K

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        """``KG(u)`` of :func:`.update_KG`, unknown dofs"""
        m = self.model
        return m.restrict(m.calc_KG(m.expand(c)))


def run_riks(shell=1, bc='B1', nx=16, ny=4, xi=0.01, load='follower',
             kinematics='sanders', wmax=3., inertia_num_eigvals=6,
             silent=True, **kwargs):
    r"""Riks analysis of the imperfect cantilever until the largest ``|w|``
    reaches ``wmax*t``

    The imperfection has the amplitude ``xi*t`` in the shape of the lowest
    linear buckling mode of the same ``load``. ``kwargs`` are attributes of
    :class:`structsolve.Analysis`. Returns a dict with the load factors
    ``ks``, the largest ``|w|/t`` ``wt``, from the unloaded state, the
    current stiffness parameter ``cs_param`` of each increment, ``dk/|du|``
    over its value of the first increment, with the Euclidean norm of the
    increment of the unknown dofs (after Bergan, 1980),
    the counts of negative eigenvalues ``neg`` and the determinant signs
    ``det``, of the converged increments, the linear buckling factor
    ``k_lin``, the run time ``elapsed``, the analysis ``an``, the model and
    the imperfection ``u0``.

    Notes
    -----
    With ``inertia=True``, ``linear_solver='splu'`` and
    ``inertia_num_eigvals``, structsolve factorizes the tangent stiffness
    matrix of every converged increment with SuperLU and records the sign
    of its determinant and, since the matrix is unsymmetric, the number of
    negative real eigenvalues among the ``inertia_num_eigvals`` of smallest
    magnitude, by shift-invert with ARPACK on the LU factorization. The
    count increases by one at a limit point of the load beyond which the
    path is unstable. The Riks method follows Riks [1]_.

    References
    ----------
    .. [1] Riks, E., An incremental approach to the solution of snapping and
       buckling problems, International Journal of Solids and Structures,
       15(7), 529-551, 1979, doi:10.1016/0020-7683(79)90081-7.

    """
    model = CantileverModel(shell, bc, nx, ny, kinematics)
    eigvals, modes, _ = linear_buckling(model, load)
    u0 = imperfection(model, modes[:, 0], xi)
    cb = FollowerCylinder(model, u0, load)
    an = Analysis(cb.calc_fext, cb.calc_fint, cb.calc_kC, cb.calc_kG)
    an.NL_method = 'arc_length_riks'
    an.final_load_factor = None
    # first increment 0.5 of k; the arc length then grows with the number
    # of iterations, the limit of maxInc being lifted
    an.initialInc = 0.5
    an.maxInc = 100.
    an.minInc = 1e-4
    an.maxArcLength = 1e9
    an.inertia = True
    an.inertia_num_eigvals = inertia_num_eigvals
    an.linear_solver = 'splu'
    an.stop_criterion = lambda an: (
        model.w_max(model.expand(an.cs[-1])) >= wmax*model.t)
    for k, v in kwargs.items():
        setattr(an, k, v)
    t0 = time.perf_counter()
    an.static(NLgeom=True, silent=silent)
    elapsed = time.perf_counter() - t0
    wt = np.array([0.] + [model.w_max(model.expand(c)) for c in an.cs])
    ks = np.concatenate(([0.], an.increments))
    # current stiffness parameter: the stiffness dk/|du| of each increment,
    # with the Euclidean norm of the increment of the unknown dofs, over
    # that of the first one
    cs = np.vstack((np.zeros(model.bu.sum()), np.array(an.cs)))
    stiff = np.diff(ks)/np.linalg.norm(np.diff(cs, axis=0), axis=1)
    return dict(ks=ks, wt=wt/model.t, cs_param=stiff/stiff[0],
                neg=list(an.num_negative_eigvals), det=list(an.det_signs),
                k_lin=eigvals[0], elapsed=elapsed, an=an, model=model, u0=u0)
