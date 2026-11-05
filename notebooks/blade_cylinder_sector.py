r"""Periodic sector of a cylinder with integral blade stringers

A circular cylinder with ``nstringers`` identical blade stringers, equally
spaced, is modelled by one stringer pitch, the sector `0 \le \theta \le
\phi`, `\phi = 2 \pi/n_s`, with the stringer at its middle, `\theta =
\phi/2`. The skin is meshed with the BFSC cylinder element of Sanders-type
kinematics, :class:`bfsccylinder.sanders.BFSCCylinderSanders`, and the
stringer with the BFSC plate element :class:`bfscplate2d.BFSCPlate2D`, in
the plane `(x, r)` of the stringer, from the middle surface of the skin to
the tip. Without stringer, ``bs=None``, the sector is a sector of an
unstiffened cylinder.

The plate of the stringer has the local coordinates `X = x` and `Y = r -
R`, and its displacements `U`, `V` and `W` are along `x`, along the outward
radius and along the circumference. Along its root, `Y = 0`, its nodes are
tied to the skin nodes of the line `\theta = \phi/2`:

.. math::

    U = u, \quad V = w, \quad W = v, \quad W_{,Y} = -w_{,y} + \frac{v}{R}

with the derivatives of these relations along `x`, the rotation about `x`
being that of the Sanders-type kinematics, as in the module
``stiffened_cylinder.py`` of the notebooks of this repository.

Cyclic symmetry
---------------
The structure is invariant under a rotation by `\phi` about its axis, and
the degrees of freedom of the elements, in the axial, circumferential and
radial directions, rotate with it. A mode of the whole cylinder can then be
taken with `n` circumferential waves, the degrees of freedom of the sector
`k` being those of the sector `0` multiplied by `e^{i n k \phi}`: the
equations of the sector `k` are those of the sector `0` multiplied by the
same factor, and the sum of the `n_s` factors is zero unless the harmonics
match, so that the harmonics are uncoupled. The nodes of the edge `\theta
= \phi` are eliminated with

.. math::

    \boldsymbol{u}(\theta = \phi) = e^{i n \phi}\,\boldsymbol{u}(\theta = 0)

which, with the ties of the stringer root, gives the constraint basis
`\boldsymbol{u} = \boldsymbol{Z}_n \boldsymbol{u}_r` and the Hermitian
matrices `\boldsymbol{Z}_n^H \boldsymbol{K} \boldsymbol{Z}_n`. The
harmonics `n = 0, 1, \ldots, n_s/2` give all the modes of the cylinder: the
modes of `n` and `-n` (or `n_s - n`) are the two modes, of equal
eigenvalue, in `\cos n\theta` and `\sin n\theta`, and the harmonic `n`
contains the circumferential waves `n + j n_s` coupled by the discrete
stringers.

"""
import numpy as np
from numpy import pi
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import LinearOperator, eigs, splu, spsolve

from composites import isotropic_plate
import bfsccylinder.sanders as sanders
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD
import bfscplate2d
from bfscplate2d import BFSCPlate2D

DOF = 10
# positions of the degrees of freedom in a node
U, U_X, U_Y, V, V_X, V_Y, W, W_X, W_Y, W_XY = range(10)

#: constrained degrees of freedom of the end nodes, skin and stringer
END_DOFS = {
    'S': [V, V_Y, W, W_Y],
    'C': [U, U_Y, V, V_Y, W, W_Y, W_X, W_XY],
    'F': [],
}


class BladeCylinderSector:
    r"""Sector of one stringer pitch of a blade-stiffened cylinder

    Parameters
    ----------
    R, L, h : float
        Radius of the middle surface, length and thickness of the skin.
    E, nu, rho : float
        Young's modulus, Poisson's ratio and density, skin and stringer.
    nstringers : int
        Number of stringers of the cylinder; the sector angle is
        `2 \pi/n_s`.
    nx : int
        Number of nodes along `x`.
    nbay : int
        Number of skin elements across the sector, even.
    bs, Hs : float or None
        Thickness of the stringer plate (the width of the blade) and its
        height from the middle surface of the skin to the tip. ``None``
        for a sector without stringer.
    side : str
        ``'external'`` or ``'internal'``.
    nh : int
        Number of stringer elements along the height.
    nint : int
        Number of Gauss-Legendre points per direction.
    Dy_factor : float or array of ``nbay`` floats
        Factor of the circumferential bending stiffness `D_{22}` of the
        skin, `D_y/D`, 1 for the isotropic skin; an array gives one factor
        per column of skin elements across the sector, from `\theta = 0`
        to `\theta = \phi`.
    skin_inplane_inertia : bool
        ``False`` removes the inertia of the in-surface displacements `u`,
        `v` of the skin (their rows and columns of the mass matrix); the
        stringer keeps all its inertia.
    """

    def __init__(self, R, L, h, E, nu, rho, nstringers, nx, nbay=4,
                 bs=None, Hs=None, side='external', nh=4, nint=4,
                 Dy_factor=1., skin_inplane_inertia=True):
        assert nbay % 2 == 0
        assert side in ('external', 'internal')
        self.R, self.L, self.h = R, L, h
        self.E, self.nu, self.rho = E, nu, rho
        self.nstringers = nstringers
        self.phi = 2*pi/nstringers
        self.nx, self.nbay, self.nh = nx, nbay, nh
        self.bs, self.Hs, self.side = bs, Hs, side
        self.stiffened = bs is not None
        self.skin_inplane_inertia = skin_inplane_inertia
        Dy = np.broadcast_to(np.asarray(Dy_factor, dtype=float), (nbay,))
        self.points, self.weights = get_points_weights(nint=nint)

        # skin nodes: (nx, nbay + 1), the last column on the edge theta = phi
        self.xlin = np.linspace(0, L, nx)
        self.theta = np.linspace(0, self.phi, nbay + 1)
        skin = np.arange(nx*(nbay + 1)).reshape(nx, nbay + 1)
        self.skin_nodes = skin
        num_nodes = skin.size
        self.j_stringer = nbay//2
        if self.stiffened:
            sign = 1. if side == 'external' else -1.
            self.Y = sign*np.linspace(0, Hs, nh + 1)
            self.str_nodes = num_nodes + np.arange(nx*(nh + 1)).reshape(
                nx, nh + 1)
            num_nodes += self.str_nodes.size
        self.N = DOF*num_nodes

        lex = L/(nx - 1)
        self.skin_prop = isotropic_plate(thickness=h, E=E, nu=nu, rho=rho)
        self.skin_elements = []
        for i in range(nx - 1):
            for j in range(nbay):
                shell = sanders.BFSCCylinderSanders(nint)
                shell.n1 = skin[i, j]
                shell.n2 = skin[i + 1, j]
                shell.n3 = skin[i + 1, j + 1]
                shell.n4 = skin[i, j + 1]
                shell.c1 = DOF*shell.n1
                shell.c2 = DOF*shell.n2
                shell.c3 = DOF*shell.n3
                shell.c4 = DOF*shell.n4
                shell.R = R
                shell.lex = lex
                shell.ley = R*self.phi/nbay
                shell.h = h
                shell.rho = rho
                assign_constant_ABD(shell, self.skin_prop)
                np.asarray(shell.D22)[:] *= Dy[j]
                self.skin_elements.append(shell)
        self.str_elements = []
        if self.stiffened:
            self.str_prop = isotropic_plate(thickness=bs, E=E, nu=nu,
                                            rho=rho)
            for i in range(nx - 1):
                for k in range(nh):
                    # node 1 at the smallest X and Y
                    k1, k4 = (k, k + 1) if sign > 0 else (k + 1, k)
                    plate = BFSCPlate2D()
                    plate.n1 = self.str_nodes[i, k1]
                    plate.n2 = self.str_nodes[i + 1, k1]
                    plate.n3 = self.str_nodes[i + 1, k4]
                    plate.n4 = self.str_nodes[i, k4]
                    plate.c1 = DOF*plate.n1
                    plate.c2 = DOF*plate.n2
                    plate.c3 = DOF*plate.n3
                    plate.c4 = DOF*plate.n4
                    plate.lex = lex
                    plate.ley = Hs/nh
                    plate.h = bs
                    plate.rho = rho
                    plate.ABD = self.str_prop.ABD
                    self.str_elements.append(plate)

    # ------------------------------------------------------------------
    # constraint basis
    def Z(self, n, free_axial=False):
        r"""Constraint basis `\boldsymbol{Z}_n` of the harmonic ``n``

        Real for ``n = 0`` and ``n = n_s/2``, complex otherwise. With
        ``free_axial=True`` and ``n = 0``, the mean of `u` along the skin
        edge `x = 0` is tied to zero, which removes the axial rigid-body
        translation when `u` is free at both ends. Sets the attributes
        ``red`` (index of each degree of freedom among the reduced ones,
        ``-1`` for the eliminated ones) and ``Nr``.
        """
        skin, nbay = self.skin_nodes, self.nbay
        if n == 0:
            phase = 1.
        elif 2*n == self.nstringers:
            phase = -1.
        else:
            phase = np.exp(1j*n*self.phi)
        dependent = {}
        for i in range(self.nx):
            cl = DOF*skin[i, 0]
            cr = DOF*skin[i, nbay]
            for d in range(DOF):
                dependent[cr + d] = [(cl + d, phase)]
        if self.stiffened:
            R = self.R
            for i in range(self.nx):
                c = DOF*skin[i, self.j_stringer]
                b = DOF*self.str_nodes[i, 0]
                dependent[b + U] = [(c + U, 1.)]
                dependent[b + U_X] = [(c + U_X, 1.)]
                dependent[b + V] = [(c + W, 1.)]
                dependent[b + V_X] = [(c + W_X, 1.)]
                dependent[b + W] = [(c + V, 1.)]
                dependent[b + W_X] = [(c + V_X, 1.)]
                dependent[b + W_Y] = [(c + W_Y, -1.), (c + V, 1/R)]
                dependent[b + W_XY] = [(c + W_XY, -1.), (c + V_X, 1/R)]
        if free_axial and n == 0:
            c0 = DOF*skin[0, 0]
            dependent[c0 + U] = [(DOF*skin[0, j] + U, -1.)
                                 for j in range(1, nbay)]
        # express every dependent degree of freedom by independent ones
        changed = True
        while changed:
            changed = False
            for d, terms in dependent.items():
                if any(d2 in dependent for d2, _ in terms):
                    new = []
                    for d2, coef in terms:
                        if d2 in dependent:
                            new += [(d3, coef*c3) for d3, c3 in dependent[d2]]
                        else:
                            new.append((d2, coef))
                    dependent[d] = new
                    changed = True
        independent = np.ones(self.N, dtype=bool)
        independent[list(dependent.keys())] = False
        Nr = int(independent.sum())
        red = -np.ones(self.N, dtype=int)
        red[independent] = np.arange(Nr)
        rows = list(np.flatnonzero(independent))
        cols = list(red[independent])
        vals = [1.]*Nr
        for d, terms in dependent.items():
            for d2, coef in terms:
                assert red[d2] >= 0
                rows.append(d)
                cols.append(red[d2])
                vals.append(coef)
        dtype = complex if isinstance(phase, complex) else float
        self.red, self.Nr = red, Nr
        return coo_matrix((np.array(vals, dtype=dtype), (rows, cols)),
                          shape=(self.N, Nr)).tocsr()

    def constrained(self, ends):
        r"""Constrained reduced degrees of freedom, boolean array

        ``ends`` has two letters, for `x = 0` and `x = L`: ``'S'``, simply
        supported, `v = w = 0` and, for the stringer, `V = W = 0` along its
        end edge; ``'C'``, clamped, `u = v = w = w_{,x} = 0` and the end
        edge of the stringer clamped; ``'F'``, free. The derivatives along
        an edge of its constrained displacements are constrained as well.
        Uses the reduced numbering of the last call of :meth:`Z`.
        """
        bk = np.zeros(self.Nr, dtype=bool)
        for end, i in zip(ends, [0, self.nx - 1]):
            nodes = list(self.skin_nodes[i, :])
            if self.stiffened:
                nodes += list(self.str_nodes[i, :])
            self._mark(bk, nodes, END_DOFS[end])
        return bk

    def end_axial_dofs(self, i):
        """Reduced degrees of freedom of `u` and `U` of the cross-section of
        index ``i`` along `x`, and of their derivatives along the edge"""
        nodes = list(self.skin_nodes[i, :])
        if self.stiffened:
            nodes += list(self.str_nodes[i, :])
        bu = np.zeros(self.Nr, dtype=bool)
        self._mark(bu, nodes, [U])
        buy = np.zeros(self.Nr, dtype=bool)
        self._mark(buy, nodes, [U_Y])
        return bu, buy

    def _mark(self, bk, nodes, dofs):
        for nd in nodes:
            for d in dofs:
                rd = self.red[DOF*nd + d]
                if rd >= 0:
                    bk[rd] = True

    # ------------------------------------------------------------------
    # matrices of all the degrees of freedom of the sector
    def _assemble(self, skin_func, skin_size, str_func, str_size):
        nse = len(self.skin_elements)
        nst = len(self.str_elements)
        r = np.zeros(skin_size*nse + str_size*nst, dtype=sanders.INT)
        c = np.zeros_like(r)
        v = np.zeros(r.shape[0], dtype=float)
        k = 0
        for shell in self.skin_elements:
            skin_func(shell, k, r, c, v)
            k += skin_size
        for plate in self.str_elements:
            str_func(plate, k, r, c, v)
            k += str_size
        return coo_matrix((v, (r, c)), shape=(self.N, self.N)).tocsr()

    def KC0(self):
        """Linear stiffness matrix, computed once"""
        if not hasattr(self, '_KC0'):
            self._KC0 = self._KC0_assemble()
        return self._KC0

    def M(self):
        """Mass matrix, computed once"""
        if not hasattr(self, '_M'):
            self._M = self._M_assemble()
        return self._M

    def _KC0_assemble(self):
        p, w = self.points, self.weights

        def skin(shell, k, r, c, v):
            shell.init_k_KC0 = k
            sanders.update_KC0(shell, p, w, r, c, v)

        def stringer(plate, k, r, c, v):
            plate.init_k_KC0 = k
            bfscplate2d.update_KC0(plate, p, w, r, c, v)

        return self._assemble(skin, sanders.KC0_SPARSE_SIZE, stringer,
                              bfscplate2d.KC0_SPARSE_SIZE)

    def _M_assemble(self):
        def skin(shell, k, r, c, v):
            shell.init_k_M = k
            sanders.update_M(shell, r, c, v)

        def stringer(plate, k, r, c, v):
            plate.init_k_M = k
            bfscplate2d.update_M(plate, r, c, v)

        M = self._assemble(skin, sanders.M_SPARSE_SIZE, stringer,
                           bfscplate2d.M_SPARSE_SIZE)
        if not self.skin_inplane_inertia:
            keep = np.ones(self.N)
            for d in (U, U_X, U_Y, V, V_X, V_Y):
                keep[DOF*self.skin_nodes.ravel() + d] = 0.
            M = (diags(keep) @ M @ diags(keep)).tocsr()
        return M

    def KG_membrane(self, eps):
        r"""Geometric stiffness matrix of the membrane state of the uniform
        axial strain ``eps``, `N_{xx} = E h \epsilon` in the skin and `E b_s
        \epsilon` in the stringer, without hoop stress, with the in-plane
        terms of the stringer of :meth:`KG_inplane`"""
        p, w = self.points, self.weights
        Nskin = self.E*self.h*eps
        Nstr = self.E*self.bs*eps if self.stiffened else 0.

        def skin(shell, k, r, c, v):
            shell.init_k_KG = k
            sanders.update_KG_constant_stress(shell, p, w, r, c, v, Nskin,
                                              0., 0.)

        def stringer(plate, k, r, c, v):
            plate.init_k_KG = k
            bfscplate2d.update_KG_cte_N(Nstr, 0., 0., plate, p, w, r, c, v)

        KG = self._assemble(skin, sanders.KG_SPARSE_SIZE, stringer,
                            bfscplate2d.KG_SPARSE_SIZE)
        if not self.stiffened:
            return KG
        N = np.zeros((len(self.str_elements), len(p)**2, 3))
        N[:, :, 0] = Nstr
        return KG + self.KG_inplane(N)

    def KG(self, u):
        """Geometric stiffness matrix of the real displacements ``u`` of all
        the degrees of freedom of the sector, with the in-plane terms of the
        stringer of :meth:`KG_inplane`"""
        p, w = self.points, self.weights

        def skin(shell, k, r, c, v):
            shell.init_k_KG = k
            sanders.update_KG(u, shell, p, w, r, c, v)

        def stringer(plate, k, r, c, v):
            plate.init_k_KG = k
            bfscplate2d.update_KG(u, plate, p, w, r, c, v)

        KG = self._assemble(skin, sanders.KG_SPARSE_SIZE, stringer,
                            bfscplate2d.KG_SPARSE_SIZE)
        if not self.stiffened:
            return KG
        return KG + self.KG_inplane(self.stringer_membrane_forces(u))

    def _stringer_operators(self):
        if hasattr(self, '_ops'):
            return self._ops
        plate = self.str_elements[0]
        p, w = self.points, self.weights
        Bm, Sv_x, Sv_y, wts = [], [], [], []
        for i, xi in enumerate(p):
            for j, eta in enumerate(p):
                plate.update_Bm(xi, eta)
                plate.update_Sv_x(xi, eta)
                plate.update_Sv_y(xi, eta)
                Bm.append(np.array(plate.Bm))
                Sv_x.append(np.array(plate.Sv_x))
                Sv_y.append(np.array(plate.Sv_y))
                wts.append(w[i]*w[j]*plate.lex*plate.ley/4)
        dofs = np.array([[pl.c1 + d for d in range(DOF)]
                         + [pl.c2 + d for d in range(DOF)]
                         + [pl.c3 + d for d in range(DOF)]
                         + [pl.c4 + d for d in range(DOF)]
                         for pl in self.str_elements])
        self._ops = (np.array(Bm), np.array(Sv_x), np.array(Sv_y),
                     np.array(wts), dofs)
        return self._ops

    def stringer_membrane_forces(self, u):
        """Membrane forces `N_{xx}, N_{yy}, N_{xy}` of the stringer elements
        at their integration points, shape ``(num_elements, nint**2, 3)``"""
        Bm, Sv_x, Sv_y, wts, dofs = self._stringer_operators()
        A = self.str_prop.ABD[:3, :3]
        eps = np.einsum('pij,ej->epi', Bm, u[dofs])
        return eps @ A.T

    def KG_inplane(self, N):
        r"""Geometric stiffness of the radial displacement `V` of the
        stringer in its plane, for the membrane forces ``N`` of
        :meth:`stringer_membrane_forces`:

        .. math::

            \int (N_{xx} V_{,X}^2 + 2 N_{xy} V_{,X} V_{,Y} + N_{yy}
            V_{,Y}^2) \, dX \, dY

        These terms, the counterpart of those of `w` of the skin, are not
        in the geometric stiffness matrix of the BFSC plate element, which
        has only the terms of its out-of-plane displacement `W`."""
        Bm, Sv_x, Sv_y, wts, dofs = self._stringer_operators()
        Nw = N*wts[None, :, None]
        Ke = (np.einsum('ep,pi,pj->eij', Nw[:, :, 0], Sv_x, Sv_x)
              + np.einsum('ep,pi,pj->eij', Nw[:, :, 1], Sv_y, Sv_y)
              + np.einsum('ep,pi,pj->eij', Nw[:, :, 2], Sv_x, Sv_y)
              + np.einsum('ep,pi,pj->eij', Nw[:, :, 2], Sv_y, Sv_x))
        r = np.repeat(dofs, dofs.shape[1], axis=1).ravel()
        c = np.tile(dofs, (1, dofs.shape[1])).ravel()
        return coo_matrix((Ke.ravel(), (r, c)),
                          shape=(self.N, self.N)).tocsr()

    def area(self):
        """Cross-sectional area of the whole cylinder"""
        A = 2*pi*self.R*self.h
        if self.stiffened:
            A += self.nstringers*self.bs*self.Hs
        return A

    # ------------------------------------------------------------------
    # fields of a mode of the harmonic n
    def skin_field(self, u, dof=W):
        """Field of one degree of freedom of the skin of the sector, shape
        ``(nx, nbay + 1)``"""
        return u[DOF*self.skin_nodes + dof]

    def full_circumference(self, field, n):
        r"""Real field over the whole circumference from the field of the
        sector ``field``, shape ``(nx, nbay + 1)``, of the harmonic ``n``:
        returns `\theta`, shape ``(n_s nbay + 1,)``, and the field, shape
        ``(nx, n_s nbay + 1)``"""
        parts = [field[:, :-1]*np.exp(1j*n*k*self.phi)
                 for k in range(self.nstringers)]
        parts.append(field[:, :1]*np.exp(1j*n*2*pi))
        theta = np.concatenate([self.theta[:-1] + k*self.phi
                                for k in range(self.nstringers)]
                               + [[2*pi]])
        return theta, np.real(np.hstack(parts))


def normalize_phase(u):
    """Mode ``u`` multiplied by the unit complex number that makes its
    largest entry real and positive"""
    i = np.argmax(np.abs(u))
    ui = u.ravel()[i]
    return u*np.conj(ui)/abs(ui)


def lowest_eigenpairs(A, B, k, sigma=0.):
    r"""The ``k`` lowest eigenvalues `\lambda > \sigma` of `\boldsymbol{A}
    \boldsymbol{x} = \lambda \boldsymbol{B} \boldsymbol{x}`, Hermitian
    matrices with `\boldsymbol{A} - \sigma \boldsymbol{B}` nonsingular, and
    their eigenvectors, from the largest eigenvalues `1/(\lambda - \sigma)`
    of `(\boldsymbol{A} - \sigma \boldsymbol{B})^{-1} \boldsymbol{B}`
    (ARPACK through :func:`scipy.sparse.linalg.eigs`)"""
    lu = splu((A - sigma*B).tocsc())
    dtype = np.result_type(A.dtype, B.dtype)
    op = LinearOperator(A.shape, matvec=lambda x: lu.solve(B @ x),
                        dtype=dtype)
    k = min(k, A.shape[0] - 2)
    ncv = min(A.shape[0] - 1, max(2*k + 1, 20))
    nu, X = eigs(op, k=k, which='LR', ncv=ncv, tol=1e-8)
    nu = nu.real
    keep = nu > 0
    nu, X = nu[keep], X[:, keep]
    order = np.argsort(-nu)
    return sigma + 1/nu[order], X[:, order]


def frequencies(model, ends, n, k=8):
    """Natural frequencies (Hz) and modes of the harmonic ``n`` for the
    end conditions ``ends`` (see :meth:`BladeCylinderSector.constrained`);
    the modes are returned for all the degrees of freedom of the sector.
    For ``ends='SS'`` and ``n = 0`` the axial rigid-body translation is
    removed (``free_axial``)."""
    Zn = model.Z(n, free_axial=(ends == 'SS'))
    K = (Zn.conj().T @ model.KC0() @ Zn).tocsr()
    M = (Zn.conj().T @ model.M() @ Zn).tocsr()
    bu = ~model.constrained(ends)
    Kuu = K[bu][:, bu]
    Muu = M[bu][:, bu]
    sigma = -1. if 'C' not in ends else 0.
    omega2, X = lowest_eigenpairs(Kuu, Muu, k, sigma=sigma)
    ur = np.zeros((model.Nr, X.shape[1]), dtype=X.dtype)
    ur[bu] = X
    modes = Zn @ ur
    f = np.sqrt(np.maximum(omega2, 0.))/(2*pi)
    return f, modes


def buckling_membrane(model, ends, n, k=6):
    r"""Linear buckling of the harmonic ``n`` with the membrane pre-buckling
    state of a uniform axial strain; returns the critical average axial
    force per unit circumference `\bar{N} = P/(2 \pi R)` and the modes"""
    eps = -1e-3
    if not hasattr(model, '_KG_membrane'):
        model._KG_membrane = model.KG_membrane(eps)
    Zn = model.Z(n, free_axial=(ends == 'SS'))
    K = (Zn.conj().T @ model.KC0() @ Zn).tocsr()
    KG = (Zn.conj().T @ model._KG_membrane @ Zn).tocsr()
    bu = ~model.constrained(ends)
    lam, X = lowest_eigenpairs(K[bu][:, bu], -KG[bu][:, bu], k)
    ur = np.zeros((model.Nr, X.shape[1]), dtype=X.dtype)
    ur[bu] = X
    P = model.E*abs(eps)*model.area()
    return lam*P/(2*pi*model.R), Zn @ ur


def prebuckling_clamped(model, delta=1e-3):
    r"""Linear static axisymmetric state of the cylinder clamped at both
    ends, the end `x = L` shortened by ``delta``; returns the displacements
    of all the degrees of freedom of the sector and the axial force `P` of
    the whole cylinder"""
    Z0 = model.Z(0)
    K = (Z0.T @ model.KC0() @ Z0).tocsr()
    bk = model.constrained('CC')
    bu_end, buy_end = model.end_axial_dofs(model.nx - 1)
    bk = bk | bu_end | buy_end
    bu = ~bk
    ur = np.zeros(model.Nr)
    ur[bu_end] = -delta
    ur[bu] = spsolve(K[bu][:, bu].tocsc(), -K[bu][:, bk] @ ur[bk])
    P = model.nstringers*(K @ ur)[bu_end].sum()
    return Z0 @ ur, P


def buckling_prebuckled(model, u0, P, n, k=6):
    r"""Linear buckling of the harmonic ``n`` of the clamped cylinder with
    the pre-buckling state ``u0`` of :func:`prebuckling_clamped`, the end
    `x = L` held at its pre-buckling position; returns `\bar{N}` and the
    modes"""
    if getattr(model, '_KG_u0', (None,))[0] is not u0:
        model._KG_u0 = (u0, model.KG(u0))
    KG0 = model._KG_u0[1]
    Zn = model.Z(n)
    K = (Zn.conj().T @ model.KC0() @ Zn).tocsr()
    KG = (Zn.conj().T @ KG0 @ Zn).tocsr()
    bk = model.constrained('CC')
    bu_end, buy_end = model.end_axial_dofs(model.nx - 1)
    bu = ~(bk | bu_end | buy_end)
    lam, X = lowest_eigenpairs(K[bu][:, bu], -KG[bu][:, bu], k)
    ur = np.zeros((model.Nr, X.shape[1]), dtype=X.dtype)
    ur[bu] = X
    return lam*abs(P)/(2*pi*model.R), Zn @ ur


def axial_nodes(model, mode, tol=0.02):
    """Number of sign changes along `x` of the normal displacement `w` of
    the skin, on the line of the sector where `w` is largest, values below
    ``tol`` times the largest one ignored"""
    w = normalize_phase(model.skin_field(mode, W))
    j = np.argmax(np.sum(np.abs(w)**2, axis=0))
    f = np.real(w[:, j])
    f = f[np.abs(f) > tol*np.abs(f).max()]
    return int(np.sum(np.sign(f[1:]) != np.sign(f[:-1])))


def axial_nodes_lines(model, mode, tol=0.02):
    """Number of sign changes along `x` of the normal displacement `w` of
    the skin on each line of nodes `0 \\le \\theta < \\phi` of the sector,
    each line multiplied by the unit complex number that makes its largest
    entry real and positive, values below ``tol`` times the largest one of
    the line ignored; returns the numbers and the mean square of `w` of
    each line"""
    w = model.skin_field(mode, W)[:, :-1]
    counts, msq = [], []
    for j in range(w.shape[1]):
        f = np.real(normalize_phase(w[:, j]))
        msq.append(np.mean(np.abs(w[:, j])**2))
        f = f[np.abs(f) > tol*np.abs(f).max()]
        counts.append(int(np.sum(np.sign(f[1:]) != np.sign(f[:-1]))))
    return np.array(counts), np.array(msq)
