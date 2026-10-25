r"""Stringer-stiffened cylinder of BFSC elements

The skin is modelled with the BFSC cylinder elements of ``bfsccylinder``,
Sanders-type or Donnell-type, and each blade stringer with BFSC plate
elements of ``bfscplate2d``, in the plane `(x, r)` of the stringer, from the
middle surface of the skin to the tip of the stringer. The plate of a
stringer has the local coordinates `X = x` and `Y = r - R`, positive
outwards, and its displacements `U`, `V` and `W` are along `x`, along the
outward radius and along the circumference, the direction of `v` of the
skin.

Along the line where a stringer meets the skin, `Y = 0`, the nodes of the
stringer are tied to the nodes of the skin, the translations and the
rotation about `x` being the same:

.. math::

    U = u, \quad V = w, \quad W = v, \quad W_{,Y} = \beta_y = -w_{,y} +
    \frac{v}{R}

and the derivatives of these relations along `x`: `U_{,X} = u_{,x}`,
`V_{,X} = w_{,x}`, `W_{,X} = v_{,x}`, `W_{,XY} = -w_{,xy} + v_{,x}/R`. The
rotation `\beta_y` is that of the Sanders-type kinematics, with which
rigid-body motions of the stiffened cylinder give no strains; it is also
used with the Donnell-type skin, whose kinematics neglect `v/R`. The
derivatives `U_{,Y}` and `V_{,Y}` of the stringer, its in-plane shear and
radial strains at the root, stay free. The 8 tied degrees of freedom of
each root node are eliminated with the transformation
`\{u\} = [T]\{u_r\}`, and the matrices of the reduced degrees of freedom
`\{u_r\}` are `[T]^T [K] [T]`, `[T]^T [K_G] [T]` and `[T]^T [M] [T]`.

The model is the half of the cylinder `0 \le \theta \le \pi` with two
planes of symmetry midway between two stringers, or the whole cylinder.
"""
import numpy as np
from numpy import pi
from scipy.sparse import coo_matrix

from composites import isotropic_plate
import bfsccylinder
import bfsccylinder.sanders
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD
import bfscplate2d
from bfscplate2d import BFSCPlate2D

DOF = 10
# positions of the degrees of freedom in a node
U, U_X, U_Y, V, V_X, V_Y, W, W_X, W_Y, W_XY = range(10)


class StiffenedCylinder:
    r"""Skin of BFSC cylinder elements with blade stringers of BFSC plate
    elements

    Parameters
    ----------
    R, L, h : float
        Radius of the middle surface, length and thickness of the skin.
    E, nu, rho : float
        Young's modulus, Poisson's ratio and density of the skin and of the
        stringers.
    nstringers : int
        Number of stringers, equally spaced, even when ``half=True``.
    bs, Hs : float
        Thickness of a stringer and height of its plate, from the middle
        surface of the skin to the tip.
    side : str
        ``'external'`` or ``'internal'``.
    nx : int
        Number of nodes along `x`.
    nbay : int
        Number of skin elements between two stringers, even.
    nh : int
        Number of stringer elements along the height.
    kinematics : str
        ``'sanders'`` or ``'donnell'``, of the skin elements.
    half : bool
        Half model with two planes of symmetry, or the whole cylinder.
    free_axial : bool
        With ``True``, the mean of `u` along the edge `x = 0` is tied to
        zero, which removes the rigid-body axial translation of the boundary
        conditions with `u` free at both ends.
    """

    def __init__(self, R, L, h, E, nu, rho, nstringers, bs, Hs, side, nx,
                 nbay=2, nh=2, kinematics='sanders', half=True,
                 free_axial=False, nint=4):
        assert side in ('external', 'internal')
        assert kinematics in ('sanders', 'donnell')
        assert nbay % 2 == 0
        if half:
            assert nstringers % 2 == 0
        self.R, self.L, self.h = R, L, h
        self.E, self.nu, self.rho = E, nu, rho
        self.nstringers, self.bs, self.Hs, self.side = nstringers, bs, Hs, side
        self.nx, self.nbay, self.nh = nx, nbay, nh
        self.kinematics, self.half = kinematics, half
        if kinematics == 'sanders':
            self.mod = bfsccylinder.sanders
            Element = bfsccylinder.sanders.BFSCCylinderSanders
        else:
            self.mod = bfsccylinder
            Element = bfsccylinder.BFSCCylinder
        self.points, self.weights = get_points_weights(nint=nint)

        # skin nodes, x varying slowest
        dtheta = 2*pi/nstringers/nbay
        ny_el = nstringers*nbay//2 if half else nstringers*nbay
        nyn = ny_el + 1 if half else ny_el
        self.nyn = nyn
        self.theta = np.arange(nyn)*dtheta
        self.xlin = np.linspace(0, L, nx)
        skin = np.arange(nx*nyn).reshape(nx, nyn)
        self.skin_nodes = skin
        num_skin_nodes = nx*nyn
        self.x = np.repeat(self.xlin, nyn)
        self.y = np.tile(self.theta, nx)*R

        # stringers: positions midway between the planes of symmetry
        nstr_model = nstringers//2 if half else nstringers
        self.j_stringer = np.arange(nstr_model)*nbay + nbay//2
        self.theta_stringer = self.theta[self.j_stringer]
        sign = 1. if side == 'external' else -1.
        self.Y = sign*np.arange(nh + 1)*Hs/nh
        str_nodes = (num_skin_nodes
                     + np.arange(nstr_model*nx*(nh + 1)).reshape(
                         nstr_model, nx, nh + 1))
        self.str_nodes = str_nodes
        num_nodes = num_skin_nodes + str_nodes.size
        self.N = N = DOF*num_nodes

        # elements
        lex = L/(nx - 1)
        self.skin_elements = []
        for i in range(nx - 1):
            for j in range(ny_el):
                jp = (j + 1) % nyn if not half else j + 1
                shell = Element(nint)
                shell.n1 = skin[i, j]
                shell.n2 = skin[i + 1, j]
                shell.n3 = skin[i + 1, jp]
                shell.n4 = skin[i, jp]
                shell.c1 = DOF*shell.n1
                shell.c2 = DOF*shell.n2
                shell.c3 = DOF*shell.n3
                shell.c4 = DOF*shell.n4
                shell.R = R
                shell.lex = lex
                shell.ley = R*dtheta
                shell.h = h
                shell.rho = rho
                self.skin_elements.append(shell)
        self.skin_prop = isotropic_plate(thickness=h, E=E, nu=nu, rho=rho)
        for shell in self.skin_elements:
            assign_constant_ABD(shell, self.skin_prop)
        self.str_prop = isotropic_plate(thickness=bs, E=E, nu=nu, rho=rho)
        self.str_elements = []
        for s in range(nstr_model):
            for i in range(nx - 1):
                for k in range(nh):
                    # node 1 at the smallest X and Y
                    k1, k4 = (k, k + 1) if sign > 0 else (k + 1, k)
                    plate = BFSCPlate2D()
                    plate.n1 = str_nodes[s, i, k1]
                    plate.n2 = str_nodes[s, i + 1, k1]
                    plate.n3 = str_nodes[s, i + 1, k4]
                    plate.n4 = str_nodes[s, i, k4]
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

        # transformation u = T u_r
        dependent = {}
        for s in range(nstr_model):
            for i in range(nx):
                c = DOF*skin[i, self.j_stringer[s]]
                b = DOF*str_nodes[s, i, 0]
                dependent[b + U] = [(c + U, 1.)]
                dependent[b + U_X] = [(c + U_X, 1.)]
                dependent[b + V] = [(c + W, 1.)]
                dependent[b + V_X] = [(c + W_X, 1.)]
                dependent[b + W] = [(c + V, 1.)]
                dependent[b + W_X] = [(c + V_X, 1.)]
                dependent[b + W_Y] = [(c + W_Y, -1.), (c + V, 1/R)]
                dependent[b + W_XY] = [(c + W_XY, -1.), (c + V_X, 1/R)]
        if free_axial:
            # trapezoidal weights of the edge x = 0
            wts = np.ones(nyn)
            if half:
                wts[0] = wts[-1] = 0.5
            c0 = DOF*skin[0, 0]
            dependent[c0 + U] = [(DOF*skin[0, j] + U, -wts[j]/wts[0])
                                 for j in range(1, nyn)]
        independent = np.ones(N, dtype=bool)
        independent[list(dependent.keys())] = False
        self.Nr = Nr = independent.sum()
        red = -np.ones(N, dtype=int)
        red[independent] = np.arange(Nr)
        self.red = red
        rows = list(np.flatnonzero(independent))
        cols = list(red[independent])
        vals = [1.]*Nr
        for d, terms in dependent.items():
            for d2, coef in terms:
                assert red[d2] >= 0
                rows.append(d)
                cols.append(red[d2])
                vals.append(coef)
        self.T = coo_matrix((vals, (rows, cols)), shape=(N, Nr)).tocsr()

    # assembly of the matrices of all the degrees of freedom
    def _assemble(self, skin_func, skin_size, str_func, str_size):
        nse = len(self.skin_elements)
        nst = len(self.str_elements)
        r = np.zeros(skin_size*nse + str_size*nst, dtype=self.mod.INT)
        c = np.zeros_like(r)
        v = np.zeros(r.shape[0], dtype=float)
        init_k = 0
        for shell in self.skin_elements:
            skin_func(shell, init_k, r, c, v)
            init_k += skin_size
        for plate in self.str_elements:
            str_func(plate, init_k, r, c, v)
            init_k += str_size
        return coo_matrix((v, (r, c)), shape=(self.N, self.N)).tocsr()

    def reduce(self, A):
        """Matrix of the reduced degrees of freedom, `[T]^T [A] [T]`"""
        return (self.T.T @ A @ self.T).tocsr()

    def KC0(self):
        """Linear stiffness matrix, all the degrees of freedom"""
        m, p, w = self.mod, self.points, self.weights

        def skin(shell, k, r, c, v):
            shell.init_k_KC0 = k
            m.update_KC0(shell, p, w, r, c, v)

        def stringer(plate, k, r, c, v):
            plate.init_k_KC0 = k
            bfscplate2d.update_KC0(plate, p, w, r, c, v)

        return self._assemble(skin, m.KC0_SPARSE_SIZE, stringer,
                              bfscplate2d.KC0_SPARSE_SIZE)

    def M(self):
        """Mass matrix, all the degrees of freedom"""
        m = self.mod

        def skin(shell, k, r, c, v):
            shell.init_k_M = k
            m.update_M(shell, r, c, v)

        def stringer(plate, k, r, c, v):
            plate.init_k_M = k
            bfscplate2d.update_M(plate, r, c, v)

        return self._assemble(skin, m.M_SPARSE_SIZE, stringer,
                              bfscplate2d.M_SPARSE_SIZE)

    def KG(self, u):
        """Geometric stiffness matrix of the displacements ``u`` of all the
        degrees of freedom, with the in-plane terms of the stringers of
        :meth:`KG_inplane`"""
        m, p, w = self.mod, self.points, self.weights

        def skin(shell, k, r, c, v):
            shell.init_k_KG = k
            m.update_KG(u, shell, p, w, r, c, v)

        def stringer(plate, k, r, c, v):
            plate.init_k_KG = k
            bfscplate2d.update_KG(u, plate, p, w, r, c, v)

        KG = self._assemble(skin, m.KG_SPARSE_SIZE, stringer,
                            bfscplate2d.KG_SPARSE_SIZE)
        return KG + self.KG_inplane(self.stringer_membrane_forces(u))

    def KG_membrane(self, eps):
        r"""Geometric stiffness matrix of the uniform axial strain ``eps``
        of the membrane pre-buckling state, `N_{xx} = E h \epsilon` in the
        skin and `E b_s \epsilon` in the stringers, without hoop stress,
        with the in-plane terms of the stringers of :meth:`KG_inplane`"""
        m, p, w = self.mod, self.points, self.weights
        Nskin = self.E*self.h*eps
        Nstr = self.E*self.bs*eps

        def skin(shell, k, r, c, v):
            shell.init_k_KG = k
            m.update_KG_constant_stress(shell, p, w, r, c, v, Nskin, 0., 0.)

        def stringer(plate, k, r, c, v):
            plate.init_k_KG = k
            bfscplate2d.update_KG_cte_N(Nstr, 0., 0., plate, p, w, r, c, v)

        KG = self._assemble(skin, m.KG_SPARSE_SIZE, stringer,
                            bfscplate2d.KG_SPARSE_SIZE)
        nel, npts = len(self.str_elements), len(p)**2
        N = np.zeros((nel, npts, 3))
        N[:, :, 0] = Nstr
        return KG + self.KG_inplane(N)

    def _stringer_operators(self):
        """Interpolation rows of the stringer elements at the integration
        points, the same for all the stringer elements"""
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
        at their integration points, shape ``(num_elements, nint**2, 3)``,
        from the displacements ``u`` of all the degrees of freedom"""
        Bm, Sv_x, Sv_y, wts, dofs = self._stringer_operators()
        A = self.str_prop.ABD[:3, :3]
        eps = np.einsum('pij,ej->epi', Bm, u[dofs])
        return eps @ A.T

    def KG_inplane(self, N):
        r"""Geometric stiffness matrix of the radial displacement `V` of the
        stringers in their plane, for the membrane forces ``N`` of
        :meth:`stringer_membrane_forces`:

        .. math::

            \int (N_{xx} V_{,X}^2 + 2 N_{xy} V_{,X} V_{,Y} + N_{yy}
            V_{,Y}^2) \, dX \, dY

        These terms, the counterpart of the terms of `w` of the skin, are
        not in the geometric stiffness matrix of the BFSC plate element,
        which has only the terms of its out-of-plane displacement `W`."""
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

    def axial_force_membrane(self, eps):
        """Axial force of the whole cylinder for the membrane state of the
        uniform axial strain ``eps``"""
        return self.E*eps*(2*pi*self.R*self.h
                           + self.nstringers*self.bs*self.Hs)

    # boundary conditions, as boolean arrays of the reduced dofs
    def _mark(self, bk, nodes, dofs):
        for n in np.atleast_1d(nodes).ravel():
            for d in dofs:
                rd = self.red[DOF*n + d]
                if rd >= 0:
                    bk[rd] = True

    def constrained(self, ends='SS'):
        r"""Constrained reduced degrees of freedom

        ``ends='SS'``: `v = w = 0` at both ends, `u` and `w_{,x}` free, and
        `V = W = W_{,Y} = 0` along the end edges of the stringers.
        ``ends='CC'``: `u = v = w = w_{,x} = 0` at both ends and the end
        edges of the stringers clamped. The derivatives along the edges of
        the constrained displacements are constrained as well. In the half
        model, the planes `\theta = 0` and `\theta = \pi` are planes of
        symmetry: `v = v_{,x} = u_{,y} = w_{,y} = w_{,xy} = 0`.
        """
        bk = np.zeros(self.Nr, dtype=bool)
        skin, strn = self.skin_nodes, self.str_nodes
        ends_skin = np.concatenate((skin[0, :], skin[-1, :]))
        ends_str = np.concatenate((strn[:, 0, :], strn[:, -1, :]), axis=None)
        if ends == 'SS':
            self._mark(bk, ends_skin, [V, V_Y, W, W_Y])
            self._mark(bk, ends_str, [V, V_Y, W, W_Y])
        elif ends == 'CC':
            self._mark(bk, ends_skin, [U, U_Y, V, V_Y, W, W_Y, W_X, W_XY])
            self._mark(bk, ends_str, [U, U_Y, V, V_Y, W, W_Y, W_X, W_XY])
        else:
            raise ValueError(ends)
        if self.half:
            sym = np.concatenate((skin[:, 0], skin[:, -1]))
            self._mark(bk, sym, [V, V_X, U_Y, W_Y, W_XY])
        return bk

    def axial_dofs(self, i):
        """Reduced degrees of freedom of `u` and `U` of the cross-section of
        index ``i`` along `x`, and of their derivatives along the edge"""
        bk = np.zeros(self.Nr, dtype=bool)
        nodes = np.concatenate((self.skin_nodes[i, :],
                                self.str_nodes[:, i, :].ravel()))
        self._mark(bk, nodes, [U])
        bkd = np.zeros(self.Nr, dtype=bool)
        self._mark(bkd, nodes, [U_Y])
        return bk, bkd

    def skin_field(self, u, dof=W):
        """Field of one degree of freedom of the skin, shape ``(nx, nyn)``,
        from the displacements ``u`` of all the degrees of freedom"""
        return u[DOF*self.skin_nodes + dof]


def wave_numbers(model, u, mmax=12, nmax=40):
    r"""Axial half-waves `m` and circumferential waves `n` of the normal
    displacement `w` of the skin, from the largest coefficient of its
    expansion in `\sin(m \pi x/L) \cos(n \theta)` (`\theta` from a plane of
    symmetry), for simply supported ends, or in `\cos(n \theta)` and the
    number of sign changes along `x` of the dominant term, for other ends"""
    w = model.skin_field(u, W)
    x = model.xlin
    th = model.theta
    if model.half:
        # even about theta = 0: cosine terms
        wy = np.ones_like(th)
        wy[0] = wy[-1] = 0.5
        ns = np.arange(nmax + 1)
        Cn = (w*wy) @ np.cos(np.outer(th, ns))
    else:
        Cn = np.abs(np.fft.rfft(w, axis=1))[:, :nmax + 1]
    n = int(np.argmax(np.sum(Cn**2, axis=0)))
    wn = Cn[:, n]
    wx = np.ones_like(x)
    wx[0] = wx[-1] = 0.5
    ms = np.arange(1, mmax + 1)
    Sm = (wn*wx) @ np.sin(np.outer(x, ms)*pi/x[-1])
    m = int(ms[np.argmax(np.abs(Sm))])
    return m, n
