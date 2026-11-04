r"""Laminated stringer-stiffened cylinders and curved panels of BFSC elements

The skin is modelled with the BFSC cylinder element of Sanders-type
kinematics, :class:`bfsccylinder.sanders.BFSCCylinderSanders`, over the
developed mid-surface of radius `R`, with `x` the axial coordinate and `y =
R \theta` the arc length. A stringer is assembled from plates:

- a plate bonded on the skin (the foot of an L- or T-section, a reinforcing
  strip of the skin) is a laminate of cylinder elements on the nodes of the
  skin, its stiffness matrices `\boldsymbol{A}`, `\boldsymbol{B}`,
  `\boldsymbol{D}` referred to the mid-surface of the skin through the
  offset of its plies (a perfect bond under the Kirchhoff hypothesis: the
  skin and the plate share the normal); the matrices of the skin and of
  these plates add up;
- the blade is a plate of :class:`bfscplate2d.BFSCPlate2D` elements in the
  plane `(x, r)` of the stringer, from the mid-surface of the skin to the
  tip, with the local coordinates `X = x` and `Y = r - R`, positive
  outwards, and the displacements `U`, `V` and `W` along `x`, along the
  outward radius and along `y`. Its root, `Y = 0`, is connected to the skin
  along their common line, at the edge of the foot of an L-section or at
  the middle of the foot of a T-section:

.. math::

    U = u, \quad V = w, \quad W = v, \quad W_{,Y} = -w_{,y} + \frac{v}{R}

with the derivatives of these relations along `x`, and `U_{,Y}`, `V_{,Y}`
free. The constraints, the boundary conditions and the rigid end rings are
imposed with a constraint basis `\boldsymbol{u} = \boldsymbol{Z}
\boldsymbol{u}_r`, and the matrices of the reduced degrees of freedom are
`\boldsymbol{Z}^\top \boldsymbol{K} \boldsymbol{Z}`.

The geometric stiffness matrix of the BFSC plate element has the terms of
its out-of-plane displacement `W` only; the terms of the in-plane radial
displacement `V` of a blade under its membrane forces,

.. math::

    \int (N_{xx} V_{,X}^2 + 2 N_{xy} V_{,X} V_{,Y} + N_{yy} V_{,Y}^2) \,
    dX \, dY

are added by :meth:`StiffenedShell.KG_inplane`.
"""
import numpy as np
from numpy import pi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from structsolve import lb

import bfsccylinder.sanders as sanders
from bfsccylinder.sanders import BFSCCylinderSanders
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD
import bfscplate2d
from bfscplate2d import BFSCPlate2D

DOF = 10
# positions of the degrees of freedom in a node
U, U_X, U_Y, V, V_X, V_Y, W, W_X, W_Y, W_XY = range(10)


def node_coordinates(segments):
    """Node coordinates from a list of ``(length, number of elements)``,
    starting at zero"""
    y = [0.]
    for length, nel in segments:
        y.extend(y[-1] + length*np.arange(1, nel + 1)/nel)
    return np.array(y)


class StiffenedShell:
    r"""Skin of BFSC cylinder elements, plates bonded on the skin and
    blades of BFSC plate elements

    Parameters
    ----------
    R, L : float
        Radius of the mid-surface of the skin and length.
    y : array
        Arc-length coordinates of the lines of nodes along `x`, increasing,
        from ``0``; for a closed cylinder the last one is `2 \pi R`, the
        first one again.
    closed : bool
        Closed cylinder or open curved panel.
    nx : int
        Number of nodes along `x`.
    skin : composites Laminate
        Laminate of the skin, referred to its mid-surface.
    nint : int
        Number of integration points in each direction.
    """

    def __init__(self, R, L, y, closed, nx, skin, nint=4):
        self.R, self.L, self.closed, self.nx = R, L, closed, nx
        y = np.asarray(y, dtype=float)
        if closed:
            assert abs(y[-1] - 2*pi*R) < 1e-6*R
            self.yn = y[:-1]
        else:
            self.yn = y
        self.ye = y
        self.nyn = nyn = self.yn.shape[0]
        self.ncol = y.shape[0] - 1
        self.xlin = np.linspace(0, L, nx)
        self.skin_nodes = np.arange(nx*nyn).reshape(nx, nyn)
        self.num_nodes = nx*nyn
        self.points, self.weights = get_points_weights(nint=nint)
        self.nint = nint
        self.layers = [(skin, 0., y[-1])]
        self.blades = []
        self._built = False

    def add_plate(self, laminate, y1, y2):
        """Plate bonded on the skin between ``y1`` and ``y2`` (node
        coordinates), its laminate referred to the mid-surface of the skin
        through the ``offset`` of the laminate"""
        self.layers.append((laminate, y1, y2))

    def add_blade(self, y, H, laminate, nh, side='internal'):
        """Blade at the node coordinate ``y``, of height ``H`` from the
        mid-surface of the skin to the tip, with ``nh`` elements along the
        height"""
        assert side in ('internal', 'external')
        j = int(np.argmin(np.abs(self.yn - y)))
        assert abs(self.yn[j] - y) < 1e-6*self.R, 'no node at the blade'
        self.blades.append(dict(j=j, H=H, laminate=laminate, nh=nh,
                                sign=1. if side == 'external' else -1.))

    def _column(self, j):
        jp = (j + 1) % self.nyn if self.closed else j + 1
        return j, jp

    def build(self):
        """Create the elements"""
        nx, R = self.nx, self.R
        lex = self.L/(nx - 1)
        self.lex = lex
        skin = self.skin_nodes
        self.shells = []
        ycen = 0.5*(self.ye[1:] + self.ye[:-1])
        tol = 1e-9*R
        for lam, y1, y2 in self.layers:
            for jc in range(self.ncol):
                if not (y1 - tol <= ycen[jc] <= y2 + tol):
                    continue
                j, jp = self._column(jc)
                ley = self.ye[jc + 1] - self.ye[jc]
                for i in range(nx - 1):
                    shell = BFSCCylinderSanders(self.nint)
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
                    shell.ley = ley
                    assign_constant_ABD(shell, lam)
                    self.shells.append(shell)
        n0 = self.num_nodes
        self.plates = []
        self.plate_A = []
        for b in self.blades:
            nh = b['nh']
            b['Y'] = b['sign']*np.arange(nh + 1)*b['H']/nh
            b['nodes'] = n0 + np.arange(nx*(nh + 1)).reshape(nx, nh + 1)
            n0 += nx*(nh + 1)
            for i in range(nx - 1):
                for k in range(nh):
                    # node 1 at the smallest X and Y
                    k1, k4 = (k, k + 1) if b['sign'] > 0 else (k + 1, k)
                    plate = BFSCPlate2D()
                    plate.n1 = b['nodes'][i, k1]
                    plate.n2 = b['nodes'][i + 1, k1]
                    plate.n3 = b['nodes'][i + 1, k4]
                    plate.n4 = b['nodes'][i, k4]
                    plate.c1 = DOF*plate.n1
                    plate.c2 = DOF*plate.n2
                    plate.c3 = DOF*plate.n3
                    plate.c4 = DOF*plate.n4
                    plate.lex = lex
                    plate.ley = b['H']/nh
                    plate.ABD = b['laminate'].ABD
                    self.plates.append(plate)
                    self.plate_A.append(b['laminate'].ABD[:3, :3])
        self.plate_A = np.array(self.plate_A)
        self.num_nodes_total = n0
        self.N = DOF*n0
        # connection of the blade roots to the skin
        self.ties = {}
        self.fixed = set()
        for b in self.blades:
            for i in range(nx):
                c = DOF*skin[i, b['j']]
                r = DOF*b['nodes'][i, 0]
                self.ties[r + U] = [(c + U, 1.)]
                self.ties[r + U_X] = [(c + U_X, 1.)]
                self.ties[r + V] = [(c + W, 1.)]
                self.ties[r + V_X] = [(c + W_X, 1.)]
                self.ties[r + W] = [(c + V, 1.)]
                self.ties[r + W_X] = [(c + V_X, 1.)]
                self.ties[r + W_Y] = [(c + W_Y, -1.), (c + V, 1/R)]
                self.ties[r + W_XY] = [(c + W_XY, -1.), (c + V_X, 1/R)]
        self._built = True

    # boundary conditions
    def fix(self, nodes, dofs):
        """Prescribe zero to the degrees of freedom ``dofs`` of ``nodes``"""
        for n in np.atleast_1d(nodes).ravel():
            for d in dofs:
                self.fixed.add(DOF*int(n) + d)

    def blade_end_nodes(self, i):
        """Nodes of the blades at the section ``i`` along `x`, without the
        roots, which follow the skin"""
        if not self.blades:
            return np.zeros(0, dtype=int)
        return np.concatenate([b['nodes'][i, 1:] for b in self.blades])

    def clamp_end(self, i, rotation_free=False):
        """End section ``i`` (``0`` or ``-1``) fixed: `u = v = w = 0` and
        `w_{,x} = 0` of the skin, unless ``rotation_free``, and the same for
        the blades, `U = V = W = 0` and `W_{,X} = 0`; the derivatives along
        the edge of the constrained displacements are constrained as well"""
        rot = [] if rotation_free else [W_X, W_XY]
        self.fix(self.skin_nodes[i, :], [U, U_Y, V, V_Y, W, W_Y] + rot)
        self.fix(self.blade_end_nodes(i), [U, U_Y, V, V_Y, W, W_Y] + rot)

    def rigid_end(self, i=-1, rotation_free=False):
        r"""End section ``i`` attached to a rigid ring: a uniform axial
        displacement and a rotation about the axis, with `w = 0` and
        `w_{,x} = 0` (unless ``rotation_free``) of the skin, and for the
        blades `U` uniform, `V = 0`, `W = (R + Y) \phi`, `W_{,Y} = \phi`
        and `W_{,X} = 0`, with `\phi` the rotation of the ring. Returns the
        master degrees of freedom of the axial displacement and of the
        circumferential displacement `R \phi` of the ring, those of `u` and
        `v` of the first node of the section of the skin"""
        R = self.R
        rot = [] if rotation_free else [W_X, W_XY]
        nodes = self.skin_nodes[i, :]
        mu = DOF*nodes[0] + U
        mv = DOF*nodes[0] + V
        self.fix(nodes, [U_Y, V_Y, W, W_Y] + rot)
        for n in nodes[1:]:
            self.ties[DOF*n + U] = [(mu, 1.)]
            self.ties[DOF*n + V] = [(mv, 1.)]
        for b in self.blades:
            for k in range(1, b['nh'] + 1):
                c = DOF*b['nodes'][i, k]
                self.ties[c + U] = [(mu, 1.)]
                self.ties[c + W] = [(mv, (R + b['Y'][k])/R)]
                self.ties[c + W_Y] = [(mv, 1/R)]
                self.fix(b['nodes'][i, k], [U_Y, V, V_Y] + rot)
        return mu, mv

    def support_side(self, j, clamped=False):
        """Unloaded straight edge ``j`` (``0`` or ``-1``) of an open panel:
        `w = 0` (and `w_{,y} = 0` when ``clamped``), `u` and `v` free"""
        dofs = [W, W_X] + ([W_Y, W_XY] if clamped else [])
        self.fix(self.skin_nodes[:, j], dofs)

    def finalize(self):
        """Constraint basis `\boldsymbol{Z}` from the ties and fixed
        degrees of freedom"""
        N = self.N
        dep = dict(self.ties)
        for d in self.fixed:
            dep[d] = []
        resolved = {}

        def expand(d, depth=0):
            assert depth < 10
            if d in resolved:
                return resolved[d]
            if d not in dep:
                return [(d, 1.)]
            out = {}
            for d2, coef in dep[d]:
                for d3, c3 in expand(d2, depth + 1):
                    out[d3] = out.get(d3, 0.) + coef*c3
            resolved[d] = list(out.items())
            return resolved[d]

        for d in list(dep.keys()):
            expand(d)
        independent = np.ones(N, dtype=bool)
        independent[list(dep.keys())] = False
        self.Nr = Nr = int(independent.sum())
        red = -np.ones(N, dtype=int)
        red[independent] = np.arange(Nr)
        self.red = red
        rows = list(np.flatnonzero(independent))
        cols = list(red[independent])
        vals = [1.]*Nr
        for d, terms in resolved.items():
            for d2, coef in terms:
                assert red[d2] >= 0
                rows.append(d)
                cols.append(red[d2])
                vals.append(coef)
        self.Z = coo_matrix((vals, (rows, cols)), shape=(N, Nr)).tocsr()

    # matrices
    def _assemble(self, skin_func, skin_size, plate_func, plate_size):
        ns, nt = len(self.shells), len(self.plates)
        r = np.zeros(skin_size*ns + plate_size*nt, dtype=sanders.INT)
        c = np.zeros_like(r)
        v = np.zeros(r.shape[0], dtype=float)
        k = 0
        for shell in self.shells:
            skin_func(shell, k, r, c, v)
            k += skin_size
        for plate in self.plates:
            plate_func(plate, k, r, c, v)
            k += plate_size
        return coo_matrix((v, (r, c)), shape=(self.N, self.N)).tocsr()

    def reduce(self, A):
        r"""`\boldsymbol{Z}^\top \boldsymbol{A} \boldsymbol{Z}`"""
        return (self.Z.T @ A @ self.Z).tocsr()

    def KC0(self):
        """Linear stiffness matrix, all the degrees of freedom"""
        p, w = self.points, self.weights

        def skin(shell, k, r, c, v):
            shell.init_k_KC0 = k
            sanders.update_KC0(shell, p, w, r, c, v)

        def plate(pl, k, r, c, v):
            pl.init_k_KC0 = k
            bfscplate2d.update_KC0(pl, p, w, r, c, v)

        return self._assemble(skin, sanders.KC0_SPARSE_SIZE, plate,
                              bfscplate2d.KC0_SPARSE_SIZE)

    def KG(self, u):
        """Geometric stiffness matrix of the displacements ``u`` of all the
        degrees of freedom, with the in-plane terms of the blades"""
        p, w = self.points, self.weights

        def skin(shell, k, r, c, v):
            shell.init_k_KG = k
            sanders.update_KG(u, shell, p, w, r, c, v)

        def plate(pl, k, r, c, v):
            pl.init_k_KG = k
            bfscplate2d.update_KG(u, pl, p, w, r, c, v)

        KG = self._assemble(skin, sanders.KG_SPARSE_SIZE, plate,
                            bfscplate2d.KG_SPARSE_SIZE)
        if self.plates:
            KG = KG + self.KG_inplane(self.blade_membrane_forces(u))
        return KG

    def _blade_operators(self):
        if hasattr(self, '_ops'):
            return self._ops
        p, w = self.points, self.weights
        ops = {}
        for e, pl in enumerate(self.plates):
            key = (round(pl.lex, 12), round(pl.ley, 12))
            if key in ops:
                continue
            Bm, Sv_x, Sv_y, wts = [], [], [], []
            for i, xi in enumerate(p):
                for j, eta in enumerate(p):
                    pl.update_Bm(xi, eta)
                    pl.update_Sv_x(xi, eta)
                    pl.update_Sv_y(xi, eta)
                    Bm.append(np.array(pl.Bm))
                    Sv_x.append(np.array(pl.Sv_x))
                    Sv_y.append(np.array(pl.Sv_y))
                    wts.append(w[i]*w[j]*pl.lex*pl.ley/4)
            ops[key] = tuple(np.array(a) for a in (Bm, Sv_x, Sv_y, wts))
        keys = [(round(pl.lex, 12), round(pl.ley, 12)) for pl in self.plates]
        dofs = np.array([[pl.c1 + d for d in range(DOF)]
                         + [pl.c2 + d for d in range(DOF)]
                         + [pl.c3 + d for d in range(DOF)]
                         + [pl.c4 + d for d in range(DOF)]
                         for pl in self.plates])
        self._ops = (ops, keys, dofs)
        return self._ops

    def blade_membrane_forces(self, u):
        """Membrane forces `N_{xx}, N_{yy}, N_{xy}` of the blade elements at
        their integration points, shape ``(num_elements, nint**2, 3)``"""
        ops, keys, dofs = self._blade_operators()
        N = np.zeros((len(self.plates), self.nint**2, 3))
        for key, (Bm, Sv_x, Sv_y, wts) in ops.items():
            sel = np.array([k == key for k in keys])
            eps = np.einsum('pij,ej->epi', Bm, u[dofs[sel]])
            N[sel] = np.einsum('epi,eki->epk', eps, self.plate_A[sel])
        return N

    def KG_inplane(self, N):
        r"""Geometric stiffness matrix of the radial displacement `V` of the
        blades for their membrane forces ``N``"""
        ops, keys, dofs = self._blade_operators()
        rows, cols, vals = [], [], []
        for key, (Bm, Sv_x, Sv_y, wts) in ops.items():
            sel = np.array([k == key for k in keys])
            Nw = N[sel]*wts[None, :, None]
            Ke = (np.einsum('ep,pi,pj->eij', Nw[:, :, 0], Sv_x, Sv_x)
                  + np.einsum('ep,pi,pj->eij', Nw[:, :, 1], Sv_y, Sv_y)
                  + np.einsum('ep,pi,pj->eij', Nw[:, :, 2], Sv_x, Sv_y)
                  + np.einsum('ep,pi,pj->eij', Nw[:, :, 2], Sv_y, Sv_x))
            d = dofs[sel]
            rows.append(np.repeat(d, d.shape[1], axis=1).ravel())
            cols.append(np.tile(d, (1, d.shape[1])).ravel())
            vals.append(Ke.ravel())
        return coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.N, self.N)).tocsr()

    # analyses
    def static(self, loads):
        """Linear static solution for the forces ``loads``, a dictionary
        ``{dof: force}`` of independent degrees of freedom; returns the
        displacements of all the degrees of freedom and the reduced
        stiffness matrix"""
        Kr = self.reduce(self.KC0())
        f = np.zeros(self.N)
        for d, val in loads.items():
            assert self.red[d] >= 0
            f[d] = val
        ur = spsolve(Kr.tocsc(), self.Z.T @ f)
        return self.Z @ ur, Kr

    def buckling(self, loads, num_eigvalues=10, reverse=False):
        """Linear buckling analysis about the linear static state of
        ``loads``: load multipliers and modes (all the degrees of freedom),
        with the load reversed if ``reverse``; also returns the static
        displacements"""
        u, Kr = self.static(loads)
        KGr = self.reduce(self.KG(u))
        if reverse:
            KGr = -KGr
        lam, vecs = lb(Kr, KGr, num_eigvalues=num_eigvalues, silent=True)
        modes = self.Z @ vecs
        return lam, modes, u

    def skin_field(self, u, dof=W):
        """Field of one degree of freedom of the skin, shape ``(nx, nyn)``"""
        return u[DOF*self.skin_nodes + dof]


def plot_skin_w(ax, model, mode, title='', blade_color='k'):
    """Normal displacement `w` of the skin over the developed surface, with
    the blades as lines"""
    w = model.skin_field(mode, W)
    y = model.yn
    if model.closed:
        y = np.append(y, model.ye[-1])
        w = np.hstack((w, w[:, :1]))
    lim = np.abs(w).max()
    yy, xx = np.meshgrid(y, model.xlin)
    ax.contourf(yy, xx, w, levels=np.linspace(-lim, lim, 31), cmap='RdBu_r')
    for b in model.blades:
        ax.axvline(model.yn[b['j']], color=blade_color, lw=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('$y = R\\theta$ (mm)')
    ax.set_aspect('equal')
