r"""Periodic sector of a cylinder with Z-section stringers made of three plates

Model of ``tests/test_peterson1959_z_stiffened.py``. A circular cylinder
with ``nstringers`` identical Z-section stringers, equally spaced, is
modelled by one stringer pitch, the sector `0 \le \theta \le \phi`, `\phi =
2 \pi/n_s`, with cyclic symmetry, one harmonic `n` at a time. The skin is
meshed with :class:`bfsccylinder.sanders.BFSCCylinderSanders`; the Z is
assembled from three plates of the same thickness `t_z`, with the
dimensions of their middle lines (:func:`z_midlines`):

- the attached flange, riveted to the skin, from the web to the free edge
  of the flange, of width `b_f`: the skin elements under it have the
  stiffnesses of the two-layer laminate skin + flange of
  :class:`TwoLayerSkin` (a continuous bond), referred to the middle
  surface of the skin;
- the web, a :class:`bfscplate2d.BFSCPlate2D` plate in the plane `(x, r)`
  at `\theta_w = \phi/2`, from the middle surface of the skin to the middle
  surface of the free flange, of height `H_w`;
- the free flange, a plate in the plane tangent to the circumference at the
  tip of the web, of width `b_l` from the middle surface of the web to its
  free edge, on the side of the web opposite to the attached flange.

``flange_side=-1`` puts the attached flange at `\theta < \theta_w` and the
free flange at `\theta > \theta_w`, ``flange_side=1`` the reverse. The
model is written for any pitch, side, dimensions and orientation of the Z,
and for any constant membrane pre-buckling state of the skin, see
:meth:`ZCylinderSector.KG_membrane`.

Notes
-----
**Ties.** The web has the local coordinates `X = x` and `Y = r - R`; its
displacements `U`, `V`, `W` are along `x`, along the outward radius and
along the circumference. Its nodes along the root are tied to the skin
nodes of the line `\theta_w`:

.. math::

    U = u, \quad V = w, \quad W = v, \quad W_{,Y} = -w_{,y} + \frac{v}{R}

with the derivatives along `x`, the rotation about `x` of the Sanders-type
kinematics. The free flange has the local coordinates `X = x` and `Y = s`,
the arc coordinate along `\theta`, and the displacements `U`, `V`, `W`
along `x`, along the circumference and along the outward radius; its nodes
along the junction are tied to the tip of the web by a rigid corner of 90
degrees,

.. math::

    U_l = U_w, \quad V_l = W_w, \quad W_l = V_w, \quad W_{l,Y} = -W_{w,Y}

since a rotation `\psi` about `x` gives `\partial u_r/\partial s = \psi`
along the free flange and `\partial u_\theta/\partial r = -\psi` along the
web.

**Overlap.** The web starts at the middle surface of the skin, so that it
overlaps the skin and the attached flange over `h/2 + t_z/2`; the
section as modelled is given by :func:`section_properties`.

**Cyclic symmetry.** The
structure, with its Z sections, is invariant under a rotation by `\phi`
about its axis (not under a reflection, which turns a Z into an S). The
nodes of the edge `\theta = \phi` are eliminated with
`\boldsymbol{u}(\theta = \phi) = e^{i n \phi} \boldsymbol{u}(\theta = 0)`,
which gives one Hermitian eigenvalue problem per harmonic `n = 0, 1,
\ldots, n_s/2`; the harmonic `n` contains the circumferential waves `n +
j n_s` coupled by the stringers, among them the local modes of the skin
between the stringers for `n` near `n_s/2`.

**Geometric stiffness of the plates.** The geometric stiffness of the BFSC
plate element has the terms of its out-of-plane displacement `W` only; the
terms `N_{xx} V_{,X}^2 + 2 N_{xy} V_{,X} V_{,Y} + N_{yy} V_{,Y}^2` of the
in-plane displacement `V` normal to `x` (radial in the web, circumferential
in the free flange) are added by :meth:`ZCylinderSector.KG_inplane`.

"""
import numpy as np
from numpy import pi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, eigs, splu

from composites import isotropic_plate
import bfsccylinder.sanders as sanders
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD
import bfscplate2d
from bfscplate2d import BFSCPlate2D

DOF = 10
# positions of the degrees of freedom in a node
U, U_X, U_Y, V, V_X, V_Y, W, W_X, W_Y, W_XY = range(10)

#: constrained degrees of freedom of the end nodes, skin and plates
END_DOFS = {
    'S': [V, V_Y, W, W_Y],
    'C': [U, U_Y, V, V_Y, W, W_Y, W_X, W_XY],
    'F': [],
}


def z_midlines(tz, flange, depth, free_flange, h=0., bend_radius=0.):
    r"""Middle-line dimensions of the three plates of a Z section

    Parameters
    ----------
    tz : float
        Thickness of the sheet.
    flange : float
        Width of the attached flange, from its free edge to the outer face
        of the web.
    depth : float
        Depth of the section, from the face of the attached flange on the
        skin to the outer face of the free flange.
    free_flange : float
        Width of the free flange, from the outer face of the web to its free
        edge.
    h : float
        Skin thickness; the web of the model starts at the middle surface
        of the skin.
    bend_radius : float
        Inner radius of the two bends; each bend of centre-line radius
        `r_c` = ``bend_radius`` `+ t_z/2` shortens the adjacent flange by
        `(2 - \pi/2) r_c`, so that the plates have the length of the centre
        line of the bent sheet; ``0`` gives square corners.

    Returns
    -------
    dict
        ``tz``, ``bf`` (width of the attached flange, from the middle
        surface of the web), ``Hw`` (height of the web, from the middle
        surface of the skin to the middle surface of the free flange) and
        ``bl`` (width of the free flange, from the middle surface of the
        web).

    """
    rc = bend_radius + tz/2 if bend_radius > 0 else 0.
    cut = (2 - pi/2)*rc
    return dict(tz=tz, bf=flange - tz/2 - cut, Hw=h/2 + depth - tz/2,
                bl=free_flange - tz/2 - cut)


def section_properties(sec, h, E, nu):
    r"""Properties of the stringer as modelled, per stringer

    The rectangles of the attached flange (centred at `h/2 + t_z/2` from
    the middle surface of the skin), of the web (from the middle surface of
    the skin, overlapping skin and flange) and of the free flange.
    Distances from the middle surface of the skin, positive away from the
    skin on the side of the stringer.

    Returns
    -------
    dict
        ``A`` area, ``zc`` centroid, ``Io`` moment of inertia about the
        middle surface of the skin (bending in the radial plane), ``J``
        torsion constant `\sum b t_z^3/3`, ``EA``, ``EIo`` and ``GJ``.

    """
    t = sec['tz']
    zf = h/2 + t/2
    Hw = sec['Hw']
    rects = [(sec['bf']*t, zf, sec['bf']*t**3/12),
             (Hw*t, Hw/2, t*Hw**3/12),
             (sec['bl']*t, Hw, sec['bl']*t**3/12)]
    A = sum(r[0] for r in rects)
    zc = sum(r[0]*r[1] for r in rects)/A
    Io = sum(r[2] + r[0]*r[1]**2 for r in rects)
    J = (sec['bf'] + Hw + sec['bl'])*t**3/3
    G = E/(2*(1 + nu))
    return dict(A=A, zc=zc, Io=Io, J=J, EA=E*A, EIo=E*Io, GJ=G*J)


class TwoLayerSkin:
    r"""Stiffnesses of the skin with a second isotropic layer, of the same
    material, on one side, referred to the middle surface of the skin

    Parameters
    ----------
    h, t : float
        Thicknesses of the skin and of the added layer.
    E, nu : float
        Young's modulus and Poisson's ratio of both layers.
    side : str
        ``'internal'``, layer at `-h/2 - t \le z \le -h/2`, or
        ``'external'``, at `h/2 \le z \le h/2 + t`, with `z` along the
        outward normal.

    Attributes
    ----------
    A11, ..., D66 : float
        The entries of `\boldsymbol{A}`, `\boldsymbol{B}` and
        `\boldsymbol{D}`.

    Notes
    -----
    `(A, B, D)_{ij} = Q_{ij} \int (1, z, z^2) dz` over both layers, with
    the plane-stress stiffnesses `Q_{11} = Q_{22} = E/(1 - \nu^2)`,
    `Q_{12} = \nu Q_{11}`, `Q_{66} = E/(2 (1 + \nu))`.

    """

    def __init__(self, h, t, E, nu, side='internal'):
        if side == 'internal':
            z0, z3 = -h/2 - t, h/2
        else:
            z0, z3 = -h/2, h/2 + t
        i1 = z3 - z0
        i2 = (z3**2 - z0**2)/2
        i3 = (z3**3 - z0**3)/3
        Q11 = E/(1 - nu**2)
        Q = {'11': Q11, '12': nu*Q11, '22': Q11, '66': E/(2*(1 + nu)),
             '16': 0., '26': 0.}
        for k, q in Q.items():
            setattr(self, 'A' + k, q*i1)
            setattr(self, 'B' + k, q*i2)
            setattr(self, 'D' + k, q*i3)


class ZCylinderSector:
    r"""Sector of one stringer pitch of a cylinder with Z stringers

    Parameters
    ----------
    R, L, h : float
        Radius of the middle surface, length and thickness of the skin.
    E, nu : float
        Young's modulus and Poisson's ratio, skin and stringer.
    nstringers : int
        Number of stringers; the sector angle is `\phi = 2 \pi/n_s`.
    nx : int
        Number of nodes along `x`.
    sec : dict or None
        Middle lines of :func:`z_midlines`; ``None`` for an unstiffened
        sector.
    side : str
        ``'internal'`` or ``'external'``.
    flange_side : int
        ``-1``: the attached flange at `\theta < \phi/2`; ``1``: at
        `\theta > \phi/2`.
    nbay : int
        Number of skin elements across the free skin of the pitch, divided
        between its two parts in proportion to their widths (at least one
        each); without stringer, across the sector.
    nflange, nweb, nlip : int
        Number of elements across the attached flange, along the web and
        across the free flange.
    nint : int
        Number of Gauss-Legendre points per direction.

    Attributes
    ----------
    theta : array
        Angles of the columns of skin nodes, from 0 to `\phi`.
    j_web : int
        Column of the web.
    xlin : array
        Axial positions of the nodes.
    N, Nr : int
        Number of degrees of freedom of the sector, and of the reduced
        ones after the last call of :meth:`Z`.

    """

    def __init__(self, R, L, h, E, nu, nstringers, nx, sec=None,
                 side='internal', flange_side=-1, nbay=6, nflange=3, nweb=4,
                 nlip=2, nint=4):
        assert side in ('internal', 'external')
        assert flange_side in (-1, 1)
        self.R, self.L, self.h = R, L, h
        self.E, self.nu = E, nu
        self.nstringers = nstringers
        self.phi = phi = 2*pi/nstringers
        self.nx = nx
        self.sec, self.side = sec, side
        self.stiffened = sec is not None
        self.points, self.weights = get_points_weights(nint=nint)

        if self.stiffened:
            bf = sec['bf']/R
            near = phi/2 - bf
            assert near > 0, 'attached flange wider than half the pitch'
            far = phi/2
            n1 = max(1, int(round(nbay*near/(near + far))))
            n2 = max(1, nbay - n1)
            parts = [np.full(n1, near/n1), np.full(nflange, bf/nflange),
                     np.full(n2, far/n2)]
            if flange_side == 1:
                parts = parts[::-1]
            dtheta = np.concatenate(parts)
            if flange_side == -1:
                self.j_web = n1 + nflange
                self.j_flange = (n1, n1 + nflange)
            else:
                self.j_web = n2
                self.j_flange = (n2, n2 + nflange)
        else:
            dtheta = np.full(nbay, phi/nbay)
        self.theta = np.concatenate([[0.], np.cumsum(dtheta)])
        self.theta[-1] = phi
        ncol = self.theta.size
        skin = np.arange(nx*ncol).reshape(nx, ncol)
        self.skin_nodes = skin
        num_nodes = skin.size
        self.xlin = np.linspace(0, L, nx)
        lex = L/(nx - 1)

        if self.stiffened:
            sign = 1. if side == 'external' else -1.
            self.web_nodes = num_nodes + np.arange(nx*(nweb + 1)).reshape(
                nx, nweb + 1)
            num_nodes += self.web_nodes.size
            # Y = s from the junction; the free flange points to -flange_side
            if flange_side == -1:
                self.k_junction = 0
            else:
                self.k_junction = nlip
            self.lip_nodes = num_nodes + np.arange(nx*(nlip + 1)).reshape(
                nx, nlip + 1)
            num_nodes += self.lip_nodes.size
        self.N = DOF*num_nodes

        self.skin_prop = isotropic_plate(thickness=h, E=E, nu=nu)
        if self.stiffened:
            self.flange_prop = TwoLayerSkin(h, sec['tz'], E, nu, side=side)
        self.skin_elements = []
        self.skin_Nfactor = []
        for i in range(nx - 1):
            for j in range(ncol - 1):
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
                shell.ley = R*dtheta[j]
                if (self.stiffened
                        and self.j_flange[0] <= j < self.j_flange[1]):
                    shell.h = h + sec['tz']
                    assign_constant_ABD(shell, self.flange_prop)
                    self.skin_Nfactor.append((h + sec['tz'])/h)
                else:
                    shell.h = h
                    assign_constant_ABD(shell, self.skin_prop)
                    self.skin_Nfactor.append(1.)
                self.skin_elements.append(shell)

        self.str_elements = []
        if self.stiffened:
            self.str_prop = isotropic_plate(thickness=sec['tz'], E=E, nu=nu)
            Hw, bl = sec['Hw'], sec['bl']
            for i in range(nx - 1):
                for k in range(nweb):
                    # node 1 at the smallest X and Y
                    k1, k4 = (k, k + 1) if sign > 0 else (k + 1, k)
                    self._plate(self.web_nodes[i, k1],
                                self.web_nodes[i + 1, k1],
                                self.web_nodes[i + 1, k4],
                                self.web_nodes[i, k4], lex, Hw/nweb)
                for k in range(nlip):
                    self._plate(self.lip_nodes[i, k],
                                self.lip_nodes[i + 1, k],
                                self.lip_nodes[i + 1, k + 1],
                                self.lip_nodes[i, k + 1], lex, bl/nlip)

    def _plate(self, n1, n2, n3, n4, lex, ley):
        plate = BFSCPlate2D()
        plate.n1, plate.n2, plate.n3, plate.n4 = n1, n2, n3, n4
        plate.c1 = DOF*n1
        plate.c2 = DOF*n2
        plate.c3 = DOF*n3
        plate.c4 = DOF*n4
        plate.lex = lex
        plate.ley = ley
        plate.h = self.sec['tz']
        plate.rho = 0.
        plate.ABD = self.str_prop.ABD
        self.str_elements.append(plate)

    # ------------------------------------------------------------------
    def Z(self, n, free_axial=False):
        r"""Constraint basis `\boldsymbol{Z}_n` of the harmonic ``n``

        Real for ``n = 0`` and ``n = n_s/2``, complex otherwise. With
        ``free_axial=True`` and ``n = 0`` the mean of `u` along the skin
        edge `x = 0` is tied to zero (axial rigid-body translation removed).
        Sets the attributes ``red`` (index of each degree of freedom among
        the reduced ones, ``-1`` for the eliminated ones) and ``Nr``.
        """
        skin = self.skin_nodes
        ncol = skin.shape[1]
        if n == 0:
            phase = 1.
        elif 2*n == self.nstringers:
            phase = -1.
        else:
            phase = np.exp(1j*n*self.phi)
        dependent = {}
        for i in range(self.nx):
            cl = DOF*skin[i, 0]
            cr = DOF*skin[i, ncol - 1]
            for d in range(DOF):
                dependent[cr + d] = [(cl + d, phase)]
        if self.stiffened:
            Rinv = 1/self.R
            for i in range(self.nx):
                # web root to the skin
                c = DOF*skin[i, self.j_web]
                b = DOF*self.web_nodes[i, 0]
                dependent[b + U] = [(c + U, 1.)]
                dependent[b + U_X] = [(c + U_X, 1.)]
                dependent[b + V] = [(c + W, 1.)]
                dependent[b + V_X] = [(c + W_X, 1.)]
                dependent[b + W] = [(c + V, 1.)]
                dependent[b + W_X] = [(c + V_X, 1.)]
                dependent[b + W_Y] = [(c + W_Y, -1.), (c + V, Rinv)]
                dependent[b + W_XY] = [(c + W_XY, -1.), (c + V_X, Rinv)]
                # free flange to the tip of the web
                c = DOF*self.web_nodes[i, -1]
                b = DOF*self.lip_nodes[i, self.k_junction]
                dependent[b + U] = [(c + U, 1.)]
                dependent[b + U_X] = [(c + U_X, 1.)]
                dependent[b + V] = [(c + W, 1.)]
                dependent[b + V_X] = [(c + W_X, 1.)]
                dependent[b + W] = [(c + V, 1.)]
                dependent[b + W_X] = [(c + V_X, 1.)]
                dependent[b + W_Y] = [(c + W_Y, -1.)]
                dependent[b + W_XY] = [(c + W_XY, -1.)]
        if free_axial and n == 0:
            c0 = DOF*skin[0, 0]
            dependent[c0 + U] = [(DOF*skin[0, j] + U, -1.)
                                 for j in range(1, ncol - 1)]
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
        supported, `v = w = 0` on the skin and the in-section displacements
        `V = W = 0` on the end edges of the plates; ``'C'``, clamped,
        `u = v = w = w_{,x} = 0` and the end edges of the plates clamped;
        ``'F'``, free. The derivatives along an edge of its constrained
        displacements are constrained as well. Uses the reduced numbering
        of the last call of :meth:`Z`.
        """
        bk = np.zeros(self.Nr, dtype=bool)
        for end, i in zip(ends, [0, self.nx - 1]):
            nodes = list(self.skin_nodes[i, :])
            if self.stiffened:
                nodes += list(self.web_nodes[i, :])
                nodes += list(self.lip_nodes[i, :])
            for nd in nodes:
                for d in END_DOFS[end]:
                    rd = self.red[DOF*nd + d]
                    if rd >= 0:
                        bk[rd] = True
        return bk

    # ------------------------------------------------------------------
    def _assemble(self, skin_func, skin_size, str_func, str_size):
        nse = len(self.skin_elements)
        nst = len(self.str_elements)
        r = np.zeros(skin_size*nse + str_size*nst, dtype=sanders.INT)
        c = np.zeros_like(r)
        v = np.zeros(r.shape[0], dtype=float)
        k = 0
        for e, shell in enumerate(self.skin_elements):
            skin_func(e, shell, k, r, c, v)
            k += skin_size
        for plate in self.str_elements:
            str_func(plate, k, r, c, v)
            k += str_size
        return coo_matrix((v, (r, c)), shape=(self.N, self.N)).tocsr()

    def KC0(self):
        """Linear stiffness matrix of all the degrees of freedom of the
        sector, computed once"""
        if not hasattr(self, '_KC0'):
            p, w = self.points, self.weights

            def skin(e, shell, k, r, c, v):
                shell.init_k_KC0 = k
                sanders.update_KC0(shell, p, w, r, c, v)

            def stringer(plate, k, r, c, v):
                plate.init_k_KC0 = k
                bfscplate2d.update_KC0(plate, p, w, r, c, v)

            self._KC0 = self._assemble(skin, sanders.KC0_SPARSE_SIZE,
                                       stringer, bfscplate2d.KC0_SPARSE_SIZE)
        return self._KC0

    def KG_membrane(self, eps=0., Nxy=0., Nyy=0.):
        r"""Geometric stiffness matrix of a constant membrane state

        The uniform axial strain ``eps`` gives `N_{xx} = E h` ``eps`` in
        the skin, `E (h + t_z)` ``eps`` in the skin under the attached
        flange and `E t_z` ``eps`` in the web and the free flange (no
        Poisson expansion). ``Nxy`` and ``Nyy`` are constant shear and hoop
        forces per unit length of the skin (also under the flange); the
        plates carry none. The terms of the in-plane displacement of the
        plates are added with :meth:`KG_inplane`.
        """
        p, w = self.points, self.weights
        Nskin = self.E*self.h*eps
        Nstr = self.E*self.sec['tz']*eps if self.stiffened else 0.
        fac = self.skin_Nfactor

        def skin(e, shell, k, r, c, v):
            shell.init_k_KG = k
            sanders.update_KG_constant_stress(shell, p, w, r, c, v,
                                              Nskin*fac[e], Nyy, Nxy)

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

    def KG_inplane(self, N):
        r"""Geometric stiffness of the in-plane displacement `V` of the
        plates normal to `x`, for the membrane forces ``N``, shape
        ``(num_plate_elements, nint**2, 3)`` (`N_{xx}, N_{yy}, N_{xy}`):
        `\int (N_{xx} V_{,X}^2 + 2 N_{xy} V_{,X} V_{,Y} + N_{yy}
        V_{,Y}^2) dX dY`"""
        p, w = self.points, self.weights
        ops = {}
        rows, cols, vals = [], [], []
        for e, plate in enumerate(self.str_elements):
            key = (plate.lex, plate.ley)
            if key not in ops:
                Sx, Sy, wts = [], [], []
                for i, xi in enumerate(p):
                    for j, eta in enumerate(p):
                        plate.update_Sv_x(xi, eta)
                        plate.update_Sv_y(xi, eta)
                        Sx.append(np.array(plate.Sv_x))
                        Sy.append(np.array(plate.Sv_y))
                        wts.append(w[i]*w[j]*plate.lex*plate.ley/4)
                ops[key] = (np.array(Sx), np.array(Sy), np.array(wts))
            Sx, Sy, wts = ops[key]
            Nw = N[e]*wts[:, None]
            Ke = (np.einsum('p,pi,pj->ij', Nw[:, 0], Sx, Sx)
                  + np.einsum('p,pi,pj->ij', Nw[:, 1], Sy, Sy)
                  + np.einsum('p,pi,pj->ij', Nw[:, 2], Sx, Sy)
                  + np.einsum('p,pi,pj->ij', Nw[:, 2], Sy, Sx))
            dofs = np.concatenate([plate.c1 + np.arange(DOF),
                                   plate.c2 + np.arange(DOF),
                                   plate.c3 + np.arange(DOF),
                                   plate.c4 + np.arange(DOF)])
            rows.append(np.repeat(dofs, dofs.size))
            cols.append(np.tile(dofs, dofs.size))
            vals.append(Ke.ravel())
        return coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                           np.concatenate(cols))),
                          shape=(self.N, self.N)).tocsr()

    def area(self):
        """Cross-sectional area of the whole cylinder as modelled"""
        A = 2*pi*self.R*self.h
        if self.stiffened:
            s = self.sec
            A += self.nstringers*s['tz']*(s['bf'] + s['Hw'] + s['bl'])
        return A

    # ------------------------------------------------------------------
    def skin_field(self, u, dof=W):
        """Field of one degree of freedom of the skin of the sector, shape
        ``(nx, ncol)``"""
        return u[DOF*self.skin_nodes + dof]

    def full_circumference(self, field, n):
        r"""Real field over the whole circumference from the field of the
        sector ``field`` of the harmonic ``n``: returns `\theta` and the
        field, shape ``(nx, n_s (ncol - 1) + 1)``"""
        parts = [field[:, :-1]*np.exp(1j*n*k*self.phi)
                 for k in range(self.nstringers)]
        parts.append(field[:, :1]*np.exp(1j*n*2*pi))
        theta = np.concatenate([self.theta[:-1] + k*self.phi
                                for k in range(self.nstringers)] + [[2*pi]])
        return theta, np.real(np.hstack(parts))


def lowest_eigenpairs(A, B, k, sigma=0.):
    r"""The ``k`` lowest eigenvalues `\lambda > \sigma` of `\boldsymbol{A}
    \boldsymbol{x} = \lambda \boldsymbol{B} \boldsymbol{x}`, Hermitian
    matrices, and their eigenvectors, by shift-and-invert ARPACK
    (:func:`scipy.sparse.linalg.eigs`)"""
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


def buckling_membrane(model, ends, n, k=4):
    r"""Linear buckling under axial compression of the harmonic ``n``

    The pre-buckling state is the uniform axial strain of
    :meth:`ZCylinderSector.KG_membrane`; ``ends`` see
    :meth:`ZCylinderSector.constrained` (``'SS'``: the axial rigid-body
    translation of ``n = 0`` removed). Returns the critical average axial
    forces per unit circumference `\bar{N} = P/(2 \pi R)`, in increasing
    order, and the modes for all the degrees of freedom of the sector.
    """
    eps = -1e-3
    if not hasattr(model, '_KG_axial'):
        model._KG_axial = model.KG_membrane(eps)
    Zn = model.Z(n, free_axial=(ends == 'SS'))
    K = (Zn.conj().T @ model.KC0() @ Zn).tocsr()
    KG = (Zn.conj().T @ model._KG_axial @ Zn).tocsr()
    bu = ~model.constrained(ends)
    lam, X = lowest_eigenpairs(K[bu][:, bu], -KG[bu][:, bu], k)
    ur = np.zeros((model.Nr, X.shape[1]), dtype=X.dtype)
    ur[bu] = X
    P = model.E*abs(eps)*model.area()
    return lam*P/(2*pi*model.R), Zn @ ur


def axial_half_waves(model, mode, tol=0.02):
    """Number of sign changes along `x`, plus one, of the normal
    displacement `w` of the skin on the line of the sector where `w` is
    largest, values below ``tol`` times the largest one ignored"""
    w = model.skin_field(mode, W)
    i = np.argmax(np.abs(w))
    w = w*np.conj(w.ravel()[i])/abs(w.ravel()[i])
    j = np.argmax(np.sum(np.abs(w)**2, axis=0))
    f = np.real(w[:, j])
    f = f[np.abs(f) > tol*np.abs(f).max()]
    return int(np.sum(np.sign(f[1:]) != np.sign(f[:-1]))) + 1


def scan(model, ends, ns, k=1):
    r"""Lowest `\bar{N}` of each harmonic of ``ns``; returns a dict
    ``{n: (N, m, mode)}`` and the critical harmonic"""
    out = {}
    for n in ns:
        N, X = buckling_membrane(model, ends, n, k=k)
        out[n] = (N[0], axial_half_waves(model, X[:, 0]), X[:, 0])
    return out, min(out, key=lambda n: out[n][0])
