r"""Torsion cylinders of Moore and Wescoat (1944): data and sector models

Data of the 24S-T cylinders of Moore and Wescoat (1944), NACA ARR 4E31,
with intermediate ring stiffeners only (specimens 16, 17, 18, 13 and 13A,
Table I) and with rings and outside channel stringers (specimens 19, 14,
20, 15 and 21, Table II), and their models for the linear buckling under a
torque.

The model, :class:`MooreSector`, is the periodic sector of
:class:`channel_cylinder_sector.ChannelCylinderSector` (skin of BFSC
cylinder elements, channel stringer of BFSC plate elements on the outside,
base as a two-layer laminate skin + base), with three changes:

- the nodes along `x` are placed at the edges and at the middle of every
  ring band, the lengths `lex` of the skin and plate elements are set row by
  row before any matrix is computed;
- a ring, a solid bar of width `b_r` and depth `d_r` inside the skin, is a
  band of skin elements of axial width `b_r`, centred at the ring station,
  whose laminate is the skin plus a layer of thickness `d_r` on its inner
  face (and the base of the channel on its outer face where a stringer
  crosses), stiffnesses referred to the middle surface of the skin
  (:func:`layered_ABD`); the ring layer has `\nu = 0` (see
  :func:`layered_ABD`);
- the boundary conditions of the bulkheads, hinged or clamped, and rigid
  ring supports (:meth:`MooreSector.constrained_bc`), and the geometric
  stiffness matrix of a constant membrane shear force `N_{xy}` in the skin
  elements (:meth:`MooreSector.KG_shear`); rings and stringers carry no
  membrane force under a torque.

For the cylinders with rings only, the sector is a thin slice of the
axisymmetric cylinder, ``nstringers`` = number of slices around the
circumference, and its Bloch index ``j`` is the circumferential wave
number `n` of the mode.

"""
import numpy as np
from numpy import pi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, eigs, splu

from bfsccylinder.utils import assign_constant_ABD
from channel_cylinder_sector import (ChannelCylinderSector, DOF, U, U_X,
                                     U_Y, V, V_X, V_Y, W, W_X, W_Y, W_XY)

# --------------------------------------------------------------------------
# data of Moore and Wescoat (1944)
# --------------------------------------------------------------------------
#: Young's modulus (psi) and Poisson's ratio of 24S-T, p. 7
E = 10.5e6
NU = 1/3
#: mean diameter and radius (in), Tables I and II
D = 30.08
R = D/2
#: length between the centres of the end bulkheads (in), Figs. 5 and 11
L = 35.625
#: width and depth (in) of the solid ring bars, p. 3
RING_B = 0.5
RING_D = 0.5
#: channel stringer, Fig. 30 and Fig. 1: thickness, outer width of the
#: base, overall height, overall width, A, I_XX, I_YY
CH_T = 0.032
CH_BASE = 0.5
CH_HEIGHT = 0.64
CH_WIDTH = 1.12
CH_A = 0.072
CH_IXX = 0.00375
CH_IYY = 0.0063

#: lengths of the bays between the bulkhead centres and the ring centres
#: (in), Fig. 5 (ring-only) and Fig. 11 (stiffened)
_B18 = [4 + 5/16, 13.5, 13.5, 4 + 5/16]
_B13 = [8 + 13/16, 9., 9., 8 + 13/16]
_B13A = [4 + 5/16, 4.5, 9., 9., 4.5, 4 + 5/16]
SPECIMENS = {
    '16': dict(t=0.020, ns=0, bays=[L]),
    '17': dict(t=0.020, ns=0, bays=[4 + 5/16, 27., 4 + 5/16]),
    '18': dict(t=0.020, ns=0, bays=_B18),
    '13': dict(t=0.020, ns=0, bays=_B13),
    '13A': dict(t=0.020, ns=0, bays=_B13A),
    '19': dict(t=0.020, ns=8, bays=_B18),
    '14': dict(t=0.020, ns=8, bays=_B13A),
    '20': dict(t=0.020, ns=16, bays=_B18),
    '15': dict(t=0.0195, ns=16, bays=_B13),
    '21': dict(t=0.0205, ns=16,
               bays=[2 + 1/16, 6.75, 9., 9., 6.75, 2 + 1/16]),
}

#: Table I, p. 25: length of the buckled section (in), average observed
#: and theoretical (Donnell, hinged edges) shear-buckling stresses (psi)
TABLE_I = {
    '16': (35.62, 1110, 1325),
    '17': (27.00, 1280, 1520),
    '18': (13.50, 1710, 2210),
    '13': (8.81, 2310, 2870),
    '13A': (9.00, 2570, 2770),
}

#: Table II, p. 26, the lowest panel of each specimen: panel length (in),
#: mean observed shear-buckling stress and the computed ones, ANC-5, Kromm
#: (ref. 3), Ebner (ref. 4) and Donnell (ref. 6, unstiffened cylinder of the
#: bay length) (psi); None where the table gives none
TABLE_II = {
    '19': (13.50, 2110, 1620, 1810, 2320, 2210),
    '14': (9.00, 2950, 1750, 2700, 2960, 2770),
    '20': (13.50, 2870, 2070, 2600, 2570, 2210),
    '15': (8.81, 3370, 2110, None, None, 2680),
    '21': (9.00, 4120, 2240, None, None, 2910),
}


def donnell_tau(t, Lb, E=E, nu=NU, D=D):
    r"""Shear-buckling stress (psi) of a cylinder with hinged edges, the
    formula of Donnell (1933) used by Moore and Wescoat (1944), appendix,
    p. 23, for `L^2 t/D^3 < 5.2`:

    .. math::

        \tau_0 = \frac{E t^2}{(1 - \nu^2) L^2} \left[2.8 + \sqrt{2.6 + 1.4
        \left(\frac{\sqrt{1 - \nu^2} L^2}{t D}\right)^{3/2}}\right]
    """
    z = np.sqrt(1 - nu**2)*Lb**2/(t*D)
    return E*t**2/((1 - nu**2)*Lb**2)*(2.8 + np.sqrt(2.6 + 1.4*z**1.5))


def torque_ftlb(tau, t, D=D):
    """Torque (ft-lb) of the shear stress ``tau`` (psi) in a skin of
    thickness ``t`` (in), Moore and Wescoat (1944), `\\tau = 24 T/(\\pi D^2
    t)`"""
    return tau*pi*D**2*t/24


def ring_stations(bays):
    """Axial positions of the rings (in) from the bay lengths"""
    return np.cumsum(bays)[:-1]


def channel_section(bl, t=CH_T, base=CH_BASE, height=CH_HEIGHT):
    r"""Area, height of the centroid above the base outer face, `I_{XX}`
    (about the axis parallel to the base) and `I_{YY}` (about the axis of
    symmetry) of the idealised channel: thin rectangles along the middle
    lines of the base (width ``base - t`` between the middle surfaces of
    the webs), of the webs (``height - t`` from the middle line of the base
    to that of the lips) and of two flat lips of width ``bl`` from the
    middle surface of the web, square corners."""
    b = base - t
    H = height - t
    segs = [(-b/2, t/2, b/2, t/2), (-b/2, t/2, -b/2, t/2 + H),
            (b/2, t/2, b/2, t/2 + H), (-b/2, t/2 + H, -b/2 - bl, t/2 + H),
            (b/2, t/2 + H, b/2 + bl, t/2 + H)]
    A = Sz = Izz = Iyy = 0.
    for y0, z0, y1, z1 in segs:
        ln = np.hypot(y1 - y0, z1 - z0)
        a = ln*t
        yc, zc = (y0 + y1)/2, (z0 + z1)/2
        if abs(z1 - z0) < 1e-12:   # horizontal
            iz, iy = ln*t**3/12, t*ln**3/12
        else:
            iz, iy = t*ln**3/12, ln*t**3/12
        A += a
        Sz += a*zc
        Izz += iz + a*zc**2
        Iyy += iy + a*yc**2
    zc = Sz/A
    return dict(A=A, zc=zc, Ixx=Izz - A*zc**2, Iyy=Iyy)


def lip_width_for_area(A=CH_A, t=CH_T, base=CH_BASE, height=CH_HEIGHT):
    """Width of the flat lips that gives the area ``A`` to the section of
    :func:`channel_section`"""
    return (A/t - (base - t) - 2*(height - t))/2


class Layers:
    r"""Stiffnesses `\boldsymbol{A}`, `\boldsymbol{B}`, `\boldsymbol{D}` of
    a stack of isotropic layers, referred to `z = 0`

    Parameters
    ----------
    layers : list
        ``(z0, z1, E, nu)`` of each layer, `z` along the outward normal; a
        layer with ``nu = 0`` has `Q_{11} = Q_{22} = E`, `Q_{12} = 0`,
        `Q_{66} = G` with ``G`` given as the fifth entry when present
        (``(z0, z1, E, nu, G)``).

    Attributes
    ----------
    A11, ..., D66 : float
        `(A, B, D)_{ij} = \sum Q_{ij} \int (1, z, z^2) dz`.
    """

    def __init__(self, layers):
        for k in ['11', '12', '16', '22', '26', '66']:
            for m in 'ABD':
                setattr(self, m + k, 0.)
        for lay in layers:
            z0, z1, Ey, nu = lay[:4]
            G = lay[4] if len(lay) > 4 else Ey/(2*(1 + nu))
            Q11 = Ey/(1 - nu**2)
            Q = {'11': Q11, '12': nu*Q11, '22': Q11, '66': G}
            ints = {'A': z1 - z0, 'B': (z1**2 - z0**2)/2,
                    'D': (z1**3 - z0**3)/3}
            for k, q in Q.items():
                for m, i in ints.items():
                    setattr(self, m + k, getattr(self, m + k) + q*i)


def layered_ABD(h, ring=None, base=None, E=E, nu=NU, ring_gfac=1.):
    r"""Laminate of the skin of thickness ``h`` with, optionally, a ring
    layer ``ring = (d_r, nu_r)`` on the inner face, `-h/2 - d_r \le z \le
    -h/2`, and a base layer of thickness ``base`` on the outer face. The
    ring layer has the shear modulus `G = E/(2 (1 + \nu))` of the material
    and the Poisson's ratio ``nu_r``: with ``nu_r = 0`` the circumferential
    membrane and bending stiffnesses of the band are `E b_r d_r` and `E b_r
    \int z^2 dz` of a bar, not the plate values `E/(1 - \nu^2)` times
    these. ``ring_gfac`` multiplies the shear modulus of the ring layer."""
    G = E/(2*(1 + nu))
    lay = [(-h/2, h/2, E, nu)]
    if ring is not None:
        dr, nur = ring
        lay.append((-h/2 - dr, -h/2, E, nur, ring_gfac*G))
    if base is not None:
        lay.append((h/2, h/2 + base, E, nu))
    return Layers(lay)


def node_x(bays, dx, rb=RING_B, nring=2):
    """Axial node coordinates: ``nring`` elements across each ring band of
    width ``rb`` centred at the ring stations, the remaining lengths in
    elements of at most ``dx``. Returns ``x`` and the list of the node
    indices of the ring centres (``nring`` even)."""
    st = ring_stations(bays)
    brk = [0.]
    for s in st:
        brk += [s - rb/2, s + rb/2]
    brk.append(sum(bays))
    x = [0.]
    centres = []
    for k in range(len(brk) - 1):
        a, b = brk[k], brk[k + 1]
        inring = k % 2 == 1
        n = nring if inring else max(1, int(np.ceil((b - a)/dx - 1e-9)))
        x += list(a + (b - a)*np.arange(1, n + 1)/n)
        if inring:
            centres.append(len(x) - 1 - n//2)
    return np.array(x), centres


class MooreSector(ChannelCylinderSector):
    r"""Sector of a Moore and Wescoat (1944) specimen

    Parameters
    ----------
    spec : str
        Key of :data:`SPECIMENS`.
    dx : float
        Largest element length along `x` between the ring bands (in).
    nbay, ratio, nbase, nweb, nlip, nint, kinematics :
        As in :class:`channel_cylinder_sector.ChannelCylinderSector`.
    nslices : int
        Ring-only specimens: number of sectors around the circumference
        (the sector angle is `2 \pi/n`); the stiffened specimens use one
        stringer pitch.
    rings : str
        ``'laminate'`` (ring bands of :func:`layered_ABD`), ``'none'`` (no
        ring stiffness, for rigid ring supports or for comparison).
    ring_nu, ring_gfac : float
        Poisson's ratio of the ring layer and factor on its shear modulus
        (:func:`layered_ABD`).
    nring : int
        Number of elements across a ring band (even).
    bl : float
        Width of the flat lips of the channel (in), from the middle surface
        of the web; default from :func:`lip_width_for_area`.
    stringers : bool
        ``False`` leaves the stringers out of a stiffened specimen.
    """

    def __init__(self, spec, dx, nbay=1, ratio=1., nbase=2, nweb=4, nlip=1,
                 nint=4, kinematics='sanders', nslices=120,
                 rings='laminate', ring_nu=0., ring_gfac=1., nring=2,
                 bl=None, stringers=True):
        sp = SPECIMENS[spec]
        self.spec = spec
        t = sp['t']
        self.t = t
        x, self.ring_rows = node_x(sp['bays'], dx, nring=nring)
        ns = sp['ns']
        channel = None
        if ns > 0 and stringers:
            if bl is None:
                bl = lip_width_for_area()
            channel = dict(t=CH_T, b=CH_BASE - CH_T,
                           Hw=t/2 + CH_HEIGHT - CH_T/2, bl=bl,
                           side='external', base='laminate')
        nst = ns if ns > 0 else nslices
        super().__init__(R, L, t, E, NU, 1., nst, x.size, nbay=nbay,
                         ratio=ratio, channel=channel, nbase=nbase,
                         nweb=nweb, nlip=nlip, nint=nint,
                         kinematics=kinematics)
        self.xlin = x
        lex = np.diff(x)
        ncol = self.theta.size
        # ring bands: element rows between the edges of each band
        band = np.zeros(x.size - 1, dtype=bool)
        for c in self.ring_rows:
            band[c - nring//2:c + nring//2] = True
        self.band_rows = band
        for e, shell in enumerate(self.skin_elements):
            i, jcol = divmod(e, ncol - 1)
            shell.lex = lex[i]
            inbase = (self.stiffened and
                      self.j_webs[0] <= jcol < self.j_webs[1])
            if band[i] and rings == 'laminate':
                prop = layered_ABD(t, ring=(RING_D, ring_nu),
                                   base=CH_T if inbase else None,
                                   ring_gfac=ring_gfac)
                assign_constant_ABD(shell, prop)
                shell.h = t + RING_D + (CH_T if inbase else 0.)
        if self.stiffened:
            k = 0
            for s in range(2):
                for nsub in (nweb, nlip):
                    for i in range(x.size - 1):
                        for _ in range(nsub):
                            self.str_elements[k].lex = lex[i]
                            k += 1

    # ------------------------------------------------------------------
    def constrained_bc(self, ends='hinged', rigid_rings=False):
        r"""Constrained reduced degrees of freedom, boolean array, for the
        numbering of the last call of :meth:`Z`

        ``ends='hinged'``: at the nodes of both ends (skin, webs and lips)
        the displacements and their derivatives along the edge, `u = v = w
        = 0` (and `u_{,y}`, `v_{,y}`, `w_{,y}`), `w_{,x}` free;
        ``'clamped'``: also `w_{,x} = w_{,xy} = 0`; ``'hinged, u free'``:
        `v = w = 0` (and `v_{,y}`, `w_{,y}`) only. ``rigid_rings``: `v = w
        = 0` (and `v_{,y}`, `w_{,y}`) at the skin nodes of the ring
        centres."""
        bk = np.zeros(self.Nr, dtype=bool)
        dofs = [U, U_Y, V, V_Y, W, W_Y]
        if ends == 'hinged, u free':
            dofs = [V, V_Y, W, W_Y]
        elif ends == 'clamped':
            dofs += [W_X, W_XY]
        lines = [(i, dofs) for i in [0, self.nx - 1]]
        for i, dd in lines:
            nodes = list(self.skin_nodes[i, :])
            if self.stiffened:
                for s in range(2):
                    nodes += list(self.web_nodes[s][i, :])
                    nodes += list(self.lip_nodes[s][i, :])
            self._mark(bk, nodes, dd)
        if rigid_rings:
            for i in self.ring_rows:
                self._mark(bk, self.skin_nodes[i, :], [V, V_Y, W, W_Y])
        return bk

    def _mark(self, bk, nodes, dofs):
        for nd in nodes:
            for d in dofs:
                rd = self.red[DOF*nd + d]
                if rd >= 0:
                    bk[rd] = True

    def KG_shear(self, Nxy=1.):
        r"""Geometric stiffness matrix of all the degrees of freedom of the
        sector for the constant membrane shear force ``Nxy`` in every skin
        element (lb/in), the plates unloaded"""
        p, w = self.points, self.weights
        size = self.lib.KG_SPARSE_SIZE
        nse = len(self.skin_elements)
        r = np.zeros(size*nse, dtype=self.lib.INT)
        c = np.zeros_like(r)
        v = np.zeros(r.shape[0], dtype=float)
        for e, shell in enumerate(self.skin_elements):
            shell.init_k_KG = e*size
            self.lib.update_KG_constant_stress(shell, p, w, r, c, v, 0., 0.,
                                               Nxy)
        return coo_matrix((v, (r, c)), shape=(self.N, self.N)).tocsr()

    def KG_u(self, u):
        r"""Geometric stiffness matrix of all the degrees of freedom of the
        sector from the displacements ``u`` (real, all the degrees of
        freedom), plates unloaded"""
        p, w = self.points, self.weights
        size = self.lib.KG_SPARSE_SIZE
        nse = len(self.skin_elements)
        r = np.zeros(size*nse, dtype=self.lib.INT)
        c = np.zeros_like(r)
        v = np.zeros(r.shape[0], dtype=float)
        u = np.ascontiguousarray(u, dtype=float)
        for e, shell in enumerate(self.skin_elements):
            shell.init_k_KG = e*size
            self.lib.update_KG(u, shell, p, w, r, c, v)
        return coo_matrix((v, (r, c)), shape=(self.N, self.N)).tocsr()

    # ------------------------------------------------------------------
    def buckling(self, j, ends='hinged', rigid_rings=False, nev=4, KG=None,
                 both=False):
        r"""Lowest positive load multipliers of the Bloch index ``j``

        `(\boldsymbol{K} + \lambda \boldsymbol{K}_G) \boldsymbol{x} = 0`,
        reduced by :meth:`Z` and :meth:`constrained_bc`, Hermitian; the
        largest eigenvalues `\mu = 1/\lambda` of `\boldsymbol{K}^{-1}
        (-\boldsymbol{K}_G)` are computed with ARPACK. ``KG`` defaults to
        :meth:`KG_shear` with `N_{xy} = 1`, so that `\lambda` is the
        critical `N_{xy}` (lb/in). Returns ``lam`` (ascending, positive) and
        the modes (all the degrees of freedom); with ``both`` also the
        multipliers of the opposite sign, as ``(lam, modes, lam_neg)``."""
        Zj = self.Z(j)
        if KG is None:
            if not hasattr(self, '_KGs'):
                self._KGs = self.KG_shear(1.)
            KG = self._KGs
        K = (Zj.conj().T @ self.KC0() @ Zj).tocsr()
        G = (Zj.conj().T @ KG @ Zj).tocsr()
        bu = ~self.constrained_bc(ends, rigid_rings)
        Kuu = K[bu][:, bu].tocsc()
        Guu = -G[bu][:, bu]
        lu = splu(Kuu)
        dtype = np.result_type(Kuu.dtype, Guu.dtype)
        op = LinearOperator(Kuu.shape, matvec=lambda x: lu.solve(Guu @ x),
                            dtype=dtype)
        k = min(nev, Kuu.shape[0] - 2)
        ncv = min(Kuu.shape[0] - 1, max(2*k + 1, 24))
        mu, X = eigs(op, k=k, which='LR', ncv=ncv, tol=1e-10)
        mu = mu.real
        order = np.argsort(-mu)
        mu, X = mu[order], X[:, order]
        pos = mu > 0
        lam = 1/mu[pos]
        ur = np.zeros((self.Nr, pos.sum()), dtype=X.dtype)
        ur[bu] = X[:, pos]
        modes = Zj @ ur
        if not both:
            return lam, modes
        mun = eigs(op, k=k, which='SR', ncv=ncv, tol=1e-10,
                   return_eigenvectors=False).real
        lamn = np.sort(1/mun[mun < 0])[::-1]
        return lam, modes, lamn

    def scan(self, js, ends='hinged', rigid_rings=False, nev=2):
        """Lowest positive multiplier of each Bloch index of ``js``: a
        dictionary ``j -> (lam, mode)``"""
        out = {}
        for j in js:
            lam, modes = self.buckling(j, ends, rigid_rings, nev=nev)
            out[j] = (lam[0], modes[:, 0])
        return out

    # ------------------------------------------------------------------
    def static_rotation(self, phi=1e-3, ends='hinged'):
        r"""Torsion of the axisymmetric (Bloch index 0) sector by the
        rotation ``phi`` of the end `x = L` about the axis: `v = R \phi`,
        `v_{,y} = 0` at its skin nodes, the other end conditions of
        :meth:`constrained_bc`. Returns the displacements (all the
        degrees of freedom) and the torque of the whole cylinder (lb in),
        `R` times the sum of the reactions on the prescribed `v` times the
        number of sectors."""
        Z0 = self.Z(0)
        K = (Z0.T @ self.KC0() @ Z0).tocsr()
        bk = self.constrained_bc(ends)
        up = np.zeros(self.Nr)
        pres = []
        for nd in self.skin_nodes[-1, :]:
            rd = self.red[DOF*nd + V]
            if rd >= 0:
                up[rd] = R*phi
                pres.append(rd)
        bu = ~bk
        Kuu = K[bu][:, bu].tocsc()
        ur = up.copy()
        ur[bu] = splu(Kuu).solve(-(K[bu] @ up))
        f = K @ ur
        T = R*f[pres].sum()*self.nstringers
        return Z0 @ ur, T

    def wave(self, mode, j):
        """`w` around the whole circumference, real part (see
        :meth:`ChannelCylinderSector.full_w`), with the phase that makes
        the largest value real"""
        from channel_cylinder_sector import normalize_phase
        return self.full_w(normalize_phase(mode), j)


def stringer_band_ABD(h, b, bl=None, ring=None, E=E, nu=NU):
    r"""Laminate of the skin band of width ``b`` under a stringer
    represented as a layer: skin of thickness ``h`` plus a layer with
    `Q_{11} = E_s`, the other `Q_{ij}` zero, of thickness `d_s = \sqrt{12
    I_{XX}/A}` centred at the centroid of the idealised channel of
    :func:`channel_section` (lips ``bl``), `E_s b d_s = E A`: the axial
    stiffness, the eccentricity and the bending stiffness `E I_{XX}` of the
    stringer are those of the section; its lateral bending and torsional
    stiffnesses are not. ``ring``: ``(d_r, nu_r)`` of a ring layer on the
    inner face, as in :func:`layered_ABD`."""
    if bl is None:
        bl = lip_width_for_area()
    sec = channel_section(bl)
    lam = layered_ABD(h, ring=ring, E=E, nu=nu)
    ds = np.sqrt(12*sec['Ixx']/sec['A'])
    Es = E*sec['A']/(b*ds)
    zc = h/2 + sec['zc']
    z0, z1 = zc - ds/2, zc + ds/2
    lam.A11 += Es*(z1 - z0)
    lam.B11 += Es*(z1**2 - z0**2)/2
    lam.D11 += Es*(z1**3 - z0**3)/3
    return lam


class SkinBandSector:
    r"""Sector of one stringer pitch of a Moore and Wescoat (1944) specimen
    of skin elements only (:class:`bfsccylinder.sanders.BFSCCylinderSanders`):
    the rings as in :class:`MooreSector`, each stringer as a band of the
    skin of the width of the base (between the middle surfaces of the webs)
    with the laminate of :func:`stringer_band_ABD`. The Bloch condition,
    the boundary conditions and the geometric stiffness matrix are those of
    :class:`MooreSector`, whose methods it uses.

    Parameters
    ----------
    spec : str
        Key of a stiffened specimen of :data:`SPECIMENS`.
    dx, nbay, ratio, nbase, nint, nring :
        As in :class:`MooreSector`.
    """

    def __init__(self, spec, dx, nbay=8, ratio=1., nbase=2, nint=4,
                 nring=2):
        import bfsccylinder.sanders as sanders
        from bfsccylinder.quadrature import get_points_weights
        from channel_cylinder_sector import graded
        sp = SPECIMENS[spec]
        self.spec = spec
        self.t = t = sp['t']
        self.lib = sanders
        self.R, self.L = R, L
        self.stiffened = False
        self.nstringers = sp['ns']
        self.phi = phi = 2*pi/self.nstringers
        x, self.ring_rows = node_x(sp['bays'], dx, nring=nring)
        self.xlin = x
        self.nx = nx = x.size
        b = CH_BASE - CH_T
        beta = b/(2*R)
        half = graded(R*(phi/2 - beta), nbay, ratio)/R
        dtheta = np.concatenate([half, np.full(nbase, 2*beta/nbase),
                                 half[::-1]])
        self.theta = np.concatenate([[0.], np.cumsum(dtheta)])
        self.theta[-1] = phi
        ncol = self.theta.size
        self.skin_nodes = skin = np.arange(nx*ncol).reshape(nx, ncol)
        self.num_nodes = skin.size
        self.N = DOF*skin.size
        self.points, self.weights = get_points_weights(nint=nint)
        band = np.zeros(nx - 1, dtype=bool)
        for c in self.ring_rows:
            band[c - nring//2:c + nring//2] = True
        ring = (RING_D, 0.)
        lams = {(False, False): layered_ABD(t),
                (True, False): layered_ABD(t, ring=ring),
                (False, True): stringer_band_ABD(t, b),
                (True, True): stringer_band_ABD(t, b, ring=ring)}
        self.skin_elements = []
        lex = np.diff(x)
        for i in range(nx - 1):
            for j in range(ncol - 1):
                shell = sanders.BFSCCylinderSanders(nint)
                shell.n1, shell.n2 = skin[i, j], skin[i + 1, j]
                shell.n3, shell.n4 = skin[i + 1, j + 1], skin[i, j + 1]
                shell.c1, shell.c2 = DOF*shell.n1, DOF*shell.n2
                shell.c3, shell.c4 = DOF*shell.n3, DOF*shell.n4
                shell.R = R
                shell.lex = lex[i]
                shell.ley = R*dtheta[j]
                shell.rho = 1.
                shell.h = t
                inbase = nbay <= j < nbay + nbase
                assign_constant_ABD(shell, lams[(bool(band[i]), inbase)])
                self.skin_elements.append(shell)

    def Z(self, j):
        r"""Constraint basis of the Bloch index ``j``: the nodes of the edge
        `\theta = \phi` are those of `\theta = 0` times `e^{i j \phi}`"""
        skin = self.skin_nodes
        if j == 0:
            phase = 1.
        elif 2*j == self.nstringers:
            phase = -1.
        else:
            phase = np.exp(1j*j*self.phi)
        dep = np.zeros(self.N, dtype=bool)
        dep[((DOF*skin[:, -1])[:, None] + np.arange(DOF)).ravel()] = True
        Nr = int((~dep).sum())
        red = -np.ones(self.N, dtype=int)
        red[~dep] = np.arange(Nr)
        rows = list(np.flatnonzero(~dep))
        cols = list(np.arange(Nr))
        vals = [1.]*Nr
        for i in range(self.nx):
            for d in range(DOF):
                rows.append(DOF*skin[i, -1] + d)
                cols.append(red[DOF*skin[i, 0] + d])
                vals.append(phase)
        dtype = complex if isinstance(phase, complex) else float
        self.red, self.Nr = red, Nr
        return coo_matrix((np.array(vals, dtype=dtype), (rows, cols)),
                          shape=(self.N, Nr)).tocsr()

    def KC0(self):
        """Linear stiffness matrix of all the degrees of freedom, computed
        once"""
        if not hasattr(self, '_KC0'):
            size = self.lib.KC0_SPARSE_SIZE
            n = len(self.skin_elements)
            r = np.zeros(size*n, dtype=self.lib.INT)
            c = np.zeros_like(r)
            v = np.zeros(r.shape[0], dtype=float)
            for e, shell in enumerate(self.skin_elements):
                shell.init_k_KC0 = e*size
                self.lib.update_KC0(shell, self.points, self.weights, r, c,
                                    v)
            self._KC0 = coo_matrix((v, (r, c)),
                                   shape=(self.N, self.N)).tocsr()
        return self._KC0

    constrained_bc = MooreSector.constrained_bc
    _mark = MooreSector._mark
    KG_shear = MooreSector.KG_shear
    buckling = MooreSector.buckling
    scan = MooreSector.scan
    wave = MooreSector.wave

    def full_w(self, mode, j, npts=1440, real=True):
        """As :meth:`channel_cylinder_sector.ChannelCylinderSector.full_w`"""
        return ChannelCylinderSector.full_w(self, mode, j, npts, real)


def lowest(kw):
    """Builds :class:`MooreSector` (or :class:`SkinBandSector` with the key
    ``model='band'``) from the dictionary ``kw`` (plus the keys ``js``,
    ``ends``, ``rigid_rings``), computes the lowest positive multiplier of
    each Bloch index of ``js`` and returns ``(kw, results)``, ``results``
    with the keys ``tau`` (critical shear stress `N_{xy}/t`, psi, per
    index), ``j`` (index of the lowest), ``tau_min``, ``mode`` (its mode,
    all the degrees of freedom), ``x``, ``ndof`` (reduced degrees of
    freedom of the index 0) and ``time`` (s); a top-level function, for
    :mod:`multiprocessing`"""
    import time
    t0 = time.time()
    kw = dict(kw)
    opts = {k: kw.pop(k) for k in ['js', 'ends', 'rigid_rings']
            if k in kw}
    js = opts.pop('js')
    model_kind = kw.pop('model', 'discrete')
    cls = SkinBandSector if model_kind == 'band' else MooreSector
    model = cls(**kw)
    res = model.scan(js, **opts)
    tau = {j: res[j][0]/model.t for j in js}
    jm = min(tau, key=tau.get)
    theta, w = model.wave(res[jm][1], jm)
    model.Z(0)
    out = dict(tau=tau, j=jm, tau_min=tau[jm], x=model.xlin, theta=theta,
               w=w, ndof=int((~model.constrained_bc()).sum()),
               time=time.time() - t0)
    return kw, out
