r"""Periodic sector of a cylinder with a few channel (hat-section) stringers

A circular cylinder with ``nstringers`` identical stringers, equally spaced,
is modelled by one stringer pitch, the sector `0 \le \theta \le \phi`,
`\phi = 2 \pi/n_s`, with the stringer at its middle, `\theta_c = \phi/2`.
The skin is meshed with the BFSC cylinder element of Sanders-type
kinematics, :class:`bfsccylinder.sanders.BFSCCylinderSanders` (or of
Donnell-type kinematics, :class:`bfsccylinder.BFSCCylinder`); the
stringer, a channel whose base lies on the skin, with two webs normal to
the skin and a lip at the free edge of each web, bent at 90 degrees away
from the channel, is assembled from BFSC plate elements,
:class:`bfscplate2d.BFSCPlate2D`:

- the base is not a plate: the skin elements under it, between the two
  webs, have the stiffnesses of the two-layer laminate skin + base, with
  the matrices `\boldsymbol{A}`, `\boldsymbol{B}`, `\boldsymbol{D}`
  referred to the middle surface of the skin, and the mass of both layers;
- each web is a plate in the plane `(x, r)` through the line of the skin
  where it stands, `\theta = \theta_c \mp \beta`, `\beta = b/(2R)`, from
  the middle surface of the skin to the middle surface of the lip;
- each lip is a flat plate in the plane `(x, s)` tangent to the
  circumference at the tip of its web, `s` the arc coordinate along
  `\theta`, of width `b_l` from the middle surface of the web to the free
  edge.

Ties
----
A web has the local coordinates `X = x` and `Y = r - R`; its displacements
`U`, `V` and `W` are along `x`, along the outward radius and along the
circumference. Its nodes along the root are tied to the skin nodes of its
line, as for a blade stringer:

.. math::

    U = u, \quad V = w, \quad W = v, \quad W_{,Y} = -w_{,y} + \frac{v}{R}

with the derivatives of these relations along `x` (without `v/R` for the
Donnell-type kinematics). A lip has the local
coordinates `X = x` and `Y = s`, its displacements `U`, `V` and `W` along
`x`, along the circumference and along the outward radius. Its nodes along
the junction are tied to the tip nodes of the web, a rigid corner of 90
degrees:

.. math::

    U_l = U_w, \quad V_l = W_w, \quad W_l = V_w, \quad
    W_{l,Y} = -W_{w,Y}

with the derivatives along `x`: a rotation `\psi` about `x` gives
`\partial u_r/\partial s = \psi` along the lip and `\partial
u_\theta/\partial r = -\psi` along the web.

Cyclic symmetry
---------------
As in the sector of the blade-stiffened cylinder of this folder, the nodes
of the edge `\theta = \phi` are eliminated with `\boldsymbol{u}(\theta =
\phi) = e^{i j \phi}\,\boldsymbol{u}(\theta = 0)`, which gives one
Hermitian problem per Bloch index `j = 0, 1, \ldots, n_s/2`. With few
stringers the index `j` contains the circumferential harmonics `n = |j +
k n_s|`, `k` any integer, coupled by the stringers; each mode is classified
afterwards by the Fourier series of its normal displacement `w` around the
whole circumference (:func:`classify`).

"""
import numpy as np
from numpy import pi
from scipy.interpolate import CubicHermiteSpline
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, eigs, splu

from composites import isotropic_plate
import bfsccylinder
import bfsccylinder.sanders as sanders
from bfsccylinder.quadrature import get_points_weights
import bfscplate2d
from bfscplate2d import BFSCPlate2D

DOF = 10
# positions of the degrees of freedom in a node
U, U_X, U_Y, V, V_X, V_Y, W, W_X, W_Y, W_XY = range(10)

#: constrained degrees of freedom of the end nodes (skin and plates) of the
#: freely supported cylinder: the displacements normal to the axis and
#: their derivatives along the end edge; the axial displacement is free
FREELY_SUPPORTED = [V, V_Y, W, W_Y]


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
        ``'external'``, at `h/2 \le z \le h/2 + t`, with `z` along the outward
        normal.

    Attributes
    ----------
    A11, ..., D66 : float
        The entries of `\boldsymbol{A}`, `\boldsymbol{B}` and
        `\boldsymbol{D}`, `(A, B, D)_{ij} = Q_{ij} \int (1, z, z^2) dz`
        over both layers, with the plane-stress stiffnesses `Q_{11} = Q_{22}
        = E/(1 - \nu^2)`, `Q_{12} = \nu Q_{11}`, `Q_{66} = E/(2 (1 + \nu))`.
    """

    def __init__(self, h, t, E, nu, side='internal'):
        if side == 'internal':
            z = [-h/2 - t, -h/2, h/2]
        else:
            z = [-h/2, h/2, h/2 + t]
        z0, z3 = z[0], z[-1]
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


def graded(length, nel, ratio):
    """Lengths of ``nel`` elements over ``length``, in geometric progression
    from the largest to the smallest, ``ratio`` = largest/smallest"""
    if nel == 1 or ratio == 1:
        return np.full(nel, length/nel)
    q = ratio**(1/(nel - 1))
    s = q**np.arange(nel)[::-1]
    return length*s/s.sum()


class ChannelCylinderSector:
    r"""Sector of one stringer pitch of a cylinder with channel stringers

    Parameters
    ----------
    R, L, h : float
        Radius of the middle surface, length and thickness of the skin.
    E, nu, rho : float
        Young's modulus, Poisson's ratio and density, skin and stringer.
    nstringers : int
        Number of stringers of the cylinder; the sector angle is
        `\phi = 2 \pi/n_s`.
    nx : int
        Number of nodes along `x`.
    nbay : int
        Number of skin elements between an edge of the sector and the
        nearest web (without stringer: half the number of skin elements
        across the sector).
    ratio : float
        Ratio of the largest to the smallest of these elements, the
        smallest next to the web.
    channel : dict or None
        ``None`` for an unstiffened sector; otherwise the keys ``t``
        (thickness of the channel), ``b`` (distance between the middle
        surfaces of the webs), ``Hw`` (height of the web plates, from the
        middle surface of the skin to the middle surface of the lips),
        ``bl`` (width of a lip, from the middle surface of its web to the
        free edge), ``side`` (``'internal'`` or ``'external'``) and ``base``
        (``'laminate'``: the skin under the base is the laminate skin +
        base of :class:`TwoLayerSkin`; ``'none'``: the base is left out).
    nbase, nweb, nlip : int
        Number of elements across the base, along the height of a web and
        across a lip.
    nint : int
        Number of Gauss-Legendre points per direction.
    kinematics : str
        ``'sanders'``: the skin elements are
        :class:`bfsccylinder.sanders.BFSCCylinderSanders`; ``'donnell'``:
        :class:`bfsccylinder.BFSCCylinder`, and the rotation of the skin in
        the tie of the webs is `-w_{,y}`, without `v/R`.
    """

    def __init__(self, R, L, h, E, nu, rho, nstringers, nx, nbay=8,
                 ratio=1., channel=None, nbase=2, nweb=4, nlip=1, nint=4,
                 kinematics='sanders'):
        assert kinematics in ('sanders', 'donnell')
        self.kinematics = kinematics
        self.lib = sanders if kinematics == 'sanders' else bfsccylinder
        self.R, self.L, self.h = R, L, h
        self.E, self.nu, self.rho = E, nu, rho
        self.nstringers = nstringers
        self.phi = phi = 2*pi/nstringers
        self.nx = nx
        self.channel = channel
        self.stiffened = channel is not None
        self.points, self.weights = get_points_weights(nint=nint)

        # circumferential positions of the skin nodes, symmetric about phi/2
        if self.stiffened:
            c = channel
            assert c['side'] in ('internal', 'external')
            beta = c['b']/(2*R)
            half = graded(R*(phi/2 - beta), nbay, ratio)/R
            base = np.full(nbase, 2*beta/nbase)
            dtheta = np.concatenate([half, base, half[::-1]])
            self.j_webs = [nbay, nbay + nbase]
        else:
            dtheta = np.full(2*nbay, phi/(2*nbay))
        self.theta = np.concatenate([[0.], np.cumsum(dtheta)])
        self.theta[-1] = phi
        ncol = self.theta.size
        skin = np.arange(nx*ncol).reshape(nx, ncol)
        self.skin_nodes = skin
        num_nodes = skin.size
        self.xlin = np.linspace(0, L, nx)
        lex = L/(nx - 1)

        if self.stiffened:
            sign = 1. if c['side'] == 'external' else -1.
            self.Yweb = sign*np.linspace(0, c['Hw'], nweb + 1)
            self.web_nodes = []
            for _ in range(2):
                self.web_nodes.append(num_nodes + np.arange(
                    nx*(nweb + 1)).reshape(nx, nweb + 1))
                num_nodes += nx*(nweb + 1)
            # lip 0 (theta_c - beta) points to smaller theta, lip 1 to larger
            # theta; Y = s from the junction, junction at k = nlip and k = 0
            self.Ylip = [np.linspace(-c['bl'], 0, nlip + 1),
                         np.linspace(0, c['bl'], nlip + 1)]
            self.k_junction = [nlip, 0]
            self.lip_nodes = []
            for _ in range(2):
                self.lip_nodes.append(num_nodes + np.arange(
                    nx*(nlip + 1)).reshape(nx, nlip + 1))
                num_nodes += nx*(nlip + 1)
        self.num_nodes = num_nodes
        self.N = DOF*num_nodes

        self.skin_prop = isotropic_plate(thickness=h, E=E, nu=nu, rho=rho)
        if self.stiffened and c['base'] == 'laminate':
            self.base_prop = TwoLayerSkin(h, c['t'], E, nu, side=c['side'])
        self.skin_elements = []
        for i in range(nx - 1):
            for j in range(ncol - 1):
                if kinematics == 'sanders':
                    shell = sanders.BFSCCylinderSanders(nint)
                else:
                    shell = bfsccylinder.BFSCCylinder(nint)
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
                shell.rho = rho
                prop = self.skin_prop
                shell.h = h
                if (self.stiffened and c['base'] == 'laminate'
                        and self.j_webs[0] <= j < self.j_webs[1]):
                    prop = self.base_prop
                    shell.h = h + c['t']
                _assign_ABD(shell, prop)
                self.skin_elements.append(shell)

        self.str_elements = []
        if self.stiffened:
            self.str_prop = isotropic_plate(thickness=c['t'], E=E, nu=nu,
                                            rho=rho)
            for s in range(2):
                nodes = self.web_nodes[s]
                for i in range(nx - 1):
                    for k in range(nweb):
                        # node 1 at the smallest X and Y
                        k1, k4 = (k, k + 1) if sign > 0 else (k + 1, k)
                        self._plate(nodes[i, k1], nodes[i + 1, k1],
                                    nodes[i + 1, k4], nodes[i, k4], lex,
                                    c['Hw']/nweb)
                nodes = self.lip_nodes[s]
                for i in range(nx - 1):
                    for k in range(nlip):
                        self._plate(nodes[i, k], nodes[i + 1, k],
                                    nodes[i + 1, k + 1], nodes[i, k + 1], lex,
                                    c['bl']/nlip)

    def _plate(self, n1, n2, n3, n4, lex, ley):
        plate = BFSCPlate2D()
        plate.n1, plate.n2, plate.n3, plate.n4 = n1, n2, n3, n4
        plate.c1 = DOF*n1
        plate.c2 = DOF*n2
        plate.c3 = DOF*n3
        plate.c4 = DOF*n4
        plate.lex = lex
        plate.ley = ley
        plate.h = self.channel['t']
        plate.rho = self.rho
        plate.ABD = self.str_prop.ABD
        self.str_elements.append(plate)

    # ------------------------------------------------------------------
    def Z(self, j):
        r"""Constraint basis `\boldsymbol{Z}_j` of the Bloch index ``j``,
        real for ``j = 0`` and ``j = n_s/2``, complex otherwise; sets the
        attributes ``red`` (index of each degree of freedom among the
        reduced ones, ``-1`` for the eliminated ones) and ``Nr``"""
        skin = self.skin_nodes
        ncol = skin.shape[1]
        if j == 0:
            phase = 1.
        elif 2*j == self.nstringers:
            phase = -1.
        else:
            phase = np.exp(1j*j*self.phi)
        dependent = {}
        for i in range(self.nx):
            cl = DOF*skin[i, 0]
            cr = DOF*skin[i, ncol - 1]
            for d in range(DOF):
                dependent[cr + d] = [(cl + d, phase)]
        if self.stiffened:
            # v/R in the rotation of the skin, Sanders-type kinematics only
            Rinv = 1/self.R if self.kinematics == 'sanders' else 0.
            for s in range(2):
                for i in range(self.nx):
                    # web root to the skin
                    c = DOF*skin[i, self.j_webs[s]]
                    b = DOF*self.web_nodes[s][i, 0]
                    dependent[b + U] = [(c + U, 1.)]
                    dependent[b + U_X] = [(c + U_X, 1.)]
                    dependent[b + V] = [(c + W, 1.)]
                    dependent[b + V_X] = [(c + W_X, 1.)]
                    dependent[b + W] = [(c + V, 1.)]
                    dependent[b + W_X] = [(c + V_X, 1.)]
                    dependent[b + W_Y] = [(c + W_Y, -1.), (c + V, Rinv)]
                    dependent[b + W_XY] = [(c + W_XY, -1.), (c + V_X, Rinv)]
                    # lip junction to the web tip
                    c = DOF*self.web_nodes[s][i, -1]
                    b = DOF*self.lip_nodes[s][i, self.k_junction[s]]
                    dependent[b + U] = [(c + U, 1.)]
                    dependent[b + U_X] = [(c + U_X, 1.)]
                    dependent[b + V] = [(c + W, 1.)]
                    dependent[b + V_X] = [(c + W_X, 1.)]
                    dependent[b + W] = [(c + V, 1.)]
                    dependent[b + W_X] = [(c + V_X, 1.)]
                    dependent[b + W_Y] = [(c + W_Y, -1.)]
                    dependent[b + W_XY] = [(c + W_XY, -1.)]
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
                rows.append(d)
                cols.append(red[d2])
                vals.append(coef)
        dtype = complex if isinstance(phase, complex) else float
        self.red, self.Nr = red, Nr
        return coo_matrix((np.array(vals, dtype=dtype), (rows, cols)),
                          shape=(self.N, Nr)).tocsr()

    def constrained(self):
        """Constrained reduced degrees of freedom of the freely supported
        ends, boolean array, for the numbering of the last call of
        :meth:`Z`"""
        bk = np.zeros(self.Nr, dtype=bool)
        for i in [0, self.nx - 1]:
            nodes = list(self.skin_nodes[i, :])
            if self.stiffened:
                for s in range(2):
                    nodes += list(self.web_nodes[s][i, :])
                    nodes += list(self.lip_nodes[s][i, :])
            for nd in nodes:
                for d in FREELY_SUPPORTED:
                    rd = self.red[DOF*nd + d]
                    if rd >= 0:
                        bk[rd] = True
        return bk

    # ------------------------------------------------------------------
    def _assemble(self, skin_func, skin_size, str_func, str_size):
        nse = len(self.skin_elements)
        nst = len(self.str_elements)
        r = np.zeros(skin_size*nse + str_size*nst, dtype=self.lib.INT)
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
        """Linear stiffness matrix of all the degrees of freedom of the
        sector, computed once"""
        if not hasattr(self, '_KC0'):
            p, w = self.points, self.weights

            def skin(shell, k, r, c, v):
                shell.init_k_KC0 = k
                self.lib.update_KC0(shell, p, w, r, c, v)

            def stringer(plate, k, r, c, v):
                plate.init_k_KC0 = k
                bfscplate2d.update_KC0(plate, p, w, r, c, v)

            self._KC0 = self._assemble(skin, self.lib.KC0_SPARSE_SIZE,
                                       stringer, bfscplate2d.KC0_SPARSE_SIZE)
        return self._KC0

    def M(self, skin_inplane_inertia=True):
        """Mass matrix of all the degrees of freedom of the sector, computed
        once; with ``skin_inplane_inertia=False`` the rows and columns of
        `u`, `v` and their derivatives of the skin nodes are zero (the
        in-surface inertia of the skin neglected), the plates unchanged"""
        if not hasattr(self, '_M'):
            def skin(shell, k, r, c, v):
                shell.init_k_M = k
                self.lib.update_M(shell, r, c, v)

            def stringer(plate, k, r, c, v):
                plate.init_k_M = k
                bfscplate2d.update_M(plate, r, c, v)

            self._M = self._assemble(skin, self.lib.M_SPARSE_SIZE, stringer,
                                     bfscplate2d.M_SPARSE_SIZE)
        if skin_inplane_inertia:
            return self._M
        keep = np.ones(self.N)
        dofs = DOF*self.skin_nodes.ravel()
        for d in [U, U_X, U_Y, V, V_X, V_Y]:
            keep[dofs + d] = 0.
        P = coo_matrix((keep, (np.arange(self.N), np.arange(self.N))))
        return (P @ self._M @ P).tocsr()

    # ------------------------------------------------------------------
    def full_w(self, mode, j, npts=1440, real=True):
        r"""Normal displacement `w` of the skin around the whole
        circumference from the mode ``mode`` of the Bloch index ``j``: the
        sector `k` is the sector `0` multiplied by `e^{i j k \phi}`, `w` is
        interpolated between the nodes with the cubic Hermite polynomials of
        its values and of `w_{,y}`, and its real part is taken
        (``real=True``) or not. Returns
        `\theta' = \theta - \phi/2` (the first stringer at `\theta' = 0`),
        ``npts`` equally spaced points over `[0, 2 \pi)`, and `w`, shape
        ``(nx, npts)``"""
        w = mode[DOF*self.skin_nodes + W]
        wy = mode[DOF*self.skin_nodes + W_Y]
        th, f, df = [], [], []
        for k in range(self.nstringers):
            fac = np.exp(1j*j*k*self.phi)
            th.append(self.theta[:-1] + k*self.phi)
            f.append(w[:, :-1]*fac)
            df.append(wy[:, :-1]*fac*self.R)
        th.append([2*pi])
        f.append(w[:, :1]*np.exp(1j*j*2*pi))
        df.append(wy[:, :1]*np.exp(1j*j*2*pi)*self.R)
        th = np.concatenate(th)
        spline = CubicHermiteSpline(th, np.hstack(f).T, np.hstack(df).T,
                                    axis=0)
        thetap = np.arange(npts)*2*pi/npts
        values = spline((thetap + self.phi/2) % (2*pi)).T
        if real:
            values = np.real(values)
        return thetap, values


def normalize_phase(u):
    """Mode ``u`` multiplied by the unit complex number that makes its
    largest entry real and positive"""
    i = np.argmax(np.abs(u))
    ui = u.ravel()[i]
    return u*np.conj(ui)/abs(ui)


def classify(model, mode, j, mmax=12, nmax=40):
    r"""Axial and circumferential wave numbers of a freely supported mode

    `w(x, \theta')` around the whole circumference (:meth:`full_w`) is
    expanded in the series of the analysis of Egle and Sewall (1968), `w =
    \sum_m \sum_n (a_{mn} \cos n\theta' + b_{mn} \sin n\theta') \sin(m \pi
    x/L)`, `\theta'` measured from a stringer, by numerical integration.
    Returns a dictionary: ``m`` and ``n`` of the largest amplitude
    `\sqrt{a_{mn}^2 + b_{mn}^2}`; ``ratio``, the second largest amplitude
    over the largest (for a mode of a complex Bloch index, a combination of
    the symmetric and the antisymmetric mode of a pair, the amplitudes are
    those of either mode of the pair);
    ``sym``, the fraction `\sum a_{mn}^2/\sum (a_{mn}^2 + b_{mn}^2)` (1 for a
    mode symmetric about the plane of a stringer, 0 antisymmetric); and the
    arrays ``a`` and ``b``, shape ``(mmax, nmax + 1)``."""
    mode = normalize_phase(mode)
    thetap, w = model.full_w(mode, j)
    x = model.xlin
    npts = thetap.size
    c = np.fft.rfft(w, axis=1)[:, :nmax + 1]*2/npts
    c[:, 0] /= 2
    # sine series along x by the trapezoidal rule on the nodes
    S = np.sin(np.outer(np.arange(1, mmax + 1), pi*x/model.L))
    wts = np.full(x.size, x[1] - x[0])
    wts[[0, -1]] /= 2
    coef = (S*wts) @ c*2/model.L
    a, b = np.real(coef), -np.imag(coef)
    amp = np.sqrt(a**2 + b**2)
    flat = np.sort(amp.ravel())
    im, n = np.unravel_index(np.argmax(amp), amp.shape)
    return dict(m=im + 1, n=n, ratio=flat[-2]/flat[-1],
                sym=float((a**2).sum()/(amp**2).sum()), a=a, b=b)


def frequencies(model, j, fmax, k=40, skin_inplane_inertia=True,
                sigma=-1.):
    r"""Natural frequencies (Hz) up to ``fmax`` and modes (all the degrees of
    freedom of the sector) of the Bloch index ``j`` of the freely supported
    cylinder

    The eigenvalues `\omega^2` of `\boldsymbol{K} \boldsymbol{x} = \omega^2
    \boldsymbol{M} \boldsymbol{x}`, reduced by :meth:`Z` and the end
    conditions, Hermitian, `\boldsymbol{M}` possibly singular, are the
    largest eigenvalues `1/(\omega^2 - \sigma)` of `(\boldsymbol{K} - \sigma
    \boldsymbol{M})^{-1} \boldsymbol{M}` (ARPACK through
    :func:`scipy.sparse.linalg.eigs`); ``k`` of them are computed first,
    and twice as many until the largest frequency exceeds ``fmax``. The
    rigid-body axial translation (``j = 0``) is left out (frequencies below
    1 Hz)."""
    Zj = model.Z(j)
    K = (Zj.conj().T @ model.KC0() @ Zj).tocsr()
    M = (Zj.conj().T @ model.M(skin_inplane_inertia) @ Zj).tocsr()
    bu = ~model.constrained()
    Kuu = K[bu][:, bu]
    Muu = M[bu][:, bu]
    lu = splu((Kuu - sigma*Muu).tocsc())
    dtype = np.result_type(Kuu.dtype, Muu.dtype)
    op = LinearOperator(Kuu.shape, matvec=lambda x: lu.solve(Muu @ x),
                        dtype=dtype)
    while True:
        k = min(k, Kuu.shape[0] - 2)
        ncv = min(Kuu.shape[0] - 1, max(2*k + 1, 20))
        nu, X = eigs(op, k=k, which='LR', ncv=ncv, tol=1e-9)
        nu = nu.real
        ok = nu > 1e-12
        nu, X = nu[ok], X[:, ok]
        order = np.argsort(-nu)
        omega2, X = sigma + 1/nu[order], X[:, order]
        f = np.sqrt(np.maximum(omega2, 0.))/(2*pi)
        if f.max() > fmax or k >= Kuu.shape[0] - 2:
            break
        k *= 2
    keep = (f > 1.) & (f <= fmax)
    ur = np.zeros((model.Nr, keep.sum()), dtype=X.dtype)
    ur[bu] = X[:, keep]
    return f[keep], Zj @ ur


def spectrum(model, fmax, js=None, skin_inplane_inertia=True,
             keep_modes=True):
    """All the modes up to ``fmax`` of the Bloch indices ``js`` (default
    ``0`` to ``n_s/2``), classified by :func:`classify`: a list of
    dictionaries with the keys of :func:`classify` and ``f`` (Hz), ``j`` and,
    with ``keep_modes``, ``mode``. A mode of ``0 < j < n_s/2`` stands for a
    pair of modes of equal frequency, one symmetric and one antisymmetric
    about the plane of a stringer."""
    if js is None:
        js = range(model.nstringers//2 + 1)
    out = []
    for j in js:
        f, X = frequencies(model, j, fmax,
                           skin_inplane_inertia=skin_inplane_inertia)
        for i in range(f.size):
            d = classify(model, X[:, i], j)
            d.update(f=f[i], j=j)
            if keep_modes:
                d['mode'] = X[:, i]
            out.append(d)
    return out


def solve_case(case):
    """Builds :class:`ChannelCylinderSector` from the dictionary ``case``
    (its keyword arguments plus the keys ``fmax``, ``js``,
    ``skin_inplane_inertia`` and ``keep_modes`` of :func:`spectrum`) and
    returns ``(case, spectrum, number of complex degrees of freedom of the
    reduced sector, time in s)``; a top-level function, for
    :mod:`multiprocessing`"""
    import time
    t0 = time.time()
    kw = dict(case)
    opts = {key: kw.pop(key) for key in ['fmax', 'js', 'skin_inplane_inertia',
                                        'keep_modes'] if key in kw}
    fmax = opts.pop('fmax')
    model = ChannelCylinderSector(**kw)
    spec = spectrum(model, fmax, **opts)
    model.Z(0)
    return case, spec, int((~model.constrained()).sum()), time.time() - t0


def table(spec, nstringers):
    """Lowest frequency of each ``(m, n, family)`` of the modes of
    :func:`spectrum`, ``family`` ``'sym'`` or ``'antisym'``: a mode of a real
    Bloch index (``j = 0`` or ``n_s/2``) is symmetric when its fraction
    ``sym`` exceeds 0.5, a mode of a complex index counts for both. Returns
    a dictionary ``(m, n, family) -> spectrum entry``."""
    out = {}
    for d in sorted(spec, key=lambda d: d['f']):
        real = d['j'] == 0 or 2*d['j'] == nstringers
        if real:
            fams = ['sym'] if d['sym'] > 0.5 else ['antisym']
        else:
            fams = ['sym', 'antisym']
        for fam in fams:
            out.setdefault((d['m'], d['n'], fam), d)
    return out


def _assign_ABD(shell, prop):
    for name in ['A11', 'A12', 'A16', 'A22', 'A26', 'A66', 'B11', 'B12',
                 'B16', 'B22', 'B26', 'B66', 'D11', 'D12', 'D16', 'D22',
                 'D26', 'D66']:
        getattr(shell, name)[:] = getattr(prop, name)


def section_properties(rects, G):
    r"""Properties of a stringer section made of thin rectangles

    ``rects`` is a list of ``(y, z, by, bz)``, the centre and the sides of
    each rectangle, `z` from the middle surface of the skin (negative
    inside), `y` along the circumference from the plane of symmetry.
    Returns ``A``, the centroid ``zbar``, the second moments ``I0`` about
    the middle surface of the skin and ``Ic`` about the centroid (bending
    in the plane `(x, r)`), ``Iy`` about the plane of symmetry, the radii of
    gyration `\sqrt{I_0/A}` (``d0``) and `\sqrt{I_c/A}` (``dc``), the polar
    radii of gyration about the centroid (``pc``) and about the point of
    the middle surface of the skin under the stringer (``p0``), and the
    Saint-Venant torsional stiffness ``GJ`` `= G \sum b t^3/3` of the
    rectangles (`t` the smaller side)."""
    A = sum(by*bz for _, _, by, bz in rects)
    zbar = sum(by*bz*z for _, z, by, bz in rects)/A
    I0 = sum(by*bz*z**2 + by*bz**3/12 for _, z, by, bz in rects)
    Iy = sum(by*bz*y**2 + bz*by**3/12 for y, _, by, bz in rects)
    Ic = I0 - A*zbar**2
    J = sum(max(by, bz)*min(by, bz)**3/3 for _, _, by, bz in rects)
    return dict(A=A, zbar=zbar, I0=I0, Ic=Ic, Iy=Iy, d0=np.sqrt(I0/A),
                dc=np.sqrt(Ic/A), pc=np.sqrt((Ic + Iy)/A),
                p0=np.sqrt((I0 + Iy)/A), GJ=G*J)
