r"""Linear buckling under torsion of two stiffened cylinders of Moore and Wescoat (1944), Sanders-type kinematics

Specimen 18 (four 1/2 x 1/2 in rings inside the skin) and specimen 19
(the same rings and eight channel stringers outside the skin) of Moore and
Wescoat (1944), NACA ARR 4E31, Tables I and II: 24S-T, `R` = 15.04 in, `t`
= 0.020 in, length 35.625 in between the end bulkheads. The skin is meshed
with :class:`bfsccylinder.sanders.BFSCCylinderSanders` elements over a
periodic sector, with the Bloch condition `\boldsymbol{u}(\theta = \phi) =
e^{i j \phi} \boldsymbol{u}(\theta = 0)` imposed through a constraint basis,
hinged ends (`u`, `v`, `w` and their derivatives along the edge
constrained) and the geometric stiffness matrix of
``update_KG_constant_stress`` for a constant membrane shear force ``Nxy``.
A ring is a band of skin elements whose laminate adds the ring bar as a
layer on the inner face; in specimen 19 each stringer is a band of skin
elements whose laminate adds a layer with the axial and bending stiffnesses
of the stringer section, without plate elements. The lowest critical shear
stress over the Bloch indices is checked against the converged value of
the same models computed with finer meshes; for specimen 18 the multipliers
of positive and negative torque are also checked to be equal.

Notes
-----
Under a torque `T` the membrane state of the perfect cylinder is the
uniform shear `N_{xy} = T/(2 \pi R^2)` of the skin, the rings and the
stringers unloaded, so that the geometric stiffness matrix of a constant
`N_{xy}` is exact, and the reduced problem `(\boldsymbol{K} + \lambda
\boldsymbol{K}_G) \boldsymbol{x} = 0` of each Bloch index is Hermitian. The
lowest positive `\lambda` is the critical `N_{xy}` for `N_{xy} = 1` lb/in,
and `\tau_{cr} = \lambda/t`. For an isotropic shell symmetric about its
middle section the multipliers come in pairs `\pm\lambda`.

Specimen 18 is the slice of angle `2\pi/180` of the axisymmetric cylinder
with two elements across, the Bloch index being the circumferential wave
number `n`; the rings are at 4 5/16, 17 13/16 and 31 5/16 in from an end
(Fig. 5 of [1]_). The ring bands are 1/2 in wide with two elements across;
the ring layer, `-t/2 - 1/2 \le z \le -t/2`, has `\nu = 0` so that the band
has the axial and bending stiffnesses of the bar, with its eccentricity.
Elements of at most 0.35 in along `x` between the ring bands.

Specimen 19 is one stringer pitch, `\phi = 2\pi/8`, Bloch indices `j` = 0
to 4, with the same rings. The channel of Fig. 30 of [1]_ (0.032 in, `A`
= 0.072 in², height 0.64 in, base 1/2 in) is idealised with flat lips of
0.283 in, which keep its area, and represented by a band of the skin of
0.468 in (between the middle surfaces of the webs) whose laminate adds a
layer with `Q_{11} = E_s` only, of thickness `d_s = \sqrt{12 I_{XX}/A}`,
centred at the centroid of the section, `E_s b d_s = E A`: the axial
stiffness, eccentricity and bending stiffness normal to the skin of the
stringer. Elements of at most 0.5 in along `x`, nine elements between the
sector edge and the band, two across it.

The stored values and the tolerances come from the mesh convergence of
these models: specimen 18, 2252.3 psi at the mesh of this test, 2251.0 psi
with elements of 0.25 in, 240 slices and four elements across a ring band
(0.06 % lower); specimen 19 (stringer bands), 2602.8 psi at the mesh of
this test, on which the model with the channel of plate elements changes
by 0.26 % at the next refinement (elements of 0.35 by 0.45 in). The
tolerance, 1 %, is the convergence criterion of the meshes. The model with
the channel of plate elements gives 2619 psi for specimen 19 (converged),
0.6 % above the stringer bands. For
comparison, Table I of [1]_ gives for specimen 18 the average observed
shear-buckling stress 1710 psi and Donnell's theoretical 2210 psi (hinged
edges, bay length 13.50 in), and Table II for the lowest panel of specimen
19 (13.50 x 11.81 in) 2110 psi observed and the computed 1620 (ANC-5), 1810
(Kromm), 2320 (Ebner) and 2210 psi (Donnell).

References
----------
.. [1] Moore, R. L., and Wescoat, C., Torsion tests of stiffened circular
   cylinders, NACA ARR 4E31 (Wartime Report W-89), 1944.
"""
import numpy as np
from numpy import pi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, eigs, splu

from bfsccylinder.sanders import (BFSCCylinderSanders, update_KC0,
                                  update_KG_constant_stress, DOF, INT,
                                  KC0_SPARSE_SIZE, KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD

E = 10.5e6  # psi
NU = 1/3
R = 30.08/2  # in
L = 35.625  # in
U, U_X, U_Y, V, V_X, V_Y, W, W_X, W_Y, W_XY = range(10)


class Laminate:
    """A, B, D of isotropic layers ``(z0, z1, E, nu, G)``"""
    def __init__(self, layers):
        for k in ['11', '12', '16', '22', '26', '66']:
            for m in 'ABD':
                setattr(self, m + k, 0.)
        for z0, z1, Ey, nu, G in layers:
            Q11 = Ey/(1 - nu**2)
            Q = {'11': Q11, '12': nu*Q11, '22': Q11, '66': G}
            for k, q in Q.items():
                self.add(k, q, z0, z1)

    def add(self, k, q, z0, z1):
        self.__dict__['A' + k] += q*(z1 - z0)
        self.__dict__['B' + k] += q*(z1**2 - z0**2)/2
        self.__dict__['D' + k] += q*(z1**3 - z0**3)/3


def laminate(t, ring, stringer):
    G = E/(2*(1 + NU))
    layers = [(-t/2, t/2, E, NU, G)]
    if ring:
        layers.append((-t/2 - 0.5, -t/2, E, 0., G))
    lam = Laminate(layers)
    if stringer:
        # idealised channel: base 0.468, webs 0.608, lips 0.283, t = 0.032
        tc, b, H, bl = 0.032, 0.468, 0.608, 0.283
        segs = [(b, tc/2, 'h'), (H, tc/2 + H/2, 'v'), (H, tc/2 + H/2, 'v'),
                (bl, tc/2 + H, 'h'), (bl, tc/2 + H, 'h')]
        A = sum(ln*tc for ln, _, _ in segs)
        zc = sum(ln*tc*z for ln, z, _ in segs)/A
        Ixx = sum((ln*tc**3/12 if o == 'h' else tc*ln**3/12)
                  + ln*tc*(z - zc)**2 for ln, z, o in segs)
        ds = np.sqrt(12*Ixx/A)
        Es = E*A/(b*ds)
        z0, z1 = t/2 + zc - ds/2, t/2 + zc + ds/2
        lam.add('11', Es, z0, z1)
    return lam


def critical_shear_stress(t, rings, nsectors, js, dx, nbay, stringer,
                          both=False):
    """Lowest positive critical shear stress (psi) over the Bloch indices
    ``js`` of the sector ``2 pi/nsectors``, hinged ends; with ``both`` also
    the largest negative one of the index of the lowest"""
    # nodes along x: two elements across each ring band of 0.5 in
    brk = [0.]
    for s in rings:
        brk += [s - 0.25, s + 0.25]
    brk.append(L)
    x = [0.]
    for k in range(len(brk) - 1):
        a, b = brk[k], brk[k + 1]
        n = 2 if k % 2 == 1 else int(np.ceil((b - a)/dx - 1e-9))
        x += list(a + (b - a)*np.arange(1, n + 1)/n)
    x = np.array(x)
    band = np.zeros(x.size - 1, dtype=bool)
    for k in range(len(rings)):
        band[(brk[2*k + 1] - 1e-9 < x[:-1]) & (x[:-1] < brk[2*k + 2])] = True
    phi = 2*pi/nsectors
    if stringer:
        beta = 0.468/(2*R)
        half = np.full(nbay, (phi/2 - beta)/nbay)
        dth = np.concatenate([half, [beta, beta], half])
        sband = [nbay, nbay + 1]
    else:
        dth = np.full(2*nbay, phi/(2*nbay))
        sband = []
    nx, ncol = x.size, dth.size + 1
    nodes = np.arange(nx*ncol).reshape(nx, ncol)
    N = DOF*nodes.size
    points, weights = get_points_weights(nint=4)
    lams = {(r, s): laminate(t, r, s) for r in (0, 1) for s in (0, 1)}
    elements = []
    for i in range(nx - 1):
        for j in range(ncol - 1):
            shell = BFSCCylinderSanders(4)
            shell.n1, shell.n2 = nodes[i, j], nodes[i + 1, j]
            shell.n3, shell.n4 = nodes[i + 1, j + 1], nodes[i, j + 1]
            shell.c1, shell.c2 = DOF*shell.n1, DOF*shell.n2
            shell.c3, shell.c4 = DOF*shell.n3, DOF*shell.n4
            shell.R = R
            shell.lex = x[i + 1] - x[i]
            shell.ley = R*dth[j]
            assign_constant_ABD(shell, lams[int(band[i]), int(j in sband)])
            elements.append(shell)
    ne = len(elements)
    rK = np.zeros(KC0_SPARSE_SIZE*ne, dtype=INT)
    cK = np.zeros_like(rK)
    vK = np.zeros(rK.size)
    rG = np.zeros(KG_SPARSE_SIZE*ne, dtype=INT)
    cG = np.zeros_like(rG)
    vG = np.zeros(rG.size)
    for e, shell in enumerate(elements):
        shell.init_k_KC0 = e*KC0_SPARSE_SIZE
        shell.init_k_KG = e*KG_SPARSE_SIZE
        update_KC0(shell, points, weights, rK, cK, vK)
        update_KG_constant_stress(shell, points, weights, rG, cG, vG,
                                  0., 0., 1.)
    KC0 = coo_matrix((vK, (rK, cK)), shape=(N, N)).tocsr()
    KG = coo_matrix((vG, (rG, cG)), shape=(N, N)).tocsr()
    # Bloch condition: last column of nodes = phase times the first
    dep = np.zeros(N, dtype=bool)
    dep[((DOF*nodes[:, -1])[:, None] + np.arange(DOF)).ravel()] = True
    Nr = int((~dep).sum())
    red = -np.ones(N, dtype=int)
    red[~dep] = np.arange(Nr)
    # hinged ends
    bk = np.zeros(Nr, dtype=bool)
    for i in [0, nx - 1]:
        for nd in nodes[i, :-1]:
            bk[red[DOF*nd + np.array([U, U_Y, V, V_Y, W, W_Y])]] = True
    bu = ~bk
    best = None
    for jb in js:
        phase = np.exp(1j*jb*phi)
        rows = list(np.flatnonzero(~dep)) + list(np.flatnonzero(dep))
        cols = list(range(Nr)) + list(red[np.flatnonzero(dep)
                                          - DOF*(ncol - 1)])
        vals = [1.]*Nr + [phase]*int(dep.sum())
        Z = coo_matrix((vals, (rows, cols)), shape=(N, Nr)).tocsr()
        K = (Z.conj().T @ KC0 @ Z).tocsr()[bu][:, bu].tocsc()
        G = -(Z.conj().T @ KG @ Z).tocsr()[bu][:, bu]
        lu = splu(K)
        op = LinearOperator(K.shape,
                            matvec=lambda v, lu=lu, G=G: lu.solve(G @ v),
                            dtype=complex)
        mu = eigs(op, k=2, which='LR', ncv=24, tol=1e-10,
                  return_eigenvectors=False).real
        tau = 1/mu.max()/t
        if best is None or tau < best[0]:
            best = (tau, op)
    if not both:
        return best[0]
    mun = eigs(best[1], k=2, which='SR', ncv=24, tol=1e-10,
               return_eigenvectors=False).real
    return best[0], 1/mun.min()/t


def test_ring_stiffened_specimen18():
    rings = np.cumsum([4 + 5/16, 13.5, 13.5])
    tau, tau_neg = critical_shear_stress(0.020, rings, 180, range(18, 23),
                                         dx=0.35, nbay=1, stringer=False,
                                         both=True)
    print('specimen 18: tau_cr = %.1f psi, negative torque %.1f psi'
          % (tau, tau_neg))
    assert np.isclose(tau, 2252.3, rtol=0.01)
    assert np.isclose(-tau_neg, tau, rtol=1e-6)


def test_stringer_stiffened_specimen19():
    rings = np.cumsum([4 + 5/16, 13.5, 13.5])
    tau = critical_shear_stress(0.020, rings, 8, range(5), dx=0.5, nbay=9,
                                stringer=True)
    print('specimen 19, stringers as skin bands: tau_cr = %.1f psi' % tau)
    assert np.isclose(tau, 2602.8, rtol=0.01)


if __name__ == '__main__':
    test_ring_stiffened_specimen18()
    test_stringer_stiffened_specimen19()
