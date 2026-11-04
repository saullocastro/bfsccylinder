r"""Linear buckling of a variable-angle filament-wound cylinder under axial compression, Donnell-type kinematics

The cylinder VAT-P-VT of Wang et al. (2020), ``R = 68`` mm and ``L = 300``
mm, with `E_{11} = 90` GPa, `E_{22} = 7` GPa, `\nu_{12} = 0.32` and `G_{12}
= 4.4` GPa. Two plies `[\theta(x), -\theta(x)]` whose angle varies along the
axis: ``theta_VAT_P_x`` interpolates `\theta` with quadratic Lagrange
polynomials through 45.4, 86.5 and 85.8 degrees at `x = 0`, `L/4` and `L/2`,
symmetric about `x = L/2`, and the ply thickness is 0.4 mm divided by the
cosine of `|\theta_{min} - \theta(x)|`, with `\theta_{min}` the smallest of
the three angles. The laminate stiffnesses are computed at each integration
point and set point by point in the arrays ``A11, ..., D66`` of the
elements. `u`, `v`, `w` and `w_{,x}` are constrained at the nodes of both
ends, and the end `x = L` is compressed by a prescribed axial displacement.
The linear static solution gives the geometric stiffness matrix of
``update_KG``, the lowest load multiplier is computed with ``lobpcg``, and
the critical load, the load multiplier times the sum of the reaction forces
at `x = L`, is checked within 1 % against -55671 N, a regression value: a
result of the element of the ``bfsccylinder`` namespace stored in the test.

Notes
-----
The cylinder is the variable-angle, variable-thickness design VAT-P-VT of
[1]_: 300 mm long, 136 mm in diameter, one angle-ply layer `\pm \theta(x)`
made of two plies, with the second-order variation of `\theta(x)` of Eq. (2)
of [1]_ and the control angles `[45.4, 86.5, 85.8]` degrees of Table 4 of
[1]_ (window of 15 %). The ply thickness follows Eq. (3) of [1]_, `h(x) =
h_{tow}/\cos\Delta\theta(x)`, with `h_{tow} = 0.4` mm and `\Delta\theta(x) =
|\theta_{min} - \theta(x)|`; `\theta_{min}` is here the smallest of the
three control angles instead of the minimum of `\theta(x)`, as kept for the
verification against ABAQUS (comment of the code). The material is that of
Table 1 of [1]_; [2]_ uses the same `E_{11}`, `E_{22}`, `\nu_{12}`, `G_{12}`
and tow thickness, 0.4 mm. The boundary conditions are those of the finite
element model of [1]_: all the degrees of freedom are constrained at `x =
0`, and at `x = L` all but the axial displacement, which is the same at
every node of that end.

The stored value is a regression value of the element. For comparison, Table
4 of [1]_ gives for VAT-P-VT the linear buckling load `F_{eig} = 49576` N,
computed with ABAQUS S4R elements (152 elements along the length, 213 around
the circumference); the stored value, 55671 N, is 12.3 % higher. The
difference contains the discretisation error of the 40 nodes around the
circumference of this test and the transverse shear flexibility of the S4R
element, which the element does not have.

References
----------
.. [1] Wang, Z., Almeida Jr., J. H. S., St-Pierre, L., Wang, Z., and
   Castro, S. G. P., Reliability-based buckling optimization with an
   accelerated Kriging metamodel for filament-wound variable angle tow
   composite cylinders, Composite Structures, 254, 112821, 2020,
   doi:10.1016/j.compstruct.2020.112821.
.. [2] Wang, Z., Almeida Jr., J. H. S., Ashok, A., Wang, Z., and Castro, S.
   G. P., Lightweight design of variable-angle filament-wound cylinders
   combining Kriging-based metamodels with particle swarm optimization,
   Structural and Multidisciplinary Optimization, 65(5), 140, 2022,
   doi:10.1007/s00158-022-03227-8.
"""
import time
import sys
sys.path.append(r'..')

import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh, cg, lobpcg, LinearOperator, spilu
from composites import laminated_plate

from bfsccylinder import (BFSCCylinder, update_KC0, update_KG, DOF, DOUBLE, INT,
KC0_SPARSE_SIZE, KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD


def theta_VAT_P_x(x, L, desvar):
    theta_VP_1, theta_VP_2, theta_VP_3 = desvar
    x1 = 0
    x2 = L/4
    x3 = L/2
    x4 = 3*L/4
    x5 = L
    if x <= L/2:
        N1 = (x - x2)*(x - x3)/((x1 - x2)*(x1 - x3))
        N2 = (x - x1)*(x - x3)/((x2 - x1)*(x2 - x3))
        N3L = (x - x1)*(x - x2)/((x3 - x1)*(x3 - x2))
        return N1*theta_VP_1 + N2*theta_VP_2 + N3L*theta_VP_3
    else:
        N3R = (x - x4)*(x - x5)/((x3 - x4)*(x3 - x5))
        N4 = (x - x3)*(x - x5)/((x4 - x3)*(x4 - x5))
        N5 = (x - x3)*(x - x4)/((x5 - x3)*(x5 - x4))
        return N3R*theta_VP_3 + N4*theta_VP_2 + N5*theta_VP_1


def test_linear_buckling(plot=False):
    time0 = time.process_time()
    # geometry of the filament-wound cylinder
    L = 0.3 # m
    R = 0.136/2 # m
    circ = 2*pi*R # m

    # number of nodes
    ny = 40 # circumferential
    nx = int(ny*L/circ)
    if nx % 2 == 0:
        nx += 1
    print('nx, ny', nx, ny)

    # material properties
    E11 = 90e9
    E22 = 7e9
    nu12 = 0.32
    G12 = 4.4e9
    plyt = 0.4e-3

    theta_VP_1 = 45.4
    theta_VP_2 = 86.5
    theta_VP_3 = 85.8
    #NOTE min( theta1, theta2, theta3 ) is not strictly correct
    #     I kept it here for verification purposes against ABAQUS
    #     a better model is to do min( theta(x) )
    theta_min = min([theta_VP_1, theta_VP_2, theta_VP_3])

    nids = 1 + np.arange(nx*(ny+1))
    nids_mesh = nids.reshape(nx, ny+1)
    # closing the cylinder by reassigning last row of node-ids
    nids_mesh[:, -1] = nids_mesh[:, 0]
    nids = np.unique(nids_mesh)
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    xlin = np.linspace(0, L, nx)
    ytmp = np.linspace(0, circ, ny+1)
    ylin = np.linspace(0, circ-(ytmp[-1] - ytmp[-2]), ny)
    xmesh, ymesh = np.meshgrid(xlin, ylin)
    xmesh = xmesh.T
    ymesh = ymesh.T

    # getting nodes
    ncoords = np.vstack((xmesh.flatten(), ymesh.flatten())).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]

    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    nint = 4
    points, weights = get_points_weights(nint=nint)

    num_elements = len(n1s)
    print('num_elements', num_elements)

    elements = []
    N = DOF*nx*ny
    print('N', N)
    init_k_KC0 = 0
    init_k_KG = 0
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        shell = BFSCCylinder(nint)
        shell.n1 = n1
        shell.n2 = n2
        shell.n3 = n3
        shell.n4 = n4
        shell.c1 = DOF*nid_pos[n1]
        shell.c2 = DOF*nid_pos[n2]
        shell.c3 = DOF*nid_pos[n3]
        shell.c4 = DOF*nid_pos[n4]
        shell.R = R
        shell.lex = L/(nx-1)
        shell.ley = circ/ny
        for i in range(nint):
            x1 = ncoords[nid_pos[n1]][0]
            x2 = ncoords[nid_pos[n2]][0]
            xi = points[i]
            xlocal = x1 + (x2 - x1)*(xi + 1)/2
            assert xlocal > x1 and xlocal < x2
            desvar = theta_VP_1, theta_VP_2, theta_VP_3
            theta_local = theta_VAT_P_x(xlocal, L, desvar)
            steering_angle = abs(theta_min - theta_local)
            plyt_local = plyt/np.cos(np.deg2rad(steering_angle))
            prop = laminated_plate(stack=[theta_local, -theta_local],
                    plyt=plyt_local, laminaprop=laminaprop)
            for j in range(nint):
                shell.A11[i, j] = prop.A11
                shell.A12[i, j] = prop.A12
                shell.A16[i, j] = prop.A16
                shell.A22[i, j] = prop.A22
                shell.A26[i, j] = prop.A26
                shell.A66[i, j] = prop.A66
                shell.B11[i, j] = prop.B11
                shell.B12[i, j] = prop.B12
                shell.B16[i, j] = prop.B16
                shell.B22[i, j] = prop.B22
                shell.B26[i, j] = prop.B26
                shell.B66[i, j] = prop.B66
                shell.D11[i, j] = prop.D11
                shell.D12[i, j] = prop.D12
                shell.D16[i, j] = prop.D16
                shell.D22[i, j] = prop.D22
                shell.D26[i, j] = prop.D26
                shell.D66[i, j] = prop.D66
        shell.init_k_KC0 = init_k_KC0
        shell.init_k_KG = init_k_KG
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_KG += KG_SPARSE_SIZE
        elements.append(shell)

    print('elements created')

    Kr = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kc = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kv = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KC0(shell, points, weights, Kr, Kc, Kv)

    KC0 = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()
    print('stiffness matrix OK')

    # applying boundary conditions
    bk = np.zeros(N, dtype=bool)

    # clamped
    checkSS = isclose(x, 0) | isclose(x, L)
    bk[0::DOF] = checkSS
    bk[3::DOF] = checkSS
    bk[6::DOF] = checkSS
    bk[7::DOF] = checkSS
    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # axial compression applied at x=L
    u = np.zeros(N, dtype=DOUBLE)

    compression = -0.0005
    checkTopEdge = isclose(x, L)
    u[0::DOF] += checkTopEdge*compression
    uk = u[bk]

    # sub-matrices corresponding to unknown DOFs
    Kuu = KC0[bu, :][:, bu]
    Kuk = KC0[bu, :][:, bk]
    Kkk = KC0[bk, :][:, bk]

    fu = -Kuk*uk

    Nu = N - bk.sum()

    # solving
    PREC = 1/Kuu.diagonal().max()

    print('starting static analysis')
    uu, info = cg(PREC*Kuu, PREC*fu)
    assert info == 0

    print('static analysis OK')
    u[bu] = uu

    if False:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        plt.gca().set_aspect('equal')
        w = u[6::DOF].reshape(nx, ny)
        plt.contourf(xmesh, ymesh, w, levels=200, cmap=cm.jet)
        plt.colorbar()
        plt.show()
        raise

    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KG(u, shell, points, weights, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]
    print('geometric stiffness matrix OK')

    # lowest load multipliers of the linear buckling problem
    num_eigvals = 2
    #NOTE this works and seems to be the fastest option

    if True:
        print('starting spilu')
        PREC2 = spilu(PREC*Kuu, diag_pivot_thresh=0, drop_tol=1e-8,
                fill_factor=50)
        print('spilu ok')
        def matvec(x):
            return PREC2.solve(x)
        Kuuinv = LinearOperator(matvec=matvec, shape=(Nu, Nu))

        maxiter = 1000
        Xu = np.random.default_rng(0).random((Nu, num_eigvals)) - 0.5
        Xu /= np.linalg.norm(Xu, axis=0)

        #NOTE default tolerance is too large
        tol = 1e-5
        eigvals, eigvecsu, hist = lobpcg(A=PREC*Kuu, B=-PREC*KGuu, X=Xu, M=Kuuinv, largest=False,
                maxiter=maxiter, retResidualNormsHistory=True, tol=tol)
        assert len(hist) <= maxiter, 'lobpcg did not converge'
        load_mult = eigvals
    else:
        eigvals, eigvecsu = eigsh(A=Kuu, k=num_eigvals, which='SM', M=KGuu,
                tol=1e-9, sigma=1., mode='buckling')
        load_mult = -eigvals

    print('linear buckling analysis OK')
    f = np.zeros(N)
    fk = Kuk.T*uu + Kkk*uk
    f[bk] = fk
    Pcr = load_mult[0]*(f[0::DOF][checkTopEdge]).sum()
    print('Pcr =', Pcr)
    assert isclose(Pcr, -55671, rtol=0.01)

    mode = 0
    mode_shape = np.zeros(N, dtype=float)
    mode_shape[bu] = eigvecsu[:, mode]

    if plot:
        w = mode_shape[6::DOF].reshape(nx, ny)
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        plt.gca().set_aspect('equal')
        plt.contourf(xmesh, ymesh, w, levels=200, cmap=cm.jet)
        plt.colorbar()
        plt.show()
    print('elapsed time: %f s' % (time.process_time() - time0))

if __name__ == '__main__':
    test_linear_buckling(plot=False)
