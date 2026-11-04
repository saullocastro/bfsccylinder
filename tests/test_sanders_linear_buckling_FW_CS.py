r"""Linear buckling of a filament-wound cylinder under axial compression, Sanders-type kinematics

The cylinder CS of Wang et al. (2020), ``R = 68`` mm and ``L = 300`` mm,
with `E_{11} = 90` GPa, `E_{22} = 7` GPa, `\nu_{12} = 0.32` and `G_{12} =
4.4` GPa. Two plies `[\theta, -\theta]`, `\theta = 57.9` degrees, of 0.4 mm,
the same laminate at every integration point. `u`, `v`, `w` and `w_{,x}` are
constrained at the nodes of both ends, and the end `x = L` is compressed by
a prescribed axial displacement. The linear static solution gives the
geometric stiffness matrix of ``update_KG``, the lowest load multiplier is
computed with ``lobpcg``, and the critical load, the load multiplier times
the sum of the reaction forces at `x = L`, is checked within 1 % against
-42719.877887017145 N, a regression value: a result of the element of
``bfsccylinder.sanders`` stored in the test.

Notes
-----
The cylinder is the constant-stiffness design CS of [1]_: 300 mm long, 136
mm in diameter, one angle-ply layer `\pm 57.9` degrees made of two plies of
0.4 mm (Table 4 of [1]_; the text of [1]_ gives 57.8 degrees). The material
is that of Table 1 of [1]_; [2]_ uses the same `E_{11}`, `E_{22}`,
`\nu_{12}`, `G_{12}` and tow thickness, 0.4 mm. The boundary conditions are
those of the finite element model of [1]_: all the degrees of freedom are
constrained at `x = 0`, and at `x = L` all but the axial displacement, which
is the same at every node of that end.

The stored value is a regression value of the element. For comparison, Table
4 of [1]_ gives for CS the linear buckling load `F_{eig} = 40607` N,
computed with ABAQUS S4R elements (152 elements along the length, 213 around
the circumference); the stored value, 42720 N, is 5.2 % higher. The
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
from scipy.sparse.linalg import eigsh, cg, lobpcg, LinearOperator, spilu, splu
from composites import laminated_plate

from bfsccylinder.sanders import (BFSCCylinderSanders, update_KC0, update_KG,
                                  DOF, DOUBLE, INT, KC0_SPARSE_SIZE,
                                  KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD


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

    theta = 57.9

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
    prop = laminated_plate(stack=[theta, -theta], plyt=plyt, laminaprop=laminaprop)
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        shell = BFSCCylinderSanders(nint)
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
        assign_constant_ABD(shell, prop)
        shell.init_k_KC0 = init_k_KC0
        shell.init_k_KG = init_k_KG
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_KG += KG_SPARSE_SIZE
        elements.append(shell)

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

    # solving
    PREC = 1/Kuu.diagonal().max()
    uu, info = cg(PREC*Kuu, PREC*fu)
    assert info == 0

    print('static analysis OK')
    u[bu] = uu

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
    Nu = N - bk.sum()
    if True:
        #NOTE this works and seems to be the fastest option
        if True:
            #TODO use PREC2 to accelerate cg
            PREC2 = spilu(PREC*Kuu, diag_pivot_thresh=0, drop_tol=1e-8,
                    fill_factor=50)
            print('spilu OK')
        else:
            PREC2 = splu(PREC*Kuu)
            print('splu OK')
        def matvec(x):
            return PREC2.solve(x)
        Ainv = LinearOperator(matvec=matvec, shape=(Nu, Nu))
        maxiter = 1000
        X = np.random.default_rng(0).random((Nu, num_eigvals)) - 0.5
        X /= np.linalg.norm(X, axis=0)
        #NOTE default tolerance is too large
        tol = 1e-5
        eigvals, eigvecsu, hist = lobpcg(A=PREC*Kuu, B=-PREC*KGuu, X=X, M=Ainv, largest=False,
                maxiter=maxiter, retResidualNormsHistory=True, tol=tol)
        assert len(hist) <= maxiter, 'lobpcg did not converge'
        load_mult = eigvals
    else:
        if True:
            #NOTE works, but slower than lobpcg
            eigvals, eigvecsu = eigsh(A=Kuu, k=num_eigvals, which='SM', M=KGuu,
                    tol=1e-7, sigma=1., mode='buckling')
            load_mult = -eigvals
        else:
            #NOTE this is giving close but varying results for each run
            PREC2 = spilu(PREC*Kuu, diag_pivot_thresh=0)
            print('spilu OK')
            def matvec(x):
                return PREC2.solve(x)
            Minv = LinearOperator(matvec=matvec, shape=(Nu, Nu))
            eigvals, eigvecsu = eigsh(A=-PREC*KGuu, k=num_eigvals, which='LM',
                    M=PREC*Kuu, Minv=Minv)
            load_mult = 1./eigvals

    print('linear buckling analysis OK')
    f = np.zeros(N)
    fk = Kuk.T*uu + Kkk*uk
    f[bk] = fk
    Pcr = load_mult[0]*(f[0::DOF][checkTopEdge]).sum()
    print('Pcr =', Pcr)
    assert isclose(Pcr, -42719.877887017145, rtol=0.01)

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
    test_linear_buckling(plot=True)
