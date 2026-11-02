r"""Linear buckling of the laminated cylinder Z22 under a constant axial membrane force, Sanders-type kinematics

The cylinder Z22 of Castro et al. (2014), Table 4, ``R = 250`` mm, ``L =
510`` mm, six plies of 0.125 mm, with the material of Geier and Singh
(1997). `v` and `w` are constrained at the nodes of both ends and `u` at one
node of the middle section. The geometric stiffness matrix of
``update_KG_constant_stress`` is that of the constant membrane force ``Nxx =
-1`` N/m, without a static analysis, the lowest load multiplier is computed
with ``lobpcg``, and the critical load, the load multiplier times ``Nxx``
times `2 \pi R`, is checked within 1 % against -35858.08028613658 N, a
regression value: a result of the element of ``bfsccylinder.sanders`` stored
in the test.

Notes
-----
The stacking sequence, from the inner to the outer surface, is that of the
cylinder Z22 in Table 4 of [1]_, and the elastic constants are those of the
material Geier1997 in Table 3 of [1]_, from [2]_, except `G_{12}`, 5.595 GPa
here, a typo for the 5.695 GPa of Table 3. The ends have `v = w = 0`, the
conditions SS2 of Table 1 of [1]_, with `u` constrained at one node of the
middle section to remove the rigid-body motions. The stored value is a
regression value of the element. For comparison, Table 6 of [1]_ gives for
Z22, with the conditions SS2 and the classical laminated plate theory (model
CLPT-BC2 of [1]_), 36.35 kN, and the test result 34.4 kN of Geier et al.;
the stored value, 35858 N, is 1.4 % lower than 36.35 kN. Here the geometric
stiffness matrix is that of a uniform membrane force, without the
prebuckling solution of the static analysis.

References
----------
.. [1] Castro, S. G. P., Mittelstedt, C., Monteiro, F. A. C., Arbelo, M. A.,
   Ziegmann, G., and Degenhardt, R., Linear buckling predictions of
   unstiffened laminated composite cylinders and cones under various loading
   and boundary conditions using semi-analytical models, Composite
   Structures, 118, 303-315, 2014, doi:10.1016/j.compstruct.2014.07.037.
.. [2] Geier, B., and Singh, G., Some simple solutions for buckling loads of
   thin and moderately thick cylindrical shells and panels made of laminated
   composite material, Aerospace Science and Technology, 1(1), 47-63, 1997,
   doi:10.1016/S1270-9638(97)90023-7.
"""
import sys
sys.path.append(r'..')

import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh, cg, lobpcg, LinearOperator, spilu, spsolve
from composites import laminated_plate

from bfsccylinder.sanders import (BFSCCylinderSanders, update_KC0,
                                  update_KG_constant_stress, DOF, DOUBLE, INT,
                                  KC0_SPARSE_SIZE, KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD


def test_linear_buckling(plot=False):
    # geometry of the cylinder Z22
    L = 0.510 # m
    R = 0.250 # m
    b = 2*pi*R # m

    # number of nodes
    ny = 60 # circumferential
    nx = int(ny*L/b)
    if nx % 2 == 0:
        nx += 1

    # material properties
    E11 = 123.55e9
    E22 = 8.708e9
    nu12 = 0.319
    G12 = 5.595e9
    plyt = 0.125e-3
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    stack = [+49, -49, +36, -36, 0, 0]
    prop = laminated_plate(stack=stack, plyt=plyt, laminaprop=laminaprop)

    nids = 1 + np.arange(nx*(ny+1))
    nids_mesh = nids.reshape(nx, ny+1)
    # closing the cylinder by reassigning last row of node-ids
    nids_mesh[:, -1] = nids_mesh[:, 0]
    nids = np.unique(nids_mesh)
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    xlin = np.linspace(0, L, nx)
    ytmp = np.linspace(0, b, ny+1)
    ylin = np.linspace(0, b-(ytmp[-1] - ytmp[-2]), ny)
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
        shell.ley = b/ny
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

    Nxx = -1
    Nyy = 0
    Nxy = 0

    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KG_constant_stress(shell, points, weights, KGr, KGc, KGv, Nxx, Nyy, Nxy)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()

    # applying boundary conditions
    bk = np.zeros(N, dtype=bool)

    checkBC = isclose(x, 0) | isclose(x, L)
    #bk[0::DOF] = checkBC
    #bk[1::DOF] = checkBC
    #bk[2::DOF] = checkBC
    bk[3::DOF] = checkBC
    #bk[4::DOF] = checkBC
    #bk[5::DOF] = checkBC
    bk[6::DOF] = checkBC
    #bk[7::DOF] = checkBC
    #bk[8::DOF] = checkBC
    #bk[9::DOF] = checkBC

    checkBC = isclose(x, L/2) & isclose(y, 0)
    bk[0::DOF] = checkBC

    bu = ~bk # same as np.logical_not, defining unknown DOFs

    Kuu = KC0[bu, :][:, bu]
    KGuu = KG[bu, :][:, bu]

    print('structural matrices OK')

    # lowest load multipliers of the linear buckling problem
    num_eigvals = 3
    Nu = N - bk.sum()
    if True:
        #NOTE this works and seems to be the fastest option
        PREC = 1/Kuu.diagonal().max()
        PREC2 = spilu(PREC*Kuu, diag_pivot_thresh=0, drop_tol=1e-8,
                fill_factor=50)
        print('spilu OK')
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
        assert len(hist) <= maxiter, 'did not converge'
        load_mult = eigvals
    else:
        if True:
            #NOTE works, but slower than lobpcg
            eigvals, eigvecsu = eigsh(A=Kuu, k=num_eigvals, which='SM', M=KGuu,
                    tol=1e-9, sigma=1., mode='buckling')
            load_mult = -eigvals
        else:
            #NOTE this is giving close but varying results for each run
            PREC = 1/Kuu.diagonal().max()
            PREC2 = spilu(PREC*Kuu, diag_pivot_thresh=0)
            print('spilu OK')
            def matvec(x):
                return PREC2.solve(x)
            Minv = LinearOperator(matvec=matvec, shape=(Nu, Nu))
            eigvals, eigvecsu = eigsh(A=-PREC*KGuu, k=num_eigvals, which='LM',
                    M=PREC*Kuu, Minv=Minv)
            load_mult = 1./eigvals

    print('linear buckling analysis OK')
    Pcr = load_mult[0]*Nxx*2*np.pi*R
    print('load_mult[0]', load_mult[0])
    print('Pcr top=', Pcr)
    assert np.isclose(Pcr, -35858.08028613658, rtol=0.01)

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

if __name__ == '__main__':
    test_linear_buckling(plot=True)
