r"""Linear buckling of the laminated cylinder Z33 under a constant axial membrane force, Donnell-type kinematics

The cylinder Z33 of Castro et al. (2014, Composite Structures), Table 4, ``R
= 250`` mm, ``L = 510`` mm, ten plies of 0.125 mm, with the material of
Geier and Singh (1997). `u`, `v`, `w` and `w_{,x}` are constrained at the
nodes of both ends. The geometric stiffness matrix of
``update_KG_constant_stress`` is that of the constant membrane force ``Nxx =
-1`` N/m, without a static analysis, the lowest load multiplier is computed
with ``lobpcg``, and the critical load, the load multiplier times ``Nxx``
times `2 \pi R`, is checked within 1 % against -200488 N, a regression
value: a result of the element of the ``bfsccylinder`` namespace stored in
the test.

Notes
-----
The stacking sequence, from the inner to the outer surface, is that of the
cylinder Z33 in Table 4 of [1]_, and the elastic constants, `E_{11} =
123.55` GPa, `E_{22} = 8.708` GPa, `\nu_{12} = 0.319` and `G_{12} = 5.695`
GPa, are those of the material Geier1997 that Table 4 of [1]_ gives for Z33
(Table 3 of [1]_). They are the ply properties of the test cylinders of
[3]_, Section V.4 (`E_L = 123550` MPa, `E_T = 8707.9` MPa, `\nu_{LT} =
0.319`, `G_{LT} = 5696` MPa), and of Z33 in [4]_, Section 3 (`G = 5695`
MPa, `\nu = 0.32`). The entries of the laminate stiffness matrices `A`,
`B` and `D` of this test agree in magnitude with those printed for Z33 in
Section 3 of [4]_, to their four digits, except `B_{66}` (4152 N here, 4125
N there). The transverse shear moduli do not enter the element, which is
based on the classical laminated plate theory. The ends have `u = v = w =
w_{,x} = 0`, the conditions CC1 of Table 1 of [1]_.

The stored value is a regression value of the element. For comparison, Table
1 of [4]_ gives for Z33 192 kN with built-in edges, deep shell theory and a
homogeneous membrane prebuckling state (BACCUS-C4), and the test results 173
kN and 184 kN; the stored value, 200488 N, is 4.4 % higher than 192 kN.
Table 6 of [2]_ gives 191.67 kN, computed with ABAQUS S8R5 elements and
clamped ends, but with the elastic constants `E_{11} = 142.5` GPa, `E_{22} =
8.7` GPa, `\nu_{12} = 0.28` and `G_{12} = 5.1` GPa of its Table 1; the
stored value is 4.6 % higher. The difference to 192 kN contains the
discretisation error of the 50 nodes around the circumference of this test;
here, as in BACCUS-C4, the geometric stiffness matrix is that of a uniform
membrane force.

References
----------
.. [1] Castro, S. G. P., Mittelstedt, C., Monteiro, F. A. C., Arbelo, M. A.,
   Ziegmann, G., and Degenhardt, R., Linear buckling predictions of
   unstiffened laminated composite cylinders and cones under various loading
   and boundary conditions using semi-analytical models, Composite
   Structures, 118, 303-315, 2014, doi:10.1016/j.compstruct.2014.07.037.
.. [2] Castro, S. G. P., Zimmermann, R., Arbelo, M. A., Khakimova, R.,
   Hilburger, M. W., and Degenhardt, R., Geometric imperfections and
   lower-bound methods used to calculate knock-down factors for axially
   compressed composite cylindrical shells, Thin-Walled Structures, 74,
   118-132, 2014, doi:10.1016/j.tws.2013.08.011.
.. [3] Geier, B., and Singh, G., Some simple solutions for buckling loads of
   thin and moderately thick cylindrical shells and panels made of laminated
   composite material, Aerospace Science and Technology, 1(1), 47-63, 1997,
   doi:10.1016/S1270-9638(97)90023-7.
.. [4] Geier, B., Meyer-Piening, H.-R., and Zimmermann, R., On the influence
   of laminate stacking on buckling of composite cylindrical shells subjected
   to axial compression, Composite Structures, 55(4), 467-474, 2002,
   doi:10.1016/S0263-8223(01)00175-1.
"""
import sys
sys.path.append(r'..')

import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh, cg, lobpcg, LinearOperator, spilu, spsolve
from composites import laminated_plate

from bfsccylinder import (BFSCCylinder, update_KC0, update_KG_constant_stress, DOF, DOUBLE, INT,
KC0_SPARSE_SIZE, KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD


def test_linear_buckling(plot=False):
    # geometry of the cylinder Z33
    L = 0.510 # m
    R = 0.250 # m
    b = 2*pi*R # m

    # number of nodes
    ny = 50 # circumferential
    nx = int(ny*L/b)

    # material properties
    E11 = 123.55e9
    E22 = 8.708e9
    nu12 = 0.319
    G12 = 5.695e9
    plyt = 0.125e-3
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    stack = [0, 0, 19, -19, 37, -37, 45, -45, 51, -51]
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
    bk[0::DOF] = checkBC
    #bk[1::DOF] = checkBC
    #bk[2::DOF] = checkBC
    bk[3::DOF] = checkBC
    #bk[5::DOF] = checkBC
    bk[6::DOF] = checkBC
    bk[7::DOF] = checkBC
    #bk[8::DOF] = checkBC
    #bk[9::DOF] = checkBC
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
        tol = 1e-6
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
    assert np.isclose(Pcr, -200488, rtol=0.01)

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
