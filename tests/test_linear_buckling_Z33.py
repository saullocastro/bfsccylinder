import sys
sys.path.append(r'..')

import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh, cg, lobpcg, LinearOperator, spilu, spsolve
from composites import laminated_plate

from bfsccylinder import (BFSCCylinder, update_KC0, update_KG, DOF, DOUBLE, INT,
KC0_SPARSE_SIZE, KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD


def test_linear_buckling(plot=False):
    # geometry Z33 Castro 2014
    L = 0.510 # m
    R = 0.250 # m
    b = 2*pi*R # m

    # number of nodes
    ny = 50 # circumferential
    nx = int(ny*L/b)
    if nx % 2 == 0:
        nx += 1

    # material properties Geier 1997
    E11 = 145.5e9
    E22 = 8.7e9
    nu12 = 0.28
    G12 = 5.1e9
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
    print('stiffness matrix OK')

    # applying boundary conditions
    bk = np.zeros(N, dtype=bool)

    #SS2 boundary conditions (Castro 2014)
    checkBC = isclose(x, 0) | isclose(x, L)
    #bk[0::DOF] = checkBC
    #bk[1::DOF] = checkBC
    #bk[2::DOF] = checkBC
    bk[3::DOF] = checkBC
    #bk[4::DOF] = checkBC
    #bk[5::DOF] = checkBC
    bk[6::DOF] = checkBC
    bk[7::DOF] = checkBC
    #bk[8::DOF] = checkBC
    #bk[9::DOF] = checkBC

    checkBC = isclose(x, L/2) & isclose(y, 0)
    bk[0::DOF] = checkBC

    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # force-controlled
    applied_force = 1000.
    force_nodes = applied_force/ny
    f = np.zeros(N)
    check = isclose(x, 0)
    f[0::DOF][check] = force_nodes
    check = isclose(x, L)
    f[0::DOF][check] = -force_nodes

    # sub-matrices corresponding to unknown DOFs
    Kuu = KC0[bu, :][:, bu]

    fu = f[bu]

    # solving
    #PREC = 1/Kuu.diagonal().max()
    #uu, info = cg(PREC*Kuu, PREC*fu)
    #assert info == 0
    uu = spsolve(Kuu, fu)

    print('static analysis OK')
    u = np.zeros(N, dtype=DOUBLE)
    u[bu] = uu

    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KG(u, shell, points, weights, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]
    print('geometric stiffness matrix OK')

    if False:
        # plotting stress
        xplot = []
        yplot = []
        stress = []
        for shell in elements:
            x1, y1 = ncoords[nid_pos[shell.n1]]
            x2, y2 = ncoords[nid_pos[shell.n2]]
            x3, y3 = ncoords[nid_pos[shell.n3]]
            x4, y4 = ncoords[nid_pos[shell.n4]]
            if y3 < y2:
                y3 += b
            if y4 < y2:
                y4 += b
            x = (x1 + x2 + x3 + x4)/4
            y = (y1 + y2 + y3 + y4)/4
            xplot.append(x)
            yplot.append(y)
            shell.update_Bm(xi=0, eta=0)
            shell.update_Bb(xi=0, eta=0)
            shell.update_Nu(xi=0, eta=0)
            shell.update_Nu_x(xi=0, eta=0)
            shell.update_Nv(xi=0, eta=0)
            shell.update_Nw(xi=0, eta=0)
            u = np.asarray(shell.u)
            Nm = lam.A @ shell.Bm @ u + lam.B @ shell.Bb @ u
            Nb = lam.B @ shell.Bm @ u + lam.D @ shell.Bb @ u

            displu = shell.Nu @ u
            displux = shell.Nu_x @ u
            displv = shell.Nv @ u
            displw = shell.Nw @ u
            stress.append(Nb[0])

        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        xplot = np.asarray(xplot).reshape(nx-1, ny)
        yplot = np.asarray(yplot).reshape(nx-1, ny)
        stress = np.asarray(stress).reshape(nx-1, ny)
        plt.contourf(xplot, yplot, stress, levels=10, cmap=cm.jet)
        plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.show()
        raise

    # A * x[i] = lambda[i] * M * x[i]
    num_eigvals = 3
    Nu = N - bk.sum()
    if True:
        #NOTE this works and seems to be the fastest option
        PREC = 1.#1/Kuu.diagonal().max()
        PREC2 = spilu(PREC*Kuu, diag_pivot_thresh=0, drop_tol=1e-8,
                fill_factor=50)
        print('spilu OK')
        def matvec(x):
            return PREC2.solve(x)
        Ainv = LinearOperator(matvec=matvec, shape=(Nu, Nu))
        maxiter = 1000
        X = np.random.rand(Nu, num_eigvals) - 0.5
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
            PREC2 = spilu(PREC*Kuu, diag_pivot_thresh=0, drop_tol=1e-8,
                    fill_factor=50)
            print('spilu OK')
            def matvec(x):
                return PREC2.solve(x)
            Minv = LinearOperator(matvec=matvec, shape=(Nu, Nu))
            eigvals, eigvecsu = eigsh(A=-PREC*KGuu, k=num_eigvals, which='LM',
                    M=PREC*Kuu, Minv=Minv)
            load_mult = 1./eigvals

    print('linear buckling analysis OK')
    Pcr = load_mult[0]*applied_force
    print('Pcr top=', Pcr)
    assert isclose(Pcr, 201424, rtol=0.01)

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
