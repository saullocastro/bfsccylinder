import time
import sys
sys.path.append(r'..')

import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, spilu, LinearOperator
from scipy.sparse.linalg import cg, spsolve
from composites import laminated_plate

from bfsccylinder import (BFSCCylinder, update_KC0, update_KCNL, update_KG,
        update_fint, DOF, DOUBLE, INT, KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE,
        KG_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD

def test_nonlinear_axial_compression_load_controlled():
    # geometry
    #Arbocz, J., and Starnes, J. H., 2002, “On a High-Fidelity Hierarchical Approach to Buckling Load Calculations,” New Approaches to Structural Mechanics, Shells and Biological Structures, pp. 271–292.
    L = 0.3556 # m
    R = 0.20318603 # m
    ny = 4*16

    load = 50000. # N

    # geometry our FW cylinders
    circ = 2*pi*R # m

    nx = int(1.5*ny*L/circ)
    if (nx % 2) == 0:
        nx += 1
    print('nx, ny', nx, ny)

    # material properties
    E11 = 127.629e9
    E22 = 11.3074e9
    G12 = 6.00257e9
    nu12 = 0.3002
    stack = [45, -45, 0, 90, 90, 0, -45, 45]
    plyt =  0.00101539/len(stack)
    laminaprop = (E11, E22, nu12, G12, G12, G12)
    prop = laminated_plate(stack=stack, plyt=plyt, laminaprop=laminaprop)

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

    #NOTE it didn't improve the performance
    # sorting nodes aiming more sparsity
    #asort = np.argsort(x**2 + y**2)
    #for i, nid in enumerate(nids):
        #nid_pos[nid] = asort[i]

    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    # number of integration points within element (along xi and eta)
    #TODO investigate different number of integration points
    nint = 4
    points, weights = get_points_weights(nint=nint)

    num_elements = len(n1s)
    print('num_elements', num_elements)

    elements = []
    N = DOF*nx*ny
    KCNLr = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
    KCNLc = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
    KCNLv = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    init_k_KCNL = 0
    init_k_KG = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        shell = BFSCCylinder(nint)
        shell.c1 = DOF*nid_pos[n1]
        shell.c2 = DOF*nid_pos[n2]
        shell.c3 = DOF*nid_pos[n3]
        shell.c4 = DOF*nid_pos[n4]
        shell.R = R
        shell.lex = L/(nx-1)
        shell.ley = circ/ny
        assign_constant_ABD(shell, prop)
        shell.init_k_KC0 = init_k_KC0
        shell.init_k_KCNL = init_k_KCNL
        shell.init_k_KG = init_k_KG
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_KCNL += KCNL_SPARSE_SIZE
        init_k_KG += KG_SPARSE_SIZE
        elements.append(shell)

    KC0r = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    for shell in elements:
        update_KC0(shell, points, weights, KC0r, KC0c, KC0v)
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('KC0 ok')

    # applying SS3 boundary conditions
    bk = np.zeros(N, dtype=bool)

    checkSS = isclose(x, 0) | isclose(x, L)
    bk[3::DOF] = checkSS
    bk[6::DOF] = checkSS
    check = isclose(x, L/2.)
    bk[0::DOF] = check
    bu = ~bk # same as np.logical_not, defining unknown DOFs
    u0 = np.zeros(N, dtype=DOUBLE)
    uk = u0[bk]

    # axially compressive load applied at x=0 and x=L
    checkTopEdge = isclose(x, L)
    checkBottomEdge = isclose(x, 0)
    fext = np.zeros(N)
    fext[0::DOF][checkBottomEdge] = +load/ny
    assert np.isclose(fext.sum(), load)
    fext[0::DOF][checkTopEdge] = -load/ny
    assert np.isclose(fext.sum(), 0)

    # sub-matrices corresponding to unknown DOFs
    KC0uu = KC0[bu, :][:, bu]
    KC0uk = KC0[bu, :][:, bk]

    def calc_KT(u, KCNLv, KGv):
        KCNLv *= 0
        KGv *= 0
        for shell in elements:
            update_KCNL(u, shell, points, weights, KCNLr, KCNLc, KCNLv)
            update_KG(u, shell, points, weights, KGr, KGc, KGv)
        KCNL = coo_matrix((KCNLv, (KCNLr, KCNLc)), shape=(N, N)).tocsc()
        KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
        return KC0 + KCNL + KG

    def calc_fint(u, fint):
        fint *= 0
        for shell in elements:
            update_fint(u, shell, points, weights, fint)
        return fint

    # solving using Modified Newton-Raphson method
    def scaling(vec, D):
        """
            A. Peano and R. Riccioni, Automated discretisatton error
            control in finite element analysis. In Finite Elements m
            the Commercial Enviror&ent (Editei by J. 26.  Robinson),
            pp. 368-387. Robinson & Assoc., Verwood.  England (1978)
        """
        return np.sqrt((vec*np.abs(1/D))@vec)

    #initial
    u0 = np.zeros(N) # any initial condition here

    u0[bu] = spsolve(KC0uu, fext[bu])
    #PREC = 1/KC0uu.diagonal().max()
    #u0[bu], info = cg(PREC*KC0uu, PREC*fext[bu], atol=1e-9)
    #if info != 0:
        #print('#   failed with cg()')
        #print('#   trying spsolve()')
        #uu = spsolve(KC0uu, fext[bu])
    count = 0
    fint = np.zeros(N)
    fint = calc_fint(u0, fint)
    Ri = fint - fext
    du = np.zeros(N)
    ui = u0.copy()
    epsilon = 1.e-4
    KT = calc_KT(u0, KCNLv, KGv)
    KTuu = KT[bu, :][:, bu]
    D = KC0uu.diagonal() # at beginning of load increment
    while True:
        print('count', count)
        duu = spsolve(KTuu, -Ri[bu])
        #PREC = 1/KTuu.diagonal().max()
        #duu, info = cg(PREC*KTuu, -PREC*Ri[bu], atol=1e-9)
        #if info != 0:
            #print('#   failed with cg()')
            #print('#   trying spsolve()')
            #duu = spsolve(KTuu, -Ri[bu])
        du[bu] = duu
        u = ui + du
        fint = calc_fint(u, fint)
        Ri = fint - fext
        crisfield_test = scaling(Ri[bu], D)/max(scaling(fext[bu], D), scaling(fint[bu], D))
        print('    crisfield_test', crisfield_test)
        if crisfield_test < epsilon:
            print('    converged')
            break
        count += 1
        KT = calc_KT(u, KCNLv, KGv)
        KTuu = KT[bu, :][:, bu]
        ui = u.copy()
        if count > 6:
            raise RuntimeError('Not converged!')

if __name__ == '__main__':
    test_nonlinear_axial_compression_load_controlled()
