import sys
sys.path.append(r'..')

import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from composites import isotropic_plate

from bfsccylinder.sanders import (BFSCCylinderSanders, update_KC0, DOF, DOUBLE,
                                  INT, KC0_SPARSE_SIZE)
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD


def test_point_load(plot=False):
    # number of nodes
    nx = 11 # axial, keep odd if you want a line of nodes exactly in the middle
    ny = 30 # circumferential, keep even if you want a line of nodes in the middle

    # geometry
    L = 0.8
    R = 0.4
    b = 2*pi*R

    # material properties
    E = 70e9
    nu = 0.33
    h = 0.001
    prop = isotropic_plate(thickness=h, E=E, nu=nu)

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
    Kr = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kc = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kv = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        shell = BFSCCylinderSanders(nint)
        shell.c1 = DOF*nid_pos[n1]
        shell.c2 = DOF*nid_pos[n2]
        shell.c3 = DOF*nid_pos[n3]
        shell.c4 = DOF*nid_pos[n4]
        shell.R = R
        shell.lex = L/(nx-1)
        shell.ley = b/ny
        assign_constant_ABD(shell, prop)
        shell.init_k_KC0 = init_k_KC0
        update_KC0(shell, points, weights, Kr, Kc, Kv)
        init_k_KC0 += KC0_SPARSE_SIZE
        elements.append(shell)

    K = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()

    # applying boundary conditions
    bk = np.zeros(K.shape[0], dtype=bool)

    # simply supported
    checkSS = isclose(x, 0) | isclose(x, L)
    bk[0::DOF] = checkSS
    bk[3::DOF] = checkSS
    bk[6::DOF] = checkSS

    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # external force vector for point load at center
    f = np.zeros(K.shape[0])
    fmid = -100.

    # force at center node
    check = np.isclose(x, L/2) & np.isclose(y, b/2)
    f[6::DOF][check] = fmid

    # sub-matrices corresponding to unknown DOFs
    Kuu = K[bu, :][:, bu]
    fu = f[bu]
    assert np.isclose(fu.sum(), fmid)

    # solving
    PREC = np.sqrt(1/Kuu.diagonal()).max()
    uu, info = cg(PREC*Kuu, PREC*fu, tol=1.e-4)
    u = np.zeros(K.shape[0], dtype=float)
    u[bu] = uu

    w = u[6::DOF].reshape(nx, ny)
    print('wmax', w.max())
    print('wmin', w.min())

    assert isclose(w.max(), 0.00023710782832362203)
    assert isclose(w.min(), -0.0007098715355559141)

    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib import cm
        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 300)
        plt.contourf(xmesh, ymesh, w, levels=levels, cmap=cm.jet)
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    test_point_load(plot=True)
