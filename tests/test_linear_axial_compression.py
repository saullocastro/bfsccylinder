import sys
sys.path.append(r'..')

import numpy as np
from numpy import isclose, pi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from composites import isotropic_plate

from bfsccylinder import BFSCCylinder, update_KC0, DOF, DOUBLE, INT, KC0_SPARSE_SIZE
from bfsccylinder.quadrature import get_points_weights
from bfsccylinder.utils import assign_constant_ABD

def test_linear_axial_compression(plot=False):
    # number of nodes
    nx = 10 # axial, keep odd if you want a line of nodes exactly in the middle
    ny = 18 # circumferential

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

    num_elements = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        num_elements += 1

    elements = []
    N = DOF*nx*ny
    Kr = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kc = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kv = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        shell = BFSCCylinder(nint)
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

    KC0 = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()

    # applying boundary conditions
    bk = np.zeros(N, dtype=bool)

    # simply supports
    checkSS = isclose(x, 0) | isclose(x, L)
    bk[0::DOF] = checkSS
    bk[3::DOF] = checkSS
    bk[6::DOF] = checkSS

    # axial compression applied at x=L
    u = np.zeros(N, dtype=float)

    compression = -0.001
    checkTopEdge = isclose(x, L)
    u[0::DOF] += checkTopEdge*compression
    uk = u[bk]

    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # sub-matrices corresponding to unknown DOFs
    Kuu = KC0[bu, :][:, bu]
    Kuk = KC0[bu, :][:, bk]

    fu = -Kuk*uk

    # solving
    PREC = 1/Kuu.diagonal().max()
    uu, info = cg(PREC*Kuu, PREC*fu, atol='legacy')
    assert info == 0
    u[bu] = uu

    w = u[6::DOF].reshape(nx, ny)
    print('wmax', w.max())
    print('wmin', w.min())
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
    test_linear_axial_compression(plot=True)
