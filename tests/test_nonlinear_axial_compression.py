"""Nonlinear axial compression of a cylinder, load controlled, Donnell kinematics

One load step of a full Newton-Raphson solve, with structsolve.Analysis and
the callables of ``Callables``, checking that it converges.

What this test does NOT cover: the state it reaches is mild. At 50 kN the
largest radial displacement is 0.05 of the shell thickness, and ||KCNL||
and ||KG|| are 9.7e-4 and 5.3e-4 of ||KC0||. An inconsistent tangent
converges here too, so this is not a guard on KC0 + KCNL + KG being the
Jacobian of fint; tests/test_tangent_consistency.py is. Nor does it assert
anything about the solution, so it would not notice a wrong fint either.

"""
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
from structsolve import Analysis


class Callables(object):
    """The callables of structsolve.Analysis for the ``elements`` of a model
    under the external forces ``fext``.

    They loop over the elements with the update functions and assemble, and
    work on the unknown degrees of freedom ``bu`` only, the partition of the
    free dofs of structsolve: the solution vector of the solvers is ``c =
    u[bu]``, and the force vectors and matrices are restricted to ``bu``.
    The prescribed degrees of freedom are zero.
    """
    def __init__(self, elements, points, weights, KC0, bu, fext):
        self.elements = elements
        self.points = points
        self.weights = weights
        self.N = KC0.shape[0]
        self.KC0 = KC0
        self.KC0uu = KC0[bu, :][:, bu]
        self.bu = bu
        self.fext = fext
        num_elements = len(elements)
        self.KCNLr = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        self.KCNLc = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        self.KCNLv = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        self.KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
        self.KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
        self.KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)

    def expand(self, c):
        """The displacements of the model from those of the unknown dofs."""
        u = np.zeros(self.N, dtype=DOUBLE)
        u[self.bu] = c
        return u

    def calc_fext(self, inc=1., silent=False):
        return inc*self.fext[self.bu]

    def calc_fint(self, c, silent=False):
        fint = np.zeros(self.N, dtype=DOUBLE)
        u = self.expand(c)
        for shell in self.elements:
            update_fint(u, shell, self.points, self.weights, fint)
        return fint[self.bu]

    def calc_kC(self, c=None, NLgeom=False, silent=False):
        if not NLgeom:
            return self.KC0uu
        # the generated code accumulates with "+=" over the integration
        # points, so the value arrays must be zeroed on every call. The row
        # and column arrays are rewritten identically each time, and could
        # be filled once for a fixed mesh
        self.KCNLv *= 0
        u = self.expand(c)
        for shell in self.elements:
            update_KCNL(u, shell, self.points, self.weights, self.KCNLr,
                        self.KCNLc, self.KCNLv)
        KCNL = coo_matrix((self.KCNLv, (self.KCNLr, self.KCNLc)),
                          shape=(self.N, self.N)).tocsc()
        # KCNL also carries the geometric term of the nonlinear membrane
        # stress, so that KG stays homogeneous of degree one in u and keeps
        # its linear buckling meaning. KC0 + KCNL + KG is d(fint)/du
        return (self.KC0 + KCNL)[self.bu, :][:, self.bu]

    def calc_kG(self, c=None, NLgeom=False, silent=False):
        self.KGv *= 0
        u = self.expand(c)
        for shell in self.elements:
            update_KG(u, shell, self.points, self.weights, self.KGr,
                      self.KGc, self.KGv)
        KG = coo_matrix((self.KGv, (self.KGr, self.KGc)),
                        shape=(self.N, self.N)).tocsc()
        return KG[self.bu, :][:, self.bu]


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
    #NOTE every prescribed DOF is zero here, which the expansion u[bu] = c of
    #     Callables assumes. A non-zero prescribed displacement would need
    #     the load factor dependent load vector of structsolve, calc_fext
    #     declaring lbd

    # axially compressive load applied at x=0 and x=L
    checkTopEdge = isclose(x, L)
    checkBottomEdge = isclose(x, 0)
    fext = np.zeros(N)
    fext[0::DOF][checkBottomEdge] = +load/ny
    assert np.isclose(fext.sum(), load)
    fext[0::DOF][checkTopEdge] = -load/ny
    assert np.isclose(fext.sum(), 0)

    # solving using the full Newton-Raphson method of structsolve, in one
    # load step: the tangent stiffness matrix calc_kC + calc_kG is computed
    # again at every iteration, not held fixed as a modified Newton-Raphson
    # would. That is what buys quadratic convergence, now that KC0 + KCNL +
    # KG is the exact Jacobian of fint. The predictor is the linear solution
    # of KC0uu, and the convergence criterion is Crisfield's, with the
    # residual and the forces scaled with the diagonal of KC0uu, to 1e-9.
    # A consistent tangent reaches 2.0e-9 in two iterations here
    callables = Callables(elements, points, weights, KC0, bu, fext)
    an = Analysis(calc_fext=callables.calc_fext,
                  calc_fint=callables.calc_fint,
                  calc_kC=callables.calc_kC,
                  calc_kG=callables.calc_kG)
    an.initialInc = 1.
    # at most 7 corrections, otherwise structsolve cuts the load step
    an.maxNumIter = 7
    increments, cs = an.static(NLgeom=True)
    if increments != [1.]:
        raise RuntimeError('Not converged!')


if __name__ == '__main__':
    test_nonlinear_axial_compression_load_controlled()
