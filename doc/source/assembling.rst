Assembling a model
==================

.. currentmodule:: bfsccylinder.bfsccylinder

``bfsccylinder`` computes the contributions of one element; the mesh, the
global matrices, the boundary conditions and the loads are written by the
user with NumPy and SciPy, as in the tests of the repository, see
:doc:`ex_linear_static`. The functions and the sizes below are those of
the Donnell-type element; the module :mod:`bfsccylinder.bfsccylinder_sanders`,
imported from ``bfsccylinder.sanders``, has the same names, see
:doc:`kinematics`.


Mesh of a closed cylinder
-------------------------

A cylinder of length `L` and radius `R` is divided into ``nx - 1`` elements
along `x` and ``ny`` elements around the circumference, of sides ``lex =
L/(nx - 1)`` and ``ley = 2*pi*R/ny``. The tests number ``nx`` by ``ny + 1``
nodes with `x` varying slowest and close the cylinder by giving the last
column of nodes the numbers of the first one, before taking the four nodes
of each element from the corners of the array, in the order of
:doc:`element`::

    nids = 1 + np.arange(nx*(ny+1))
    nids_mesh = nids.reshape(nx, ny+1)
    nids_mesh[:, -1] = nids_mesh[:, 0]
    nids = np.unique(nids_mesh)
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

with ``nid_pos`` mapping each node identification number to its index,
such that ``c1 = DOF*nid_pos[n1]``, and so on, and the model has ``N =
DOF*nx*ny`` degrees of freedom. The elements of the last column share their
nodes 3 and 4 with the nodes 2 and 1 of the first column, and the
continuity of the degrees of freedom closes the shell around the
circumference.


Element loop and sparse arrays
------------------------------

Each function ``update_K*`` and :func:`.update_M` writes the entries of one
element in three 1-D arrays, the row indices, the column indices and the
values, at the positions ``init_k`` to ``init_k + SPARSE_SIZE - 1``, with
``init_k`` the attribute of the element for that matrix and ``SPARSE_SIZE``
the number of entries of one element:

.. list-table::
    :header-rows: 1

    * - Function
      - Position attribute
      - Entries per element
    * - :func:`.update_KC0`
      - ``init_k_KC0``
      - ``KC0_SPARSE_SIZE``
    * - :func:`.update_KCNL`
      - ``init_k_KCNL``
      - ``KCNL_SPARSE_SIZE``
    * - :func:`.update_KG`, :func:`.update_KG_constant_stress`
      - ``init_k_KG``
      - ``KG_SPARSE_SIZE``
    * - :func:`.update_M`
      - ``init_k_M``
      - ``M_SPARSE_SIZE``
    * - :func:`.update_KCfollower`
      - ``init_k_KCfollower``
      - ``KCfollower_SPARSE_SIZE``

The sizes of the Sanders-type element are larger for KCNL, KG, M and
KCfollower, see :mod:`bfsccylinder.bfsccylinder_sanders`; the code is the
same when the sizes are taken from the namespace of the element used. For
``num_elements`` elements the arrays have ``SPARSE_SIZE*num_elements``
entries, of type ``INT`` for the indices and ``DOUBLE`` for the values, and
the element of index ``k`` gets ``init_k = k*SPARSE_SIZE``, e.g. by
increasing a counter by ``SPARSE_SIZE`` in the element loop::

    nint = 4
    points, weights = get_points_weights(nint=nint)
    KC0r = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        shell = BFSCCylinder(nint)
        shell.c1 = DOF*nid_pos[n1]
        ...
        shell.R = R
        shell.lex = L/(nx-1)
        shell.ley = 2*pi*R/ny
        assign_constant_ABD(shell, prop)
        shell.init_k_KC0 = init_k_KC0
        update_KC0(shell, points, weights, KC0r, KC0c, KC0v)
        init_k_KC0 += KC0_SPARSE_SIZE

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

``scipy.sparse.coo_matrix`` sums the values at repeated positions, the
entries of the nodes shared by several elements. ``INT``, ``DOUBLE``,
``DOF`` and the sizes are given by the ``bfsccylinder`` and
``bfsccylinder.sanders`` namespaces. The elements write the entries of both
triangles of each matrix, not only an upper triangle, and the matrices that
are symmetric come out symmetric after the sum.

The functions overwrite the indices but **add** the values to the existing
ones. Arrays reused for a new configuration, e.g. KCNL and KG at each
iteration of a nonlinear analysis, are zeroed first, ``KCNLv *= 0``; the
indices do not change for a fixed mesh.

The vectors are assembled without position attributes:
:func:`.update_fint` and :func:`.update_fext_follower` add the vector of
one element directly to a global vector of size ``N``, at the positions of
its nodes, and the functions that depend on the displacements read them
from a global vector ``u`` of size ``N``.


Boundary conditions and loads
-----------------------------

The tests constrain degrees of freedom by partitioning the global system: a
boolean array ``bk`` marks the constrained degrees of freedom, selected with
the slices of :doc:`element` and the coordinates of the nodes, and ``bu =
~bk`` the unknown ones, e.g. for `u = v = w = 0` at the nodes of both
ends::

    bk = np.zeros(N, dtype=bool)
    checkSS = isclose(x, 0) | isclose(x, L)
    bk[0::DOF] = checkSS
    bk[3::DOF] = checkSS
    bk[6::DOF] = checkSS
    bu = ~bk

    Kuu = KC0[bu, :][:, bu]

A concentrated load is a value of the global force vector at a degree of
freedom of a node, e.g. ``f[6::DOF][check] = fmid`` for a normal force, and
an axial force at the nodes of an edge, e.g. ``fext[0::DOF][checkTopEdge] =
-load/ny``, see :doc:`ex_nonlinear_static`. The force vector of a pressure
is given by :func:`.update_fext_follower`, at `\{u\} = \{0\}` the dead
pressure load, see :doc:`follower_pressure`.

A prescribed displacement, e.g. the axial shortening of one end, is a known
value ``uk = u[bk]`` of a constrained degree of freedom, which moves to the
right-hand side of the reduced system, `[K_{uu}]\{u_u\} = -[K_{uk}]\{u_k\}`,
and whose reaction forces are `\{f_k\} = [K_{uk}]^T\{u_u\} +
[K_{kk}]\{u_k\}`::

    Kuk = KC0[bu, :][:, bk]
    Kkk = KC0[bk, :][:, bk]
    fu = -Kuk*uk
    ...
    fk = Kuk.T*uu + Kkk*uk

see :doc:`ex_linear_buckling`. The solution of the reduced system is
expanded to the size of the model, ``u[bu] = uu``, before it is given to the
functions that depend on the displacements, e.g. :func:`.update_KG`.
