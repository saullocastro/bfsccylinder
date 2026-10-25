r"""Deterministic and verified lowest natural frequencies

``scipy.sparse.linalg.eigsh`` with ``M`` in shift-invert mode starts ARPACK
from a random vector, and for the clustered lowest frequencies of a cylinder
it made the natural-frequency tests fail now and then (ARPACK error -8),
scattered the lowest eigenvalue by about 0.2 % from run to run and returned
eigenpairs with relative residuals of the order of 1e-3. Here ARPACK solves
the standard problem of the shift-inverted operator from a fixed starting
vector, with a larger Krylov subspace, and the result is verified:

- the relative residual of each eigenpair must be below ``rtol``;
- no eigenvalue below the highest one returned may be missing: by
  Sylvester's law of inertia, the number of negative pivots of the
  symmetric factorization of `\boldsymbol{K} - s \boldsymbol{M}`, for `s` slightly above the
  highest eigenvalue returned, is the number of eigenvalues below `s`.
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, eigs, splu


def negative_pivots(A):
    r"""Number of negative eigenvalues of the symmetric sparse matrix ``A``,
    from an `\boldsymbol{L}\boldsymbol{D}\boldsymbol{L}^\top` factorization with a symmetric permutation"""
    lu = splu(csc_matrix(A), permc_spec='MMD_AT_PLUS_A',
              diag_pivot_thresh=0., options=dict(SymmetricMode=True))
    assert np.array_equal(lu.perm_r, lu.perm_c), 'row interchange'
    d = lu.U.diagonal()
    assert np.all(np.isfinite(d)) and np.all(d != 0)
    return int(np.count_nonzero(d < 0))


def lowest_frequencies(K, M, k, rtol=1e-8):
    r"""The ``k`` lowest `\omega_n^2` and modes of `\boldsymbol{K} \boldsymbol{u} = \omega_n^2
    \boldsymbol{M} \boldsymbol{u}`, ``K`` positive definite, verified"""
    # shift-invert with the shift -1, below the lowest eigenvalue: the
    # largest eigenvalues mu of ([K] + [M])^-1 [M] give omega_n**2 = 1/mu - 1
    lu = splu(csc_matrix(K + M))
    op = LinearOperator(K.shape, matvec=lambda x: lu.solve(M @ x),
                        dtype=float)
    mu, eigvecs = eigs(op, k=k, which='LM', tol=0, ncv=4*k,
                       v0=np.ones(K.shape[0]))
    assert np.all(np.abs(mu.imag) <= 1e-10*np.abs(mu))
    eigvals = 1/mu.real - 1
    # real modes: remove the arbitrary complex phase of each vector
    i = np.argmax(np.abs(eigvecs), axis=0)
    phase = eigvecs[i, np.arange(k)]
    eigvecs = (eigvecs*(np.abs(phase)/phase)).real
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    for lam, x in zip(eigvals, eigvecs.T):
        Kx = K @ x
        res = np.linalg.norm(Kx - lam*(M @ x))/np.linalg.norm(Kx)
        assert res < rtol, 'eigenpair residual %.1e' % res
    # the next eigenvalue, if any, is above the last one returned; the
    # shift lies in the gap between them, away from a multiple eigenvalue
    s = eigvals[-1]*(1 + 1e-6)
    assert negative_pivots(K - s*M) == k, 'eigenvalues missing'
    return eigvals, eigvecs
