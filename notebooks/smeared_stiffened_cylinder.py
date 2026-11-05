r"""Smeared-stiffener buckling theory of a stringer-stiffened cylinder

The theory of Block, Card and Mikulas [1]_ for an isotropic cylinder with
eccentric stringers whose stiffnesses are averaged over the stringer spacing
`d`, specialized to axial compression without rings: Donnell-type
kinematics of the skin, the stringers as beams whose strain varies linearly
across their depth, compatible with the skin surface, their twisting
approximated by `G_s J_s w_{,xy}^2/d`, and a membrane pre-buckling state of
the axial force per unit circumference `\bar{N}_x` (positive in compression)
carried by the skin and the stringers under a uniform axial stress.

The change of the total potential energy, equations (5), (7) and (9) of
[1]_ (report pages 7 and 8) with the isotropic constants of equations (12)
(report page 10), is

.. math::

    \Pi = \frac{1}{2} \int_0^{2 \pi R} \int_0^L \Big[
    \frac{E t}{1 - \mu^2} \left( u_{,x}^2 + 2 \mu u_{,x} \epsilon_y
    + \epsilon_y^2 \right) + G t \left(u_{,y} + v_{,x}\right)^2
    + D \left( w_{,xx}^2 + 2 \mu w_{,xx} w_{,yy} + w_{,yy}^2
    + 2 (1 - \mu) w_{,xy}^2 \right) \\
    + \frac{1}{d} \left( E_s A_s u_{,x}^2 - 2 \bar{z}_s E_s A_s u_{,x}
    w_{,xx} + E_s I_{o_s} w_{,xx}^2 + G_s J_s w_{,xy}^2 \right)
    - \bar{N}_x w_{,x}^2 \Big] \, dx \, dy

with `\epsilon_y = v_{,y} + w/R`, `w` positive outwards, `\bar{z}_s` the
distance from the centroid of a stringer to the middle surface of the skin,
positive for external stringers, and `I_{o_s} = I_s + A_s \bar{z}_s^2` the
moment of inertia of a stringer about the middle surface of the skin.

Simply supported ends, `w = M_x = N_x = v = 0` ([1]_, report page 12):
the closed-form stability equation (15) of [1]_ (report page 12), minimized
over the axial half-waves `m` and the circumferential waves `n`
(:func:`buckling_ss`).

Clamped ends, `u = v = w = w_{,x} = 0`: the Galerkin solution of Card and
Jones [2]_ (appendix A, report pages 13 to 19) of the same equations with the
series of their equation (A5) (report page 14),

.. math::

    u = \cos\frac{n y}{R} \sum_i a_i \sin\frac{i \pi x}{L}, \quad
    v = \sin\frac{n y}{R} \sum_i b_i \sin\frac{i \pi x}{L}, \quad
    w = \cos\frac{n y}{R} \sum_i c_i \left[\cos\frac{(i - 1) \pi x}{L}
    - \cos\frac{(i + 1) \pi x}{L}\right]

Every function of (A5) satisfies the boundary conditions, which are all
geometric, so that the Galerkin equations (A6) of [2]_ are those of the
Rayleigh-Ritz method applied to `\Pi`; here they are formed from `\Pi`
by Gauss-Legendre integration along `x` (:func:`buckling_cc`). The same
Ritz form with the functions of equation (14) of [1]_ reproduces equation
(15) of [1]_ (``ends='SS'`` of :func:`buckling_ritz`).

References
----------
.. [1] Block, D. L., Card, M. F., and Mikulas, M. M., Jr., "Buckling of
   eccentrically stiffened orthotropic cylinders", NASA TN D-2960, August
   1965.

.. [2] Card, M. F., and Jones, R. M., "Experimental and theoretical results
   for buckling of eccentrically stiffened cylinders", NASA TN D-3639,
   October 1966. https://ntrs.nasa.gov/citations/19660027866

"""
import numpy as np
from numpy import pi
from scipy.linalg import eigh


def blade_properties(b, h, t, side):
    r"""Section properties of a rectangular (blade) stringer of width ``b``
    and height ``h`` on the surface of a skin of thickness ``t``

    Returns a dictionary with the area `A_s = b h`, the eccentricity
    `\bar{z}_s = \pm (t + h)/2` (positive for ``side='external'``), the
    moment of inertia about the centroid `I_s = b h^3/12`, the moment of
    inertia about the middle surface of the skin `I_{o_s} = I_s + A_s
    \bar{z}_s^2` and the torsional constant of the symbol list of [2]_
    (report page 3), `J = b^3 h (1/3 - 64 b/(\pi^5 h) \tanh(\pi h/(2 b)))`.
    """
    sign = 1. if side == 'external' else -1.
    A = b*h
    z = sign*(t + h)/2
    I = b*h**3/12
    J = b**3*h*(1/3 - 64/pi**5*b/h*np.tanh(pi*h/(2*b)))
    return dict(A=A, z=z, I=I, Io=I + A*z**2, J=J)


def buckling_ss(R, L, t, E, mu, d, A, z, Io, J, m, n):
    r"""Critical `\bar{N}_x` of the simply supported cylinder for the
    half-waves ``m`` and the waves ``n``, equation (15) of [1]_ without rings
    and without `\bar{N}_y`; ``m`` and ``n`` may be arrays (broadcast)"""
    G = E/(2*(1 + mu))
    C = E*t/(1 - mu**2)
    D = E*t**3/(12*(1 - mu**2))
    am = m*pi/L
    bn = n/R
    EAd = E*A/d
    A11 = (C + EAd)*am**2 + G*t*bn**2
    A12 = (mu*C + G*t)*am*bn
    A13 = mu*C/R*am + EAd*z*am**3
    A22 = G*t*am**2 + C*bn**2
    A23 = C/R*bn
    # 2 mu_y D_x/(1 - mu_x mu_y) + 2 D_xy = 2 D for the isotropic skin
    A33 = ((D + E*Io/d)*am**4 + (2*D + G*J/d)*am**2*bn**2 + D*bn**4
           + C/R**2)
    det = A11*A22 - A12**2
    rhs = (A33 + (A12*A23 - A13*A22)/det*A13
           + (A12*A13 - A11*A23)/det*A23)
    return rhs/am**2


def _basis(ends, i, x, L):
    """Axial functions of u, v, w and the derivatives needed, term ``i``"""
    k = i*pi/L
    if ends == 'CC':
        # equation (A5) of [2]
        U, Ux = np.sin(k*x), k*np.cos(k*x)
        km, kp = (i - 1)*pi/L, (i + 1)*pi/L
        Wf = np.cos(km*x) - np.cos(kp*x)
        Wx = -km*np.sin(km*x) + kp*np.sin(kp*x)
        Wxx = -km**2*np.cos(km*x) + kp**2*np.cos(kp*x)
        return U, Ux, U, Ux, Wf, Wx, Wxx
    # equation (14) of [1]: u ~ cos, v ~ sin, w ~ sin along x
    U, Ux = np.cos(k*x), -k*np.sin(k*x)
    V, Vx = np.sin(k*x), k*np.cos(k*x)
    return U, Ux, V, Vx, V, Vx, -k**2*V


def buckling_ritz(R, L, t, E, mu, d, A, z, Io, J, n, ends='CC', nterms=12,
                  nquad=None):
    r"""Rayleigh-Ritz form of the theory of [1]_ for the waves ``n``

    ``ends='CC'``: the series (A5) of [2]_ with ``nterms`` terms in each of
    `u`, `v`, `w` (clamped ends); ``ends='SS'``: the functions of equation
    (14) of [1]_, `m = 1, \ldots,` ``nterms``. Returns the critical values
    of `\bar{N}_x`, in increasing order, and the Ritz coefficients (columns,
    ordered `a_1 \ldots a_I, b_1 \ldots b_I, c_1 \ldots c_I`)."""
    G = E/(2*(1 + mu))
    C = E*t/(1 - mu**2)
    D = E*t**3/(12*(1 - mu**2))
    bn = n/R
    I = nterms
    nquad = nquad or 2*I + 8
    xi, wi = np.polynomial.legendre.leggauss(nquad)
    x = L/2*(xi + 1)
    wq = L/2*wi
    zero = np.zeros((nquad, I))
    U, Ux, V, Vx, Wf, Wx, Wxx = (np.empty((nquad, I)) for _ in range(7))
    for j in range(I):
        U[:, j], Ux[:, j], V[:, j], Vx[:, j], Wf[:, j], Wx[:, j], Wxx[:, j] \
            = _basis(ends, j + 1, x, L)
    # linear maps from (a, b, c) to the strains and curvatures; the
    # factors cos(ny/R) or sin(ny/R), whose squares integrate to pi R for
    # n >= 1, are left out
    ex = np.hstack([Ux, zero, zero])
    ey = np.hstack([zero, bn*V, Wf/R])
    gxy = np.hstack([-bn*U, Vx, zero])
    kxx = np.hstack([zero, zero, Wxx])
    kyy = np.hstack([zero, zero, -bn**2*Wf])
    kxy = np.hstack([zero, zero, -bn*Wx])
    wx = np.hstack([zero, zero, Wx])

    def q(a, b, c=1.):
        return c*(a*wq[:, None]).T @ b

    K = (C*(q(ex, ex) + mu*(q(ex, ey) + q(ey, ex)) + q(ey, ey))
         + G*t*q(gxy, gxy)
         + D*(q(kxx, kxx) + mu*(q(kxx, kyy) + q(kyy, kxx)) + q(kyy, kyy)
              + 2*(1 - mu)*q(kxy, kxy))
         + (E*A*q(ex, ex) - z*E*A*(q(ex, kxx) + q(kxx, ex))
            + E*Io*q(kxx, kxx) + G*J*q(kxy, kxy))/d)
    KG = q(wx, wx)
    # K x = N KG x, KG semi-definite: largest 1/N of KG x = (1/N) K x
    mu_, X = eigh(KG, K)
    keep = mu_ > 1e-14*mu_.max()
    N = 1/mu_[keep][::-1]
    return N, X[:, keep][:, ::-1]


def minimum(func, ns):
    r"""Lowest value of ``func(n)`` over ``ns`` and its ``n``; ``func``
    returns an array of critical loads (its first entry is used)"""
    best = None
    for n in ns:
        N = np.atleast_1d(func(n))[0]
        if best is None or N < best[0]:
            best = (N, n)
    return best


def axial_half_waves(L, X, nterms, ends='CC', npts=401):
    """Number of axial half-waves of `w` of the Ritz mode ``X`` (one more
    than the sign changes of `w` along `x`, values below 2 % of the largest
    one ignored)"""
    x = np.linspace(0, L, npts)
    w = np.zeros(npts)
    for j in range(nterms):
        w += X[2*nterms + j]*_basis(ends, j + 1, x, L)[4]
    w = w[np.abs(w) > 0.02*np.abs(w).max()]
    return int(np.sum(np.sign(w[1:]) != np.sign(w[:-1]))) + 1
