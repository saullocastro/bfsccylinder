"""
Follower pressure for BFSC cylinder with Donnell-type kinematics

A pressure p that stays normal to the deformed mid-surface and acts on its
deformed area does the virtual work

    dW_p = int p a.du dx dy,  a = X,x x X,y,  X = X0 + u

with p per unit undeformed area and a function of the undeformed coordinates.
In the basis of the undeformed mid-surface, w positive outwards:

    A = X,x = (1 + u,x, v,x, w,x)
    B = X,y = (u,y, 1 + v,y + kw w, w,y - kv v)

with kw = 1/R and kv = 0 for the Donnell-type kinematics, which neglect the
term v/R of the rotation of the normal, as their strains do. The area vector is split into
its parts of order 0, 1 and 2 in the displacement gradients, a = a0 + a1 +
a2, and the kernels use

    a = ref a0 + a1 + qf a2

with ref = 1 including the dead load a0 = (0, 0, 1) and qf = 1 the quadratic
part a2. The force vector is

    F_p(u) = int p (Su' a_x + Sv' a_y + Sw' a_z) dA

and the load stiffness KCfollower = -dF_p/du, a full matrix that is
unsymmetric in general. With qf = 0, F_p is affine and KCfollower constant.

Operators at an integration point, in the code:

    Su1 = Su, Su2 = Su_x, Su3 = Su_y, likewise Sv1..Sv3 and Sw1..Sw3

The entries of KCfollower that are not identically zero for qf = 1 are
written for both truncations, such that KCfollower_SPARSE_SIZE does not
depend on the truncation.
"""
import numpy as np
import sympy
from sympy import var, symbols, Matrix, expand, diff

num_nodes = 4
DOF = 10

def main():
    var('xi, eta, lex, ley, weight')
    var('R')
    # pressure at the integration point, the truncation flags
    var('pij, qf, ref')
    # displacement gradients at the integration point
    var('u_x, u_y, v_x, v_y, v, w, w_x, w_y')
    var('ax, ay, az')

    ONE = sympy.Integer(1)

    # shape functions
    # - from Reference:
    #     OCHOA, O. O.; REDDY, J. N. Finite Element Analysis of Composite Laminates. Dordrecht: Springer, 1992.
    # cubic
    Hi = lambda xii, etai: ONE/16.*(xi + xii)**2*(xi*xii - 2)*(eta+etai)**2*(eta*etai - 2)
    Hxi = lambda xii, etai: -lex/32.*xii*(xi + xii)**2*(xi*xii - 1)*(eta + etai)**2*(eta*etai - 2)
    Hyi = lambda xii, etai: -ley/32.*(xi + xii)**2*(xi*xii - 2)*etai*(eta + etai)**2*(eta*etai - 1)
    Hxyi = lambda xii, etai: lex*ley/64.*xii*(xi + xii)**2*(xi*xii - 1)*etai*(eta + etai)**2*(eta*etai - 1)

    # node 1 (-1, -1)
    # node 2 (+1, -1)
    # node 3 (+1, +1)
    # node 4 (-1, +1)

    Su = Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0, 0, 0, 0,
        Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0, 0, 0, 0,
        Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0, 0, 0, 0,
        Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0, 0, 0, 0,
        ]])
    Sv = Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0,
        0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0,
        0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0,
        0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0,
        ]])
    Sw = Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        0, 0, 0, 0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), Hxyi(-1, -1),
        0, 0, 0, 0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), Hxyi(+1, -1),
        0, 0, 0, 0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), Hxyi(+1, +1),
        0, 0, 0, 0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), Hxyi(-1, +1),
        ]])

    # rows of the field and of its first derivatives at the point
    Sus = Matrix([Su, (2/lex)*Su.diff(xi), (2/ley)*Su.diff(eta)])
    Svs = Matrix([Sv, (2/lex)*Sv.diff(xi), (2/ley)*Sv.diff(eta)])
    Sws = Matrix([Sw, (2/lex)*Sw.diff(xi), (2/ley)*Sw.diff(eta)])

    def operator(name, S):
        out = []
        for i in range(S.shape[0]):
            row = []
            for j in range(S.shape[1]):
                if S[i, j] != 0:
                    print('                %s%d_%02d = %s' % (name, i+1, j+1, str(S[i, j])))
                    row.append(symbols('%s%d_%02d' % (name, i+1, j+1)))
                else:
                    row.append(0)
            out.append(row)
        return Matrix(out)

    Sus = operator('Su', Sus)
    Svs = operator('Sv', Svs)
    Sws = operator('Sw', Sws)
    print()

    # kinematics: Donnell-type
    kw = 1/R
    kv = 0

    A = Matrix([1 + u_x, v_x, w_x])
    B = Matrix([u_y, 1 + v_y + kw*w, w_y - kv*v])
    acomp = A.cross(B)
    point = (u_x, u_y, v_x, v_y, v, w, w_x, w_y)
    t = sympy.Symbol('t')
    scaled = acomp.subs({q: t*q for q in point}, simultaneous=True)
    a0, a1, a2 = [scaled.applyfunc(lambda e: expand(diff(e, t, n).subs(t, 0)/sympy.factorial(n)))
                  for n in range(3)]
    assert (a0 + a1 + a2 - acomp).applyfunc(expand) == Matrix([0, 0, 0])
    a = (ref*a0 + a1 + qf*a2).applyfunc(expand)

    # displacement gradients at the point, from the element displacements
    ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, 4*DOF)])
    rows = {u_x: Sus[1, :], u_y: Sus[2, :], v_x: Svs[1, :], v_y: Svs[2, :],
            v: Svs[0, :], w: Sws[0, :], w_x: Sws[1, :], w_y: Sws[2, :]}
    used = [q for q in point if q in set().union(*[e.free_symbols for e in a])]
    for q in used:
        print('                %s = %s' % (q, str((rows[q]*ue)[0, 0])))
    print('                ax = %s' % str(a[0]))
    print('                ay = %s' % str(a[1]))
    print('                az = %s' % str(a[2]))
    print()

    def name_ind(i):
        if i >=0 and i < DOF:
            return 'c1'
        elif i >= DOF and i < 2*DOF:
            return 'c2'
        elif i >= 2*DOF and i < 3*DOF:
            return 'c3'
        elif i >= 3*DOF and i < 4*DOF:
            return 'c4'
        else:
            raise

    # force vector
    fe = weight*(lex*ley)/4.*pij*(Sus[0, :].T*ax + Svs[0, :].T*ay + Sws[0, :].T*az)
    for i, fi in enumerate(fe):
        if fi == 0:
            continue
        print('fext[%d + %s] +=' % (i%DOF, name_ind(i)), fi)
    print()

    # load stiffness, -dF_p/du
    da = []
    for comp in a:
        row = sympy.zeros(1, 4*DOF)
        for q in used:
            dq = diff(comp, q)
            if dq != 0:
                row += dq*rows[q]
        da.append(row)
    KCfollower = -weight*(lex*ley)/4.*pij*(Sus[0, :].T*da[0]
                                           + Svs[0, :].T*da[1]
                                           + Sws[0, :].T*da[2])

    print('printing code for sparse implementation')
    for ind, val in np.ndenumerate(KCfollower):
        if val == 0:
            continue
        print('                k += 1')
        print('                KCfollowerv[k] +=', KCfollower[ind])
    print()
    print()
    print()
    KCfollower_SPARSE_SIZE = 0
    for ind, val in np.ndenumerate(KCfollower):
        if val == 0:
            continue
        KCfollower_SPARSE_SIZE += 1
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('        k += 1')
        print('        KCfollowerr[k] = %d+%s' % (i%DOF, si))
        print('        KCfollowerc[k] = %d+%s' % (j%DOF, sj))
    print('KCfollower_SPARSE_SIZE', KCfollower_SPARSE_SIZE)

if __name__ == '__main__':
    main()
