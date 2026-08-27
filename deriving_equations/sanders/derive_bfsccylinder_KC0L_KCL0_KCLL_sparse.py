"""
Constitutive nonlinear stiffness matrix for BFS cylinder with Sanders-type kinematics
"""
import numpy as np
import sympy
from sympy import var, Matrix, symbols, simplify

num_nodes = 4
cpu_count = 6
DOF = 10

def main():
    var('xi, eta, lex, ley, rho, weight')
    var('R')
    var('A11, A12, A16, A22, A26, A66')
    var('B11, B12, B16, B22, B26, B66')
    var('D11, D12, D16, D22, D26, D66')

    #ley calculated from nodal positions and radius

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

    Su = sympy.Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0, 0, 0, 0,
        Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0, 0, 0, 0,
        Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0, 0, 0, 0,
        Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0, 0, 0, 0,
        ]])
    Sv = sympy.Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0,
        0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0,
        0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0,
        0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0,
        ]])
    Sw = sympy.Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        0, 0, 0, 0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), Hxyi(-1, -1),
        0, 0, 0, 0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), Hxyi(+1, -1),
        0, 0, 0, 0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), Hxyi(+1, +1),
        0, 0, 0, 0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), Hxyi(-1, +1),
        ]])
    A = Matrix([
        [A11, A12, A16],
        [A12, A22, A26],
        [A16, A26, A66]])
    B = Matrix([
        [B11, B12, B16],
        [B12, B22, B26],
        [B16, B26, B66]])
    D = Matrix([
        [D11, D12, D16],
        [D12, D22, D26],
        [D16, D26, D66]])

    # membrane
    Su_x = (2/lex)*Su.diff(xi)
    Su_y = (2/ley)*Su.diff(eta)
    Sv_x = (2/lex)*Sv.diff(xi)
    Sv_y = (2/ley)*Sv.diff(eta)

    Bm = Matrix([
        Su_x, # epsilon_xx
        Sv_y + 1/R*Sw, # epsilon_yy
        Su_y + Sv_x # gamma_xy
        ])

    Bms = []
    for i in range(Bm.shape[0]):
        Bmis = []
        for j in range(Bm.shape[1]):
            Bmij = Bm[i, j]
            if Bmij != 0:
                print('                Bm%d_%02d = %s' % ((i+1), (j+1),
                    str(simplify(Bmij))))
                Bmis.append(symbols('Bm%d_%02d' % (i+1, j+1)))
            else:
                Bmis.append(0)
        Bms.append(Bmis)
    Bm = sympy.Matrix(Bms)

    print()
    print()
    print()

    uG = Matrix([symbols(r'ue[%d]' % i) for i in range(0, Bm.shape[1])])
    Sw_x = (2/lex)*Sw.diff(xi)
    Sw_y = (2/ley)*Sw.diff(eta)
    v = Sv*uG
    w_x = Sw_x*uG
    w_y = Sw_y*uG
    print('v =', simplify(v)[0, 0])
    print('w_x =', simplify(w_x)[0, 0])
    print('w_y =', simplify(w_y)[0, 0])
    v = var('v')
    w_x = var('w_x')
    w_y = var('w_y')
    BmL = Matrix([
        w_x*Sw_x,
        w_y*Sw_y + 1/R**2*v*Sv - 1/R*v*Sw_y - 1/R*w_y*Sv,
        w_x*Sw_y + w_y*Sw_x - 1/R*v*Sw_x - 1/R*w_x*Sv
        ])

    BmLs = []
    for i in range(BmL.shape[0]):
        BmLis = []
        for j in range(BmL.shape[1]):
            BmLij = BmL[i, j]
            if BmLij != 0:
                print('                BmL%d_%02d = %s' % ((i+1), (j+1),
                    str(simplify(BmLij))))
                BmLis.append(symbols('BmL%d_%02d' % (i+1, j+1)))
            else:
                BmLis.append(0)
        BmLs.append(BmLis)
    BmL = Matrix(BmLs)

    print()
    print()
    print()

    # bending
    Sphix = -(2/lex)*Sw.diff(xi)
    Sphiy = -(2/ley)*Sw.diff(eta)
    Sphix_x = (2/lex)*Sphix.diff(xi)
    Sphix_y = (2/ley)*Sphix.diff(eta)
    Sphiy_x = (2/lex)*Sphiy.diff(xi)
    Sphiy_y = (2/ley)*Sphiy.diff(eta)
    Bb = sympy.Matrix([
        Sphix_x,
        Sphiy_y + 1/R*Sv_y,
        Sphix_y + Sphiy_x + 3/2*1/R*Sv_x - 1/(2*R)*Su_y
        ])

    Bbs = []
    for i in range(Bb.shape[0]):
        Bbis = []
        for j in range(Bb.shape[1]):
            Bbij = Bb[i, j]
            if Bbij != 0:
                print('                Bb%d_%02d = %s' % ((i+1), (j+1),
                    str(simplify(Bbij))))
                Bbis.append(symbols('Bb%d_%02d' % (i+1, j+1)))
            else:
                Bbis.append(0)
        Bbs.append(Bbis)
    Bb = Matrix(Bbs)

    print()
    print()
    print()

    # Constitutive nonlinear stiffness matrix in element coordinate KCNLe
    KC0Le = Bm.T*A*BmL + Bb.T*B*BmL
    KCL0e = BmL.T*A*Bm + BmL.T*B*Bb
    KCLLe = BmL.T*A*BmL
    KCNLe = weight*(lex*ley)/4.*(KC0Le + KCL0e + KCLLe)

    # KCNL represents the global constitutive nonlinear stiffness matrix
    # in case we want to apply coordinate transformations
    KCNL = KCNLe

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

    print('printing code for sparse implementation')
    for ind, val in np.ndenumerate(KCNL):
        if val == 0:
            continue
        print('                k += 1')
        print('                KCNLv[k] +=', KCNL[ind])

    print()
    print()
    print()
    KCNL_SPARSE_SIZE = 0
    for ind, val in np.ndenumerate(KCNL):
        if val == 0:
            continue
        KCNL_SPARSE_SIZE += 1
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('        k += 1')
        print('        KCNLr[k] = %d+%s' % (i%DOF, si))
        print('        KCNLc[k] = %d+%s' % (j%DOF, sj))
    print('KCNL_SPARSE_SIZE', KCNL_SPARSE_SIZE)

if __name__ == '__main__':
    main()
