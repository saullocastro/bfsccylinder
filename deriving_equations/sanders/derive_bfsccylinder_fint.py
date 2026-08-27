"""
Internal force vector
"""
import numpy as np
import sympy
from sympy import var, Matrix, symbols, simplify

num_nodes = 4
cpu_count = 6
DOF = 10

if True:
#def main():
    var('xi, eta, lex, ley, rho, weight')
    var('R')
    var('Nxx, Nyy, Nxy, Mxx, Myy, Mxy')
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
                Bmis.append(symbols('Bm%d_%02d' % (i+1, j+1)))
            else:
                Bmis.append(0)
        Bms.append(Bmis)
    Bm = sympy.Matrix(Bms)

    Sw_x = (2/lex)*Sw.diff(xi)
    Sw_y = (2/ley)*Sw.diff(eta)
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
                BmLis.append(symbols('BmL%d_%02d' % (i+1, j+1)))
            else:
                BmLis.append(0)
        BmLs.append(BmLis)
    BmL = Matrix(BmLs)

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
                Bbis.append(symbols('Bb%d_%02d' % (i+1, j+1)))
            else:
                Bbis.append(0)
        Bbs.append(Bbis)
    Bb = Matrix(Bbs)

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

    ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, Bb.shape[1])])
    N = A*(Bm + BmL)*ue + B*Bb*ue
    M = B*(Bm + BmL)*ue + D*Bb*ue
    print('Nxx =', N[0])
    print('Nyy =', N[1])
    print('Nxy =', N[2])
    print('Mxx =', M[0])
    print('Myy =', M[1])
    print('Mxy =', M[2])

    N = Matrix([[Nxx, Nyy, Nxy]]).T
    M = Matrix([[Mxx, Myy, Mxy]]).T

    fint_terms = Bm.T*N + BmL.T*N + Bb.T*M
    fint = weight*(lex*ley)/4.*(fint_terms)

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

    for i, fi in enumerate(fint):
        if fi == 0:
            continue
        si = name_ind(i)
        print('fint[%d + %s] +=' % (i%DOF, si), fi)

#if __name__ == '__main__':
    #main()
