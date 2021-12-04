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
    var('Nxx0, Nyy0, Nxy0, Mxx0, Myy0, Mxy0')
    var('NxxL, NyyL, NxyL')
    var('MxxL, MyyL, MxyL')
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

    Nu = sympy.Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0, 0, 0, 0,
        Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0, 0, 0, 0,
        Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0, 0, 0, 0,
        Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0, 0, 0, 0,
        ]])
    Nv = sympy.Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0,
        0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0,
        0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0,
        0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0,
        ]])
    Nw = sympy.Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        0, 0, 0, 0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), Hxyi(-1, -1),
        0, 0, 0, 0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), Hxyi(+1, -1),
        0, 0, 0, 0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), Hxyi(+1, +1),
        0, 0, 0, 0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), Hxyi(-1, +1),
        ]])

    Nu_x = (2/lex)*Nu.diff(xi)
    Nu_y = (2/ley)*Nu.diff(eta)
    Nv_x = (2/lex)*Nv.diff(xi)
    Nv_y = (2/ley)*Nv.diff(eta)

    Bm = Matrix([
        Nu_x, # epsilon_xx
        Nv_y + 1/R*Nw, # epsilon_yy
        Nu_y + Nv_x # gamma_xy
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

    Nw_x = (2/lex)*Nw.diff(xi)
    Nw_y = (2/ley)*Nw.diff(eta)
    w_x = var('w_x')
    w_y = var('w_y')
    BmL = Matrix([
        w_x*Nw_x,
        w_y*Nw_y,
        w_x*Nw_y + w_y*Nw_x
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

    Nphix = -(2/lex)*Nw.diff(xi)
    Nphiy = -(2/ley)*Nw.diff(eta)
    Nphix_x = (2/lex)*Nphix.diff(xi)
    Nphix_y = (2/ley)*Nphix.diff(eta)
    Nphiy_x = (2/lex)*Nphiy.diff(xi)
    Nphiy_y = (2/ley)*Nphiy.diff(eta)
    Bb = Matrix([
        Nphix_x,
        Nphiy_y,
        Nphix_y + Nphiy_x
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
    N0 = A*Bm*ue + B*Bb*ue
    M0 = B*Bm*ue + D*Bb*ue
    NL = A*BmL*ue
    ML = B*BmL*ue
    print('Nxx0 =', N0[0])
    print('Nyy0 =', N0[1])
    print('Nxy0 =', N0[2])
    print('Mxx0 =', M0[0])
    print('Myy0 =', M0[1])
    print('Mxy0 =', M0[2])
    print('NxxL =', NL[0])
    print('NyyL =', NL[1])
    print('NxyL =', NL[2])
    print('MxxL =', ML[0])
    print('MyyL =', ML[1])
    print('MxyL =', ML[2])

    # Internal force vector
    # PhD thesis Saullo, Eq. 3.8.14
    N0 = Matrix([[Nxx0, Nyy0, Nxy0]]).T
    M0 = Matrix([[Mxx0, Myy0, Mxy0]]).T
    NL = Matrix([[NxxL, NyyL, NxyL]]).T
    ML = Matrix([[MxxL, MyyL, MxyL]]).T

    fint_terms = Bm.T*N0 + Bm.T*NL + BmL.T*N0 + BmL.T*NL + Bb.T*M0 + Bb.T*ML
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
