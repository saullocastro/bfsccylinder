import numpy as np
import sympy
sympy.var('xi, eta')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

lex = 5
w1 = 1
w2 = 3
w3 = 2
w4 = 6
wx1 = 3
wx2 = 5
wx3 = 7
wx4 = 11

H5wi = lambda xii, etai: 1/16.*(xi + xii)**2*(xi*xii - 2)*(eta+etai)**2*(eta*etai - 2)
H5wxi = lambda xii, etai: -lex/32.*xii*(xi + xii)**2*(xi*xii - 1)*(eta + etai)**2*(eta*etai - 2)

x = np.linspace(-1, +1, 100)
y = np.linspace(-1, +1, 100)
x, y = np.meshgrid(x, y)

expr = (H5wi(-1, -1)*w1 + H5wxi(-1, -1)*wx1
      + H5wi(+1, -1)*w2 + H5wxi(+1, -1)*wx2
      + H5wi(+1, +1)*w3 + H5wxi(+1, +1)*wx3
      + H5wi(-1, +1)*w4 + H5wxi(-1, +1)*wx4
        )

f = sympy.lambdify((xi, eta), expr, 'numpy')
fx = sympy.lambdify((xi, eta), expr.diff(xi)*(2/lex), 'numpy')
print(f(x, y).T[[0, -1, -1, 0], [0, 0, -1, -1]])
print(fx(x, y).T[[0, -1, -1, 0], [0, 0, -1, -1]])


