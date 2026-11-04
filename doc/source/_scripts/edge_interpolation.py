"""Figure of assembling.rst: w along an edge x = const between its nodes

The interpolation of the element, BFSCCylinder.update_Sw, is evaluated along
the edge xi = -1 of three adjacent elements. The nodes have w = 0 and the
nodal values of w_y of a mode with n circumferential waves, w_y =
cos(n y/R); with w_y free they stay as they are, with w_y constrained they
are zero.
"""
import numpy as np
import matplotlib.pyplot as plt

from bfsccylinder import BFSCCylinder, DOF

R = 0.25
ny = 50
ley = 2*np.pi*R/ny
n = 9
nel = 3

shell = BFSCCylinder(4)
shell.R = R
shell.lex = ley
shell.ley = ley

eta = np.linspace(-1, 1, 41)
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for constrained, style, label in ((False, '-', r'$w = 0$ at the nodes, $w_{,y}$ free'),
                                  (True, '--', r'$w = w_{,y} = 0$ at the nodes')):
    for e in range(nel):
        ya, yb = e*ley, (e + 1)*ley
        ue = np.zeros(4*DOF)
        if not constrained:
            # nodes 1 and 4 lie on the edge xi = -1, at y = ya and y = yb
            ue[0*DOF + 8] = np.cos(n*ya/R)
            ue[3*DOF + 8] = np.cos(n*yb/R)
        w = []
        for et in eta:
            shell.update_Sw(-1., et)
            w.append(np.asarray(shell.Sw) @ ue)
        y = ya + (eta + 1)/2*ley
        ax.plot(y/ley, np.asarray(w)/ley, style, color='C0' if not constrained else 'C1',
                label=label if e == 0 else None)
ax.plot(np.arange(nel + 1), np.zeros(nel + 1), 'ko', label='nodes')
ax.set_xlabel(r'$y$ / ley')
ax.set_ylabel(r'$w$ / ley')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
