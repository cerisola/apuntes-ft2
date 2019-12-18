import numpy as np
import scipy as sp

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.lines as mlines
import matplotlib.cm as cmaps
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.unicode'] = True # default in 3.0+
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['legend.fontsize'] = 8

figsize = (5.5, 2.5)
default_pad = 0.2

# =============================================================================
Npt = 1000

wt = np.linspace(0, 2, Npt)

ProbSxPt = np.cos(wt*np.pi/2)**2
ProbSxMt = np.sin(wt*np.pi/2)**2

ProbSyMt = np.cos(wt*np.pi/2 + np.pi/4)**2
ProbSyPt = np.sin(wt*np.pi/2 + np.pi/4)**2

AvgSxt = 0.5 * np.cos(wt*np.pi)
AvgSyt = 0.5 * np.sin(wt*np.pi)

# =============================================================================
wt_ticks = [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
wt_labels = ['$0$', '$\\frac{\pi}{4}$', '$\\frac{\pi}{2}$',
             '$\\frac{3\pi}{4}$', '$\pi$', '$\\frac{5\pi}{4}$',
             '$\\frac{3\pi}{2}$', '$\\frac{7\pi}{4}$', '$2\pi$']

prob_ticks = [0.00, 0.25, 0.50, 0.75, 1.00]
prob_labels = ['$0$', '$\\frac{1}{4}$', '$\\frac{1}{2}$', '$\\frac{3}{4}$',
               '$1$']

spin_ticks = [-0.50, -0.25, 0.00, 0.25, 0.50]
spin_labels = ['$-\\frac{\hbar}{2}$', '$-\\frac{\hbar}{4}$', '$0$',
               '$\\frac{\hbar}{4}$', '$\\frac{\hbar}{2}$']

# =============================================================================
plt.figure(figsize=figsize)
plt.plot(wt, ProbSxPt, '-', label='$P\left(+,x\middle|\psi(t)\\right)$')
plt.plot(wt, ProbSxMt, '-', label='$P\left(-,x\middle|\psi(t)\\right)$')
plt.xticks(wt_ticks, wt_labels)
plt.yticks(prob_ticks, prob_labels)
plt.xlabel('$\omega t$')
plt.ylabel('Probabilidad')
plt.legend(ncol=2, bbox_to_anchor=(0.7, 1.2))
plt.tight_layout(pad=default_pad)
plt.savefig('spinevol_prob_sx.pdf')

# =============================================================================
plt.figure(figsize=figsize)
plt.plot(wt, ProbSxPt, '-', label='$P\left(+,x\middle|\psi(t)\\right)$')
plt.plot(wt, ProbSxMt, '-', label='$P\left(-,x\middle|\psi(t)\\right)$')
plt.plot(wt, ProbSyPt, '-', label='$P\left(+,y\middle|\psi(t)\\right)$')
plt.plot(wt, ProbSyMt, '-', label='$P\left(-,y\middle|\psi(t)\\right)$')
plt.xticks(wt_ticks, wt_labels)
plt.yticks(prob_ticks, prob_labels)
plt.xlabel('$\omega t$')
plt.ylabel('Probabilidad')
plt.legend(ncol=4, handletextpad=0.5, columnspacing=1.0, bbox_to_anchor=(0.07, 1.0))
plt.tight_layout(pad=default_pad)
plt.savefig('spinevol_prob_sxsy.pdf')

# =============================================================================
plt.figure(figsize=figsize)
plt.plot(wt, AvgSxt, '-', label='$\langle S_x\\rangle(t)$')
plt.plot(wt, AvgSyt, '-', label='$\langle S_y\\rangle(t)$')
plt.xticks(wt_ticks, wt_labels)
plt.yticks(spin_ticks, spin_labels)
plt.xlabel('$\omega t$')
plt.ylabel('Valor medio')
plt.legend(ncol=2, bbox_to_anchor=(0.65, 1.2))
plt.tight_layout(pad=default_pad)
plt.savefig('spinevol_expval_sxsy.pdf')

# =============================================================================
#u, v = np.linspace(0, 2*np.pi, 200), np.linspace(0, np.pi, 200)
#U, V = np.meshgrid(u, v)
#X = np.cos(U)*np.sin(V)
#Y = np.sin(U)*np.sin(V)
#Z = np.cos(V)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(X, Y, Z)
#ax.axis('equal')
#ax.set_aspect('equal', 'box')
#fig.savefig('spinevol_block1.pdf')
