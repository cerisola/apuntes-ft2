import numpy as np
import scipy as sp

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.lines as mlines
import matplotlib.cm as cmaps
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import cmocean

mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.unicode'] = True # default in 3.0+
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['legend.fontsize'] = 9

figsize = (3.0, 3.0)
default_pad = 0.2

# =============================================================================
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# =============================================================================
hbar = 1
m = 1
omega = 1

Norm = np.sqrt((m*omega)/(hbar*np.pi))
Exp = (m*omega)/(hbar)
sdv = np.sqrt(hbar/(2*m*omega))

Np = 5000
x0 = round(10*sdv)
DX = round(7*sdv)
x = np.linspace(-x0 - DX, x0 + DX, Np)

psi0 = lambda x, xc: Norm*np.exp(-Exp*(x - xc)**2)

xc = np.linspace(-x0, x0, 5, endpoint=True)
psi0t = np.zeros((x.size, xc.size))
for i, xci in enumerate(xc):
  psi0t[:, i] = psi0(x, xci)

xtickpos = [-x0, 0, x0]
xticklbl = ['$-x_0$', '$0$', '$x_0$']
root_color = 'black'
styles_root = ['-', '--', '-.']
colors = [None]*xc.size
styles = [None]*xc.size
for idx in range(xc.size):
  #colors[idx] = lighten_color(root_color, 1.0 - idx*0.15)
  colors[idx] = cmocean.cm.algae_r(idx/(xc.size))
  #colors[idx] = cmaps.magma(idx/(xc.size-1))
  #styles[idx] = styles_root[idx % len(styles_root)]
  styles[idx] = '-'

# =============================================================================
plt.figure(figsize=figsize)

curves = [None]*xc.size
for t in (range(xc.size)):
  curves[t] = plt.plot(x, psi0t[:,t], styles[t], color=colors[t], label=f'$t_{t}$')

#plt.grid()
plt.xlabel('Posición $x$')
plt.ylabel('Densidad de probabilidad $P(x)$')
plt.xticks(xtickpos, xticklbl)
plt.ylim(-0.02, 0.62)
plt.legend(handlelength=1.0, handletextpad=0.5)
plt.tight_layout(pad=default_pad)
plt.savefig('coherent_oscillation.pdf')

# =============================================================================

#fig = plt.figure(figsize=figsize)
fig = plt.figure()
ax = fig.gca(projection='3d')

curves = [None]*xc.size
tlbls = [None]*xc.size
tpos = [None]*xc.size
for t in (range(xc.size)):
  tlbl = f'$t_{t}$'
  curves[t] = ax.plot(x, psi0t[:,t], zs=t, zdir='x', label=tlbl)
  tlbls[t] = tlbl
  tpos[t] = t

ax.set_xlabel('Tiempo $t$')
ax.set_ylabel('Posición $x$')
ax.set_zlabel('Densidad de probabilidad $P(x)$')

#ax.set_xlim(-0.02, 0.62)
ax.set_xticks(tpos)
ax.set_xticklabels(tlbls)
ax.set_yticks(xtickpos)
ax.set_yticklabels(xticklbl)

ax.view_init(elev=30., azim=-20)
#fig.legend(handlelength=1.0, handletextpad=0.5)
#fig.tight_layout(pad=default_pad)
fig.tight_layout()
#fig.savefig('coherent_oscillation_3d.pdf')
fig.show()
