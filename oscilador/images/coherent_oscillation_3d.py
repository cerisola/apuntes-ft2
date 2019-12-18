import numpy as np
import scipy as sp

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.lines as mlines
import matplotlib.cm as cmaps
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib.collections import PolyCollection
import cmocean

mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.unicode'] = True # default in 3.0+
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['legend.fontsize'] = 9

figsize = (3.5, 3.0)
default_pad = 0.2

# =============================================================================
def cc(arg):
  '''
  Shorthand to convert 'named' colors to rgba format at 60% opacity.
  '''
  return clr.to_rgba(arg, alpha=0.6)


def polygon_under_graph(xlist, ylist):
  '''
  Construct the vertex list which defines the polygon filling the space under
  the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
  '''
  return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]


def lighten_color(color, amount=0.5):
  '''
  Lightens the given color by multiplying (1-luminosity) by the given amount.
  Input can be matplotlib color string, hex string, or RGB tuple.

  Examples:
  >> lighten_color('g', 0.3)
  >> lighten_color('#F034A3', 0.6)
  >> lighten_color((.3,.55,.1), 0.5)
  '''
  import matplotlib.colors as mc
  import colorsys
  try:
    c = mc.cnames[color]
  except:
    c = color
  c = colorsys.rgb_to_hls(*mc.to_rgb(c))
  return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def cosspace(start, end, npoints):
  '''
  """
  COSSPACE Cosine spaced vector.
  COSSPACE(X1, X2, N) generates N cosine-spaced points between X1 and X2.

  This method of spacing concentrates samples at the ends while
  producing fewer sample points in the center.

             ** *  *   *    *    *   *  * * **
             1...                            N

  See also LOGSPACE, LINSPACE
  '''

  x = np.empty(npoints)
  x[0] = start
  x[-1] = end
  if end <= start:
    print('End point must be greater than the start point')
    return None

  mid = (end - start)/2

  # Calculate the iteration increment
  # Each point in the array corresponds to an angle.
  angleinc = np.pi/(npoints - 1)

  # Brute Force way...
  # For reference only.  The algorithm is easier to see this way,
  # but slower because it doesn't take advantage of symmetry.
  # curAngle=angleInc;
  # for idx = 2:numPoints-1
  #   x(idx)=startPoint+midPoint*(1-cos(curAngle));
  #   curAngle=curAngle+angleInc;
  # end
  # Alternative way (1/2 the "for loop" iterations)
  # The spacing before and after the midpoint is the same.
  # We can save some calculations by making use of this symmetry.
  curangle =  angleinc
  for idx in np.arange(1, int(np.ceil(npoints/2))):
    x[idx] = start + mid*(1 - np.cos(curangle))
    x[end - (idx-1)] = x[end-(idx-2)] - (x[idx] - x[idx-1])
    curangle = curangle + angleinc
  
  return x


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

#t = np.linspace(0, 1, 12, endpoint=True)
base = np.array([0, 1/24,  1/16, 1/12, 1/8, 1/6, 5/24, 1/4])
t = np.array(list(base) + list(1/2 - base) + list(1/2 + base) + list(1 - base))
xc = -x0*np.cos(2*np.pi*omega*t)

psi0t = np.zeros((x.size, xc.size))
for i, xci in enumerate(xc):
  psi0t[:, i] = psi0(x, xci)

xtickpos = [-x0, 0, x0]
xticklbl = ['$-x_0$', '$0$', '$x_0$']
tticklbl, ttickpos = [], []
for idx, tidx in enumerate(t):
  if (idx % 2) == 0: continue
  tticklbl.append(f'$t_{idx}$')
  ttickpos.append(tidx)

root_color = 'black'
styles_root = ['-', '--', '-.']
colors, styles = [], []
for idx in range(xc.size):
  colors.append(cmocean.cm.algae_r(idx/(xc.size)))
  #colors[idx] = cmaps.magma(idx/(xc.size-1))
  #styles[idx] = styles_root[idx % len(styles_root)]

# =============================================================================

fig = plt.figure(figsize=figsize)
ax = fig.gca(projection='3d')

verts = []
for idx, tidx in enumerate(t):
  verts.append(polygon_under_graph(x, psi0t[:,idx]))
  
poly = PolyCollection(verts, facecolors='white', edgecolors='black')
ax.add_collection3d(poly, zs=t, zdir='x')

for idx, tidx in enumerate(t):
  ax.plot(x, psi0t[:,idx], zs=tidx, zdir='x', color='black', linewidth=0.6, zorder=0)

ax.set_xlabel('Tiempo $\omega t / (2\pi)$', labelpad=0.15)
ax.set_ylabel('Posición $x$', labelpad=0.15)
ax.set_zlabel('Densidad de probabilidad $P(x)$', labelpad=0.15)

#ax.set_xticks(ttickpos)
#ax.set_xticklabels(tticklbl)
ax.set_yticks(xtickpos)
ax.set_yticklabels(xticklbl)
ax.tick_params(axis='x', which='both', pad=0.05)
ax.tick_params(axis='y', which='both', pad=0.05)
ax.tick_params(axis='z', which='both', pad=0.05)

ax.set_zlim(0, 0.8)

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

ax.view_init(elev=32, azim=-20)

fig.tight_layout(pad=0.5)
fig.subplots_adjust(left=-0.08, right=0.94, top=1.35, bottom=0.05)
fig.savefig('coherent_oscillation_3d.pdf')
#plt.show()
