#%%
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

npz_file = np.load('twopunctures_mplusfrac_0.500000_dx_0.250.npz')
coords = npz_file['coords']
u = npz_file['u']
x_coords = coords[:, 0]
y_coords = coords[:, 1]
z_coords = coords[:, 2]

x0_idx = np.where(((y_coords==0) & (z_coords==0)))[0]

interpolate = CubicSpline(x_coords[x0_idx], u[x0_idx])
X_plot = np.linspace(-30, 30, 1000)
u_plot = interpolate(X_plot)

fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
ax.grid(alpha=0.3)
ax.plot(x_coords[x0_idx], u[x0_idx], '.')
ax.plot(X_plot, u_plot, '--', label='interpolated data')
ax.vlines([-3, 3], 0, 0.02, linestyles='--', color='red', label='Puncture positions')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('u', fontsize=12)
ax.set_ylim(0, 0.02)
ax.set_title('Gravitational Correction to $\\Psi$ from TwoPunctures', fontsize=16)
ax.legend(loc='upper right', fontsize=10)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(0.002))
ax.yaxis.set_minor_locator(MultipleLocator(0.0005))
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
fig.tight_layout()
plt.show()
