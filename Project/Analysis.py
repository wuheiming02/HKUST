#%% Import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch

# Generate datasets
np.random.seed(42)
torch.manual_seed(42)

def GenBranchSamples(m1):
    m2 = 1 - m1
    x1 = torch.full_like(m1, 3.0)
    y1 = torch.zeros_like(m1)
    z1 = torch.zeros_like(m1)
    x2 = torch.full_like(m1, -3.0)
    y2 = torch.zeros_like(m1)
    z2 = torch.zeros_like(m1)
    P1x = torch.zeros_like(m1)
    P1y = 0.8 * m1 * m2
    P1z = torch.zeros_like(m1)
    P2x = torch.zeros_like(m1)
    P2y = -0.8 * m1 * m2
    P2z = torch.zeros_like(m1)

    return torch.hstack([m1, m2, x1, y1, z1, x2, y2, z2, P1x, P1y, P1z, P2x, P2y, P2z])
npz_15_3D = np.load('data/TP15_3D.npz')
coords_15_3D = npz_15_3D['coords']
u_15_3D = npz_15_3D['u']
params_15_3D = GenBranchSamples(torch.tensor([[0.15]] * len(u_15_3D), dtype=torch.float).reshape(-1, 1))

npz_15_2D = np.load('data/TP15_2D.npz')
coords_15_2D = npz_15_2D['coords']
u_15_2D = npz_15_2D['u']
params_15_2D = GenBranchSamples(torch.tensor([[0.15]] * len(u_15_2D), dtype=torch.float).reshape(-1, 1))

npz_15_1D = np.load('data/TP15_1D.npz')
coords_15_1D = npz_15_1D['coords']
u_15_1D = npz_15_1D['u']
params_15_1D = GenBranchSamples(torch.tensor([[0.15]] * len(u_15_1D), dtype=torch.float).reshape(-1, 1))

npz_25_3D = np.load('data/TP25_3D.npz')
coords_25_3D = npz_25_3D['coords']
u_25_3D = npz_25_3D['u']
params_25_3D = GenBranchSamples(torch.tensor([[0.25]] * len(u_25_3D), dtype=torch.float).reshape(-1, 1))

npz_25_2D = np.load('data/TP25_2D.npz')
coords_25_2D = npz_25_2D['coords']
u_25_2D = npz_25_2D['u']
params_25_2D = GenBranchSamples(torch.tensor([[0.25]] * len(u_25_2D), dtype=torch.float).reshape(-1, 1))

npz_25_1D = np.load('data/TP25_1D.npz')
coords_25_1D = npz_25_1D['coords']
u_25_1D = npz_25_1D['u']
params_25_1D = GenBranchSamples(torch.tensor([[0.25]] * len(u_25_1D), dtype=torch.float).reshape(-1, 1))

npz_35_3D = np.load('data/TP35_3D.npz')
coords_35_3D = npz_35_3D['coords']
u_35_3D = npz_35_3D['u']
params_35_3D = GenBranchSamples(torch.tensor([[0.35]] * len(u_35_3D), dtype=torch.float).reshape(-1, 1))

npz_35_2D = np.load('data/TP35_2D.npz')
coords_35_2D = npz_35_2D['coords']
u_35_2D = npz_35_2D['u']
params_35_2D = GenBranchSamples(torch.tensor([[0.35]] * len(u_35_2D), dtype=torch.float).reshape(-1, 1))

npz_35_1D = np.load('data/TP35_1D.npz')
coords_35_1D = npz_35_1D['coords']
u_35_1D = npz_35_1D['u']
params_35_1D = GenBranchSamples(torch.tensor([[0.35]] * len(u_35_1D), dtype=torch.float).reshape(-1, 1))

npz_45_3D = np.load('data/TP45_3D.npz')
coords_45_3D = npz_45_3D['coords']
u_45_3D = npz_45_3D['u']
params_45_3D = GenBranchSamples(torch.tensor([[0.45]] * len(u_45_3D), dtype=torch.float).reshape(-1, 1))

npz_45_2D = np.load('data/TP45_2D.npz')
coords_45_2D = npz_45_2D['coords']
u_45_2D = npz_45_2D['u']
params_45_2D = GenBranchSamples(torch.tensor([[0.45]] * len(u_45_2D), dtype=torch.float).reshape(-1, 1))

npz_45_1D = np.load('data/TP45_1D.npz')
coords_45_1D = npz_45_1D['coords']
u_45_1D = npz_45_1D['u']
params_45_1D = GenBranchSamples(torch.tensor([[0.45]] * len(u_45_1D), dtype=torch.float).reshape(-1, 1))

branch_test = torch.concat([params_15_1D, params_15_2D, params_15_3D, params_25_1D, params_25_2D, params_25_3D, params_35_1D, params_35_2D, params_35_3D, params_45_1D, params_45_2D, params_45_3D])
trunk_test = np.concat([coords_15_1D, coords_15_2D, coords_15_3D, coords_25_1D, coords_25_2D, coords_25_3D, coords_35_1D, coords_35_2D, coords_35_3D, coords_45_1D, coords_45_2D, coords_45_3D])
u_test = np.concat([u_15_1D, u_15_2D, u_15_3D, u_25_1D, u_25_2D, u_25_3D, u_35_1D, u_35_2D, u_35_3D, u_45_1D, u_45_2D, u_45_3D])

trunk_test = torch.tensor(trunk_test, dtype=torch.float)
u_test = torch.tensor(u_test, dtype=torch.float).reshape(-1, 1)

n_1D = 5999
n_2D = 282683
n_3D = 7236575
n_batch = n_1D + n_2D + n_3D
m1 = [0.15, 0.25, 0.35, 0.45]

u_test = u_test.detach().flatten()
#
checkpoint = torch.load("Full_PIDONet_Checkpoint4.pt", weights_only=False, map_location=torch.device('cpu'))
u_test_pred = checkpoint['u_pred_test_list']
u_test_pred = torch.concat(u_test_pred).detach().flatten()

L_data_list = checkpoint["L_data_list"]
L2_list = checkpoint["L2_list"]
L_inf_list = checkpoint["L_inf_list"]
LBC_list = checkpoint["LBC_list"]
total_loss_list = checkpoint["total_loss_list"]
L2RE_list = checkpoint["L2RE_list"]
    
# Plot losses
epochs = np.linspace(1, len(L_data_list), len(L_data_list))

fig1, ax1 = plt.subplots(1, 1, figsize=(10,4), dpi=150)
ax1.plot(epochs, total_loss_list, label='Scaled total loss', zorder=2)
ax1.set_xlabel('Epoch',fontsize = 16)
ax1.set_ylabel('Loss',fontsize = 16)
# ax1.set_yscale('log')
ax1.set_title('Loss during training',fontsize = 20)
ax1.tick_params(labelsize=12, which='both',top=True, right = True, direction='in')
ax1.grid(color='xkcd:dark blue',alpha = 0.2)
ax1.legend(loc='upper right',fontsize = 12)
plt.show()

total_loss = np.array(L_data_list) + np.array(L2_list) + np.array(L_inf_list) + np.array(LBC_list)

fig2, ax2 = plt.subplots(1, 1, figsize=(10,4), dpi=150)
ax2.plot(epochs, L_data_list, label='$L_{data}$', zorder=1)
ax2.plot(epochs, L2_list, label='$L_{2}$', zorder=1)
ax2.plot(epochs, L_inf_list, label='soft-$L_{inf}$', zorder=1)
ax2.plot(epochs, LBC_list, label='$L_{BC}$', zorder=1)
ax2.set_xlabel('Epoch',fontsize = 16)
ax2.set_ylabel('Loss',fontsize = 16)
ax2.set_yscale('log')
ax2.set_title('Loss during training',fontsize = 20)
ax2.tick_params(labelsize=12, which='both',top=True, right = True, direction='in')
ax2.grid(color='xkcd:dark blue',alpha = 0.2)
ax2.legend(fontsize = 12)
plt.show()

fig3, ax3 = plt.subplots(1, 1, figsize=(10,4), dpi=150)
ax3.plot(epochs, L2RE_list, label='L2RE', zorder=2)
ax3.set_xlabel('Epoch',fontsize = 16)
ax3.set_ylabel('Loss',fontsize = 16)
ax3.set_title('Validation loss',fontsize = 20)
ax3.tick_params(labelsize=12, which='both',top=True, right = True, direction='in')
ax3.grid(color='xkcd:dark blue',alpha = 0.2)
ax3.legend(loc='upper right',fontsize = 12)
plt.show()

# L2RE
print(torch.sqrt((u_test - u_test_pred).pow(2).sum() / u_test.pow(2).sum()))
L2RE_1D_list = []
L2RE_2D_list = []
L2RE_3D_list = []
for i in range (4):
    test_1D = u_test[n_batch*i:n_batch*i+n_1D]
    test_pred_1D = u_test_pred[n_batch*i:n_batch*i+n_1D]

    test_2D = u_test[n_batch*i+n_1D:n_batch*i+n_1D+n_2D]
    test_pred_2D = u_test_pred[n_batch*i+n_1D:n_batch*i+n_1D+n_2D]

    test_3D = u_test[n_batch*i+n_1D+n_2D:n_batch*i+n_1D+n_2D+n_3D]
    test_pred_3D = u_test_pred[n_batch*i+n_1D+n_2D:n_batch*i+n_1D+n_2D+n_3D]

    L2RE_1D = torch.sqrt((test_1D - test_pred_1D).pow(2).sum() / test_1D.pow(2).sum())
    L2RE_2D = torch.sqrt((test_2D - test_pred_2D).pow(2).sum() / test_2D.pow(2).sum())
    L2RE_3D = torch.sqrt((test_3D - test_pred_3D).pow(2).sum() / test_3D.pow(2).sum())

    L2RE_1D_list.append(L2RE_1D.item())
    L2RE_2D_list.append(L2RE_2D.item())
    L2RE_3D_list.append(L2RE_3D.item())

u_3D = torch.concat([
    u_test[n_batch*0+n_1D+n_2D:n_batch*0+n_1D+n_2D+n_3D],
    u_test[n_batch*1+n_1D+n_2D:n_batch*1+n_1D+n_2D+n_3D],
    u_test[n_batch*2+n_1D+n_2D:n_batch*2+n_1D+n_2D+n_3D],
    u_test[n_batch*3+n_1D+n_2D:n_batch*3+n_1D+n_2D+n_3D]
]).detach().flatten()

u_pred_3D = torch.concat([
    u_test_pred[n_batch*0+n_1D+n_2D:n_batch*0+n_1D+n_2D+n_3D],
    u_test_pred[n_batch*1+n_1D+n_2D:n_batch*1+n_1D+n_2D+n_3D],
    u_test_pred[n_batch*2+n_1D+n_2D:n_batch*2+n_1D+n_2D+n_3D],
    u_test_pred[n_batch*3+n_1D+n_2D:n_batch*3+n_1D+n_2D+n_3D]
]).detach().flatten()

print(torch.sqrt((u_3D - u_pred_3D).pow(2).sum() / u_3D.pow(2).sum()))

print(L2RE_3D_list)

# Visualisation
figures, axes = [], []

for i in range(4):
    fig4, ax4 = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    figures.append(fig4)
    axes.append(ax4)
    ax4.grid(alpha=0.3)
    ax4.plot(trunk_test[n_batch*i:n_batch*i+n_1D, 0], u_test[n_batch*i:n_batch*i+n_1D], '-', label='$u_{TP}$')
    ax4.plot(trunk_test[n_batch*i:n_batch*i+n_1D, 0], u_test_pred[n_batch*i:n_batch*i+n_1D], '--', label='$u_{\\theta}$')
    ax4.set_xlabel('X', fontsize=12)
    ax4.set_ylabel('u', fontsize=12, rotation=0)
    ax4.set_xlim(-30, 30)
    ax4.set_title(f'$m_+={m1[i]}$, $(\\text{{L2RE}}={L2RE_1D_list[i]:.4f})$', fontsize=16)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.tick_params(axis='x', labelsize=10)
    ax4.tick_params(axis='y', labelsize=10)
    fig4.tight_layout()
    plt.show()

    fig5, ax5 = plt.subplots(1, 1, figsize=(9,6), dpi=150)
    figures.append(fig5)
    axes.append(ax5)
    heatmap5 = ax5.scatter(trunk_test[n_batch*i+n_1D:n_batch*i+n_1D+n_2D, 0],  trunk_test[n_batch*i+n_1D:n_batch*i+n_1D+n_2D, 1], c=u_test[n_batch*i+n_1D:n_batch*i+n_1D+n_2D], cmap='plasma')
    ax5.grid(alpha=0.2)
    ax5.set_aspect('equal', adjustable='box')
    cbar5 = fig5.colorbar(heatmap5)
    cbar5.set_label('$u_{TP}$', fontsize =16)
    cbar5.ax.tick_params(labelsize=12)
    ax5.set_xlabel('X', fontsize = 16)
    ax5.set_ylabel('Y', fontsize = 16, rotation=0)
    ax5.set_xlim(-31, 31)
    ax5.set_ylim(-31, 31)
    ax5.set_title(f'$u_{{TP}}$ in XY-plane $(m_+={m1[i]})$',fontsize = 20)
    ax5.tick_params(labelsize=12, which='both',top=True, right = True, direction='out')
    fig5.tight_layout()
    plt.show()

    fig6, ax6 = plt.subplots(1, 1, figsize=(9,6), dpi=150)
    figures.append(fig6)
    axes.append(ax6)
    heatmap6 = ax6.scatter(trunk_test[n_batch*i+n_1D:n_batch*i+n_1D+n_2D, 0],  trunk_test[n_batch*i+n_1D:n_batch*i+n_1D+n_2D, 1], c=u_test_pred[n_batch*i+n_1D:n_batch*i+n_1D+n_2D], cmap='plasma')
    ax6.grid(alpha=0.2)
    ax6.set_aspect('equal', adjustable='box')
    cbar6 = fig6.colorbar(heatmap6)
    cbar6.set_label('$u_{TP}$', fontsize =16)
    cbar6.ax.tick_params(labelsize=12)
    ax6.set_xlabel('X', fontsize = 16)
    ax6.set_ylabel('Y', fontsize = 16, rotation=0)
    ax6.set_xlim(-31, 31)
    ax6.set_ylim(-31, 31)
    ax6.set_title(f'$u_{{\\theta}}$ in XY-plane $(m_+={m1[i]})$',fontsize = 20)
    ax6.tick_params(labelsize=12, which='both',top=True, right = True, direction='out')
    fig6.tight_layout()
    plt.show()

    norm7 = colors.TwoSlopeNorm(vmin=-(abs(u_test_pred[n_batch*i+n_1D:n_batch*i+n_1D+n_2D]-u_test[n_batch*i+n_1D:n_batch*i+n_1D+n_2D]).max()), vcenter=0.0, vmax=abs(u_test_pred[n_batch*i+n_1D:n_batch*i+n_1D+n_2D]-u_test[n_batch*i+n_1D:n_batch*i+n_1D+n_2D]).max())
    fig7, ax7 = plt.subplots(1, 1, figsize=(8,6), dpi=160)
    heatmap7 = ax7.scatter(trunk_test[n_batch*i+n_1D:n_batch*i+n_1D+n_2D, 0],  trunk_test[n_batch*i+n_1D:n_batch*i+n_1D+n_2D, 1], c=u_test_pred[n_batch*i+n_1D:n_batch*i+n_1D+n_2D]-u_test[n_batch*i+n_1D:n_batch*i+n_1D+n_2D], norm=norm7, cmap='RdBu_r')
    ax7.grid(alpha=0.2)
    ax7.set_aspect('equal', adjustable='box')
    cbar7 = fig7.colorbar(heatmap7)
    cbar7.ax.tick_params(labelsize=12)
    ax7.set_xlabel('X', fontsize = 16)
    ax7.set_ylabel('Y', fontsize = 16, rotation=0)
    ax7.set_xlim(-31, 31)
    ax7.set_ylim(-31, 31)
    ax7.set_title(f'$u_{{\\theta}} - u_{{TP}}$ ($m_+={m1[i]}$, $\\text{{L2RE}}={L2RE_2D_list[i]:.4f}$)',fontsize = 20)
    ax7.tick_params(labelsize=12, which='both',top=True, right = True, direction='out')
    fig7.tight_layout()
    plt.show()

#%%