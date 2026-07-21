#%% Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
import torch
import torch.nn as nn
from tqdm.auto import tqdm

#%% Initialise system
np.random.seed(42)
torch.manual_seed(42)

masses = torch.tensor([3, 2], dtype=torch.float)
positions = torch.tensor([[8, 0, 0], [-8, 0, 0]], dtype=torch.float)
momenta = torch.tensor([[0.1, 0.25, 0], [0.2, -0.25, 0]], dtype=torch.float)

npz_3D = np.load('TwoPunctures_32_3D.npz')
coords = npz_3D['coords']
u = npz_3D['u']
X_test_idx = np.random.choice(np.shape(coords)[0], 10000, replace=False)
X_test = coords[X_test_idx, :]
y_test = u[X_test_idx]
X_test = torch.tensor(X_test, dtype=torch.float)

npz_2D = np.load('TwoPunctures_32_2D.npz')
coords_2D = npz_2D['coords']
u_2D = npz_2D['u'].flatten()
coords_2D_idx = np.where(coords_2D[:, 2]==0)[0]
coords_2D = coords_2D[coords_2D_idx, :]
u_2D = u_2D[coords_2D_idx]
coords_2D = torch.tensor(coords_2D, dtype=torch.float)
u_2D = torch.tensor(u_2D, dtype=torch.float)

npz_1D = np.load('TwoPunctures_32_1D.npz')
coords_1D = npz_1D['coords']
u_1D = npz_1D['u']
coords_1D_idx = np.where((coords_1D[:, 1]==0) & (coords_1D[:, 2]==0))[0]
coords_1D = coords_1D[coords_1D_idx, :]
u_1D = u_1D[coords_1D_idx]
coords_1D = torch.tensor(coords_1D, dtype=torch.float)
u_1D = torch.tensor(u_1D, dtype=torch.float)

#%% Physics
def KBar(X):
    n_points = X.shape[0]
    delta = torch.eye(3, dtype=torch.float).view(1, 3, 3)
    kbar = torch.zeros(n_points, 3, 3, dtype=torch.float)

    for n in range(0, 2):
        xn = X - positions[n]
        xi = xn.view(n_points, 3, 1)
        xj = xn.view(n_points, 1, 3)
        rn = torch.sum(xn.pow(2), dim=1, dtype=torch.float, keepdim=True).pow(1/2).clamp_min(1e-12)
        
        Pn = momenta[n]
        Pi = Pn.view(1, 3, 1)
        Pj = Pn.view(1, 1, 3)

        P_dot_X = torch.sum(xn * Pn.view(1, 3), dim=1, dtype=torch.float, keepdim=True).view(n_points, 1, 1)

        momentum_term = xi * Pj + xj * Pi - (delta - xi * xj / rn.pow(2).view(n_points, 1, 1)) * P_dot_X

        kbar = kbar + 3 / (2 * rn.pow(3).view(n_points, 1, 1)) * momentum_term

    return kbar

def PsiSingular(X):
    psi = torch.ones(X.shape[0], 1, dtype=torch.float)
    for n in range(0, 2):
        rn = torch.sum((X - positions[n]).pow(2), dim=1, dtype=torch.float, keepdim=True).pow(1/2).clamp_min(1e-12)
        psi = psi + masses[n] / (2 * rn)
    
    return psi

def Grad(outputs, inputs, create_graph=True):
    derivative = torch.autograd.grad(
        outputs = outputs,
        inputs = inputs,
        grad_outputs = torch.ones_like(outputs),
        create_graph = create_graph,
        retain_graph = create_graph
    )[0]

    return derivative

def Laplacian(u, X):
    derivative = Grad(u, X)
    fx = derivative[:, 0:1]
    fy = derivative[:, 1:2]
    fz = derivative[:, 2:3]

    fxx = Grad(fx, X)[:, 0:1]
    fyy = Grad(fy, X)[:, 1:2]
    fzz = Grad(fz, X)[:, 2:3]

    return fxx + fyy + fzz

#%% Kappa
def SobolVolume(n_points=5e6, radius=30):
    engine = torch.quasirandom.SobolEngine(3, True, seed=42)
    u = engine.draw(int(n_points)).clamp(1e-12, 1 - 1e-12)
    r = radius * u[:, 0:1].pow(1/3)
    cos_theta = 2 * u[:, 1:2] - 1
    sin_theta = (1 - cos_theta.pow(2)).pow(1/2).clamp_min(0)
    phi = 2 * np.pi * u[:, 2:3]

    x = r * sin_theta * torch.cos(phi)
    y = r * sin_theta * torch.sin(phi)
    z = r * cos_theta

    return torch.cat([x, y, z], dim=1)

def SobolSurface(n_points=5e4, radius=30):
    engine = torch.quasirandom.SobolEngine(2, True, seed=42)
    u = engine.draw(int(n_points)).clamp(1e-12, 1 - 1e-12)
    
    cos_theta = 2 * u[:, 0:1] - 1
    sin_theta = (1 - cos_theta.pow(2)).pow(1/2).clamp_min(0)
    phi = 2 * np.pi * u[:, 1:2]

    x = radius * sin_theta * torch.cos(phi)
    y = radius * sin_theta * torch.sin(phi)
    z = radius * cos_theta

    return torch.cat([x, y, z], dim=1)

def CalcUg(X):
    total = torch.zeros(X.shape[0], 1, dtype=torch.float)

    def u0P_u2P_stable(curly_R):
        R = curly_R

        u0 = torch.empty_like(R)
        u2 = torch.empty_like(R)

        small = R < 1e-1
        large = ~small

        # Small-R Taylor expansions
        Rs = R[small]
        if Rs.numel() > 0:
            u0[small] = (1/32 - Rs.pow(5)/32 + 5*Rs.pow(6)/32 - 15*Rs.pow(7)/32 + 35*Rs.pow(8)/32 - 35*Rs.pow(9)/16 + 63*Rs.pow(10)/16 - 105*Rs.pow(11)/16 + 165*Rs.pow(12)/16 - 495*Rs.pow(13)/32 + 715*Rs.pow(14)/32)
            u2[small] = (Rs.pow(2)/400 - Rs.pow(5)/32 + 7*Rs.pow(6)/48 - 21*Rs.pow(7)/50 + 21*Rs.pow(8)/22 - 15*Rs.pow(9)/8 + 693*Rs.pow(10)/208 - 11*Rs.pow(11)/2 + 429*Rs.pow(12)/50 - 819*Rs.pow(13)/64 + 5005*Rs.pow(14)/272)

            # u0[small] = (1/32 - Rs.pow(5)/32 + 5*Rs.pow(6)/32 - 15*Rs.pow(7)/32)
            # u2[small] = (Rs.pow(2)/400 - Rs.pow(5)/32 + 7*Rs.pow(6)/48 - 21*Rs.pow(7)/50)

        # Direct expression away from the puncture
        Rl = R[large]
        if Rl.numel() > 0:
            l = 1 / (1 + Rl)
            u0[large] = (5/32* (l - 2*l.pow(2) + 2*l.pow(3) - l.pow(4) + 0.2*l.pow(5)))
            u2[large] = (15*l + 132*l.pow(2) + 53*l.pow(3) + 96*l.pow(4) + 82*l.pow(5) + 84*l.pow(5)/Rl + 84*torch.log(l)/Rl.pow(2)) / (80 * Rl)

        return u0, u2

    for n in range(0, 2):
        m = masses[n]
        P = momenta[n]
        P_mag = torch.linalg.norm(P)

        if P_mag.item() == 0:
            continue

        xn = X - positions[n]
        rn = torch.sum(xn.pow(2), dim=1, dtype=torch.float, keepdim=True).pow(1/2).clamp_min(1e-12)
        
        curly_R = (2 * rn / m).clamp_min(1e-12)
        l = 1 / (1 + curly_R)

        r_hat = xn / rn
        P_hat = P / P_mag
        mu_P = torch.sum(r_hat * P_hat.view(1, 3), dim=1, keepdim=True).clamp(-1, 1)
        P2 = 0.5 * (3 * mu_P.pow(2) - 1)

        # u0P = (5/32) * (l-2*l.pow(2) + 2*l.pow(3) - l.pow(4) + 0.2*l.pow(5))
        # u2P = (15*l + 132*l.pow(2) + 53*l.pow(3) + 96*l.pow(4) + 82*l.pow(5) + 84*l.pow(5)/curly_R + 84*torch.log(l)/curly_R.pow(2)) / (80 * curly_R)

        u0P, u2P = u0P_u2P_stable(curly_R)

        curly_P = 2 * P_mag / m
        total = total + curly_P.pow(2) * (u0P + u2P * P2)

    return total

def WindowFunction(X, ug_min, ug_max):
    ug = CalcUg(X)

    return (ug - ug_min) / (ug_max - ug_min)

def EstimateKappa():
    kappa_boundary = SobolSurface()
    kappa_boundary = kappa_boundary.clone().detach().requires_grad_(True)
    ug_boundary = CalcUg(kappa_boundary)
    grad_ug = Grad(ug_boundary, kappa_boundary, create_graph=False).detach()
    
    with torch.no_grad():
    
        n = kappa_boundary / 30
        area = 4 * np.pi * 30**2
        boundary_integrand = torch.linalg.vecdot(grad_ug, n).unsqueeze(1)
        boundary_integral = (area * boundary_integrand.mean()).detach()

        kappa_interior = SobolVolume()
        ug_interior = CalcUg(kappa_interior)
        psi = PsiSingular(kappa_interior)
        k_bar = KBar(kappa_interior)
        k2 = k_bar.pow(2).sum(dim=(1, 2)).unsqueeze(1)
        volume = 4/3 * np.pi * 30**3

        def f(guess_kappa):
            volume_integrand = 1/8 * k2 * (psi + guess_kappa * ug_interior).clamp_min(1e-12).pow(-7)
            volume_integral = (volume * volume_integrand.mean()).detach()

            return volume_integral + guess_kappa * boundary_integral        
        
        kappa_high = 1000
        kappa_low = 0

        f_low = f(kappa_low)
        f_high = f(kappa_high)

        if f_high*f_low > 0:
            return print('Does not bracket root')

        while kappa_high - kappa_low > 1e-6:
            kappa_mid = (kappa_high + kappa_low) / 2
            f_high = f(kappa_high)
            f_mid = f(kappa_mid)
            if f_mid == 0:
                break

            if f_mid * f_high < 0:
                kappa_low = kappa_mid
            else:
                kappa_high = kappa_mid

    return kappa_mid

#%% Loss functions
def Ansatz(h_theta, X, kappa, ug_min, ug_max, c_mag=1):
    ug = CalcUg(X)
    W = WindowFunction(X, ug_min, ug_max)
    u_theta = kappa * ug * (1 + c_mag * W * torch.tanh(h_theta))
    
    return u_theta

def PdeResidual(u_theta, X):
    lap_u = Laplacian(u_theta, X)
    psi = PsiSingular(X) + u_theta
    k_bar = KBar(X)
    k2 = k_bar.pow(2).sum(dim=(1, 2)).unsqueeze(1)

    return lap_u + 1/8 * psi.clamp_min(1e-12).pow(-7) * k2

def BoundaryResidual(u_theta, X):
    grad_u = Grad(u_theta, X)
    n = X / 30

    return torch.linalg.vecdot(grad_u, n).unsqueeze(1) + u_theta / 30

def SoftLinfLoss(res, beta=10):
    abs_res = res.abs().reshape(-1)
    n = len(abs_res)

    return (torch.logsumexp(beta * abs_res, dim=0) - np.log(n)) / beta

def RawLoss(u_theta_int, u_theta_bound, X_int, X_bound):
    res_int = PdeResidual(u_theta_int, X_int)
    res_bound = BoundaryResidual(u_theta_bound, X_bound)

    L2 = torch.mean(res_int.pow(2))
    Linf = SoftLinfLoss(res_int)
    LBC = torch.mean(res_bound.pow(2))

    return L2, Linf, LBC

def ScaledTotalLoss(L2, Linf, LBC):
    global ema_L2, ema_Linf, ema_LBC
    alpha = 0.99

    with torch.no_grad():
        if ema_L2 is None:
            ema_L2 = L2.detach()
            ema_Linf = Linf.detach()
            ema_LBC = LBC.detach()
        else:
            ema_L2 = alpha * ema_L2 + (1 - alpha) * L2.detach()
            ema_Linf = alpha * ema_Linf + (1 - alpha) * Linf.detach()
            ema_LBC = alpha * ema_LBC + (1 - alpha) * LBC.detach()

    L2_tilde = L2 / (ema_L2)
    Linf_tilde = Linf / (ema_Linf)
    LBC_tilde = LBC / (ema_LBC)

    return w2 * L2_tilde + w_inf * Linf_tilde + w_rob * LBC_tilde

#%% Collocation points
def GenXint(n_points=20000):
    directions = torch.randn(n_points, 3, dtype=torch.float)
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1e-12)
    radii = 30 * torch.rand(n_points, 1, dtype=torch.float)

    return radii * directions

def GenXbound(n_points=8000):
    directions = torch.randn(n_points, 3, dtype=torch.float)
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1e-12)

    return 30 * directions

#%% Setup (Do not run)
model = nn.Sequential(
    nn.Linear(3, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

kappa = EstimateKappa()

X_int = GenXint().requires_grad_(True)
X_bound = GenXbound().requires_grad_(True)

X_train = torch.concat([X_int, X_bound])
ug_train = CalcUg(X_train)
with torch.no_grad():
    ug_min = ug_train.detach().min()
    ug_max = ug_train.detach().max()

ema_L2 = None
ema_Linf = None
ema_LBC = None

w2 = 1
w_inf = 0.5
w_rob = 1

L2_list = []
Linf_list = []
LBC_list = []
total_loss_list = []
L2RE_list = []

start_epoch = 1

#%% Training code
n_epoch = 5000

for epoch in tqdm(range(start_epoch, start_epoch+n_epoch)):
    model.train()

    h_theta_int = model(X_int)
    u_theta_int = Ansatz(h_theta_int, X_int, kappa, ug_min, ug_max)

    h_theta_bound = model(X_bound)
    u_theta_bound = Ansatz(h_theta_bound, X_bound, kappa, ug_min, ug_max)

    L2, Linf, LBC = RawLoss(u_theta_int, u_theta_bound, X_int, X_bound)
    L2_list.append(L2.detach())
    Linf_list.append(Linf.detach())
    LBC_list.append(LBC.detach())

    total_loss = ScaledTotalLoss(L2, Linf, LBC)
    total_loss_list.append(total_loss.detach())

    model.eval()
    with torch.no_grad():
        h_theta_test = model(X_test)
        u_theta_test = Ansatz(h_theta_test, X_test, kappa, ug_min, ug_max)

        u_theta_test = u_theta_test.detach().numpy().flatten()
        L2RE = np.sqrt(((u_theta_test - y_test.flatten())**2).sum() / (y_test.flatten()**2).sum())
        L2RE_list.append(L2RE)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

print('Training finished')

#%% Plot losses
epochs = np.linspace(1, len(total_loss_list), len(total_loss_list))

fig1, ax1 = plt.subplots(1,1,figsize = (10,4),dpi = 150)
ax1.plot(epochs, total_loss_list, label='Scaled total loss', zorder=2)
ax1.set_xlabel('Epoch',fontsize = 16)
ax1.set_ylabel('Loss',fontsize = 16)
# ax1.set_yscale('log')
ax1.set_title('Loss during training',fontsize = 20)
ax1.tick_params(labelsize=12, which='both',top=True, right = True, direction='in')
ax1.grid(color='xkcd:dark blue',alpha = 0.2)
ax1.legend(loc='upper right',fontsize = 12)
plt.show()

total_loss = np.array(L2_list) + np.array(Linf_list) + np.array(LBC_list)

fig2, ax2 = plt.subplots(1,1,figsize = (10,4),dpi = 150)
ax2.plot(epochs, L2_list, label='L2', zorder=1)
ax2.plot(epochs, Linf_list, label='L_inf', zorder=1)
ax2.plot(epochs, LBC_list, label='LBC', zorder=1)
# ax2.plot(epochs, total_loss, label='Total loss', zorder=2)
ax2.set_xlabel('Epoch',fontsize = 16)
ax2.set_ylabel('Loss',fontsize = 16)
ax2.set_yscale('log')
ax2.set_title('Loss during training',fontsize = 20)
ax2.tick_params(labelsize=12, which='both',top=True, right = True, direction='in')
ax2.grid(color='xkcd:dark blue',alpha = 0.2)
ax2.legend(loc='right',fontsize = 12)
plt.show()

fig3, ax3 = plt.subplots(1,1,figsize = (10,4),dpi = 150)
ax3.plot(epochs, L2RE_list, label='L2RE', zorder=2)
ax3.set_xlabel('Epoch',fontsize = 16)
ax3.set_ylabel('Loss',fontsize = 16)
# ax3.set_yscale('log')
ax3.set_title('Loss during training',fontsize = 20)
ax3.tick_params(labelsize=12, which='both',top=True, right = True, direction='in')
ax3.grid(color='xkcd:dark blue',alpha = 0.2)
ax3.legend(loc='upper right',fontsize = 12)
plt.show()

#%% Plot 2D comparison
fig4, ax4 = plt.subplots(1,1,figsize = (9,6),dpi = 150)
heatmap4 = ax4.scatter(coords_2D[:, 0],  coords_2D[:, 1], c=u_2D, cmap='plasma')
ax4.set_aspect('equal', adjustable='box')
cbar4 = fig4.colorbar(heatmap4)
cbar4.set_label('$u_{TP}$', fontsize =16)
cbar4.ax.tick_params(labelsize=12)
ax4.set_xlabel('X', fontsize = 16)
ax4.set_ylabel('Y', fontsize = 16, rotation=0)
ax4.set_xlim(-31, 31)
ax4.set_ylim(-31, 31)
ax4.set_title('$u_{TP}$ in XY-plane',fontsize = 20)
ax4.tick_params(labelsize=12, which='both',top=True, right = True, direction='out')
fig4.tight_layout()
plt.show()

model.eval()
with torch.no_grad():
    h_theta_2D = model(coords_2D)
    u_theta_2D = Ansatz(h_theta_2D, coords_2D, kappa, ug_min, ug_max)

u_theta_2D = u_theta_2D.detach().flatten()

fig5, ax5 = plt.subplots(1,1,figsize = (8,6),dpi = 150)
heatmap5 = ax5.scatter(coords_2D[:, 0],  coords_2D[:, 1], c=u_theta_2D, cmap='plasma')
ax5.set_aspect('equal', adjustable='box')
cbar5 = fig5.colorbar(heatmap5)
cbar5.set_label('$u_{\\theta}$', fontsize =16)
cbar5.ax.tick_params(labelsize=12)
ax5.set_xlabel('X', fontsize = 16)
ax5.set_ylabel('Y', fontsize = 16, rotation=0)
ax5.set_xlim(-31, 31)
ax5.set_ylim(-31, 31)
ax5.set_title('Predicted $u_{\\theta}$ in XY-plane',fontsize = 20)
ax5.tick_params(labelsize=12, which='both',top=True, right = True, direction='out')
fig5.tight_layout()
plt.show()

norm6 = colors.TwoSlopeNorm(vmin=-(abs(u_theta_2D-u_2D).max()), vcenter=0.0, vmax=abs(u_theta_2D-u_2D).max())
fig6, ax6 = plt.subplots(1,1,figsize = (8,6),dpi = 160)
heatmap6 = ax6.scatter(coords_2D[:, 0],  coords_2D[:, 1], c=u_theta_2D-u_2D, norm=norm6, cmap='RdBu')
ax6.set_aspect('equal', adjustable='box')
cbar6 = fig6.colorbar(heatmap6)
cbar6.ax.tick_params(labelsize=12)
ax6.set_xlabel('X', fontsize = 16)
ax6.set_ylabel('Y', fontsize = 16, rotation=0)
ax6.set_xlim(-31, 31)
ax6.set_ylim(-31, 31)
ax6.set_title('$u_{\\theta} - u_{TP}$',fontsize = 20)
ax6.tick_params(labelsize=12, which='both',top=True, right = True, direction='out')
fig6.tight_layout()
plt.show()

#%% Plot 1D comparison
model.eval()
with torch.no_grad():
    h_theta_1D = model(coords_1D)
    u_theta_1D = Ansatz(h_theta_1D, coords_1D, kappa, ug_min, ug_max)

u_theta_1D = u_theta_1D.detach().flatten()

fig7, ax7 = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
ax7.grid(alpha=0.3)
ax7.plot(coords_1D[:, 0], u_1D, '-', label='$u_{TP}$')
ax7.plot(coords_1D[:, 0], u_theta_1D, '--', label='$u_{\\theta}$')
ax7.set_xlabel('X', fontsize=12)
ax7.set_ylabel('u', fontsize=12, rotation=0)
ax7.set_xlim(-30, 30)
# ax7.set_ylim(0)
ax7.set_title('Gravitational Correction to $\\Psi$ from TwoPunctures', fontsize=16)
ax7.legend(loc='upper right', fontsize=10)
# ax7.xaxis.set_major_locator(MultipleLocator(10))
# ax7.xaxis.set_minor_locator(MultipleLocator(2))
# ax7.yaxis.set_major_locator(MultipleLocator(0.0005))
# ax7.yaxis.set_minor_locator(MultipleLocator(0.0001))
ax7.tick_params(axis='x', labelsize=10)
ax7.tick_params(axis='y', labelsize=10)
fig7.tight_layout()
plt.show()

#%% Save checkpoint
checkpoint = {
    "model_state_dict": model.state_dict(),
    "X_int": X_int.detach(),
    "X_bound": X_bound.detach(),
    "ug_min":ug_min,
    "ug_max":ug_max,

    "kappa": kappa,

    "ema_L2": ema_L2,
    "ema_Linf": ema_Linf,
    "ema_LBC": ema_LBC,

    "w2": w2,
    "w_inf": w_inf,
    "w_rob": w_rob,

    "L2_list": L2_list,
    "Linf_list": Linf_list,
    "LBC_list": LBC_list,
    "total_loss_list": total_loss_list,
    "L2RE_list":L2RE_list
}

torch.save(checkpoint, "Not_Simple_BBH_Checkpoint.pt")

print('Checkpoint saved')

#%% Load checkpoint
checkpoint = torch.load("Not_Simple_BBH_Checkpoint.pt", weights_only=False)

model = nn.Sequential(
    nn.Linear(3, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 1)
)
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

kappa = checkpoint["kappa"]

X_int = checkpoint["X_int"].requires_grad_(True)
X_bound = checkpoint["X_bound"].requires_grad_(True)
ug_min = checkpoint["ug_min"]
ug_max = checkpoint["ug_max"]

ema_L2 = checkpoint["ema_L2"]
ema_Linf = checkpoint["ema_Linf"]
ema_LBC = checkpoint["ema_LBC"]

w2 = checkpoint["w2"]
w_inf = checkpoint["w_inf"]
w_rob = checkpoint["w_rob"]

L2_list = checkpoint["L2_list"]
Linf_list = checkpoint["Linf_list"]
LBC_list = checkpoint["LBC_list"]
total_loss_list = checkpoint["total_loss_list"]
L2RE_list = checkpoint["L2RE_list"]

start_epoch = len(total_loss_list) + 1

print('Checkpoint loaded')
print('Start epoch: {}'.format(start_epoch))

#%%
