#%% Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm.auto import tqdm

#%% Generate datasets
np.random.seed(42)
torch.manual_seed(42)

def GenBranchSamples(m1):
    N = m1.shape[0]
    m2 = 1 - m1
    x1 = torch.tensor([[3]] * N, dtype=torch.float)
    y1 = torch.zeros_like(m1)
    z1 = torch.zeros_like(m1)
    x2 = torch.tensor([[-3]] * N, dtype=torch.float)
    y2 = torch.zeros_like(m1)
    z2 = torch.zeros_like(m1)
    P1x = torch.zeros_like(m1)
    P1y = 0.8 * m1 * m2
    P1z = torch.zeros_like(m1)
    P2x = torch.zeros_like(m1)
    P2y = -0.8 * m1 * m2
    P2z = torch.zeros_like(m1)

    return torch.hstack([m1, m2, x1, y1, z1, x2, y2, z2, P1x, P1y, P1z, P2x, P2y, P2z])

npz_01 = np.load('TP01.npz')
coords_01 = npz_01['coords']
u_01 = npz_01['u']
params_01 = GenBranchSamples(torch.tensor([[0.1]] * len(u_01), dtype=torch.float).reshape(-1, 1))

npz_02 = np.load('TP02.npz')
coords_02 = npz_02['coords']
u_02 = npz_02['u']
params_02 = GenBranchSamples(torch.tensor([[0.2]] * len(u_02), dtype=torch.float).reshape(-1, 1))

npz_03 = np.load('TP03.npz')
coords_03 = npz_03['coords']
u_03 = npz_03['u']
params_03 = GenBranchSamples(torch.tensor([[0.3]] * len(u_03), dtype=torch.float).reshape(-1, 1))

npz_04 = np.load('TP04.npz')
coords_04 = npz_04['coords']
u_04 = npz_04['u']
params_04 = GenBranchSamples(torch.tensor([[0.4]] * len(u_04), dtype=torch.float).reshape(-1, 1))

npz_05 = np.load('TP05.npz')
coords_05 = npz_05['coords']
u_05 = npz_05['u']
params_05 = GenBranchSamples(torch.tensor([[0.5]] * len(u_05), dtype=torch.float).reshape(-1, 1))

branch_sample = torch.concat([params_01, params_02, params_03, params_04, params_05])
trunk_sample = np.concat([coords_01, coords_02, coords_03, coords_04, coords_05])
y_sample = np.concat([u_01, u_02, u_03, u_04, u_05])

trunk_sample = torch.tensor(trunk_sample, dtype=torch.float)
y_sample = torch.tensor(y_sample, dtype=torch.float)

dataset = TensorDataset(branch_sample, trunk_sample, y_sample)
dataset_train, dataset_validate = random_split(
    dataset = dataset,
    lengths = [0.8, 0.2],
    generator = torch.Generator().manual_seed(42)
)

dloader_data_train = DataLoader(
    dataset = dataset_train,
    batch_size = 1024,
    shuffle = True,
    drop_last = True
)

dloader_data_validate = DataLoader(
    dataset = dataset_validate,
    batch_size = 64,
    shuffle = True
)

def GenXint(n_points=1e6):
    directions = torch.randn(int(n_points), 3, dtype=torch.float)
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1e-12)
    radii = 10 * torch.rand(int(n_points), 1, dtype=torch.float)
    m1 = torch.rand(int(n_points), 1, dtype=torch.float) * 0.4 + 0.1
    branch_interior = GenBranchSamples(m1)

    return radii * directions, branch_interior

def GenXsur(n_points=1e6):
    directions = torch.randn(int(n_points), 3, dtype=torch.float)
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1e-12)
    radii = 20 * torch.rand(int(n_points), 1, dtype=torch.float) + 10
    m1 = torch.rand(int(n_points), 1, dtype=torch.float) * 0.4 + 0.1
    branch_sur = GenBranchSamples(m1)

    return radii * directions, branch_sur

def GenXbound(n_points=2e5):
    directions = torch.randn(int(n_points), 3, dtype=torch.float)
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1e-12)
    m1 = torch.rand(int(n_points), 1, dtype=torch.float) * 0.4 + 0.1
    branch_boundary = GenBranchSamples(m1)

    return 30 * directions, branch_boundary

trunk_int, branch_int = GenXint()
trunk_sur, branch_sur = GenXsur()
trunk_bound, branch_bound = GenXbound()

trunk_int = torch.concat([trunk_int, trunk_sur])
branch_int = torch.concat([branch_int, branch_sur])

dataset_int = TensorDataset(branch_int, trunk_int)

dloader_int = DataLoader(
    dataset = dataset_int,
    batch_size = 1024,
    shuffle = True,
    drop_last = True
)

dataset_bound = TensorDataset(branch_bound, trunk_bound)

dloader_bound = DataLoader(
    dataset = dataset_bound,
    batch_size = 1024,
    shuffle = True,
    drop_last = True
)

#%% Physics
def KBar(trunk, branch):
    x1 = branch[:, 2:5]
    x2 = branch[:, 5:8]
    positions = torch.stack([x1, x2], dim=1)
    P_plus = branch[:, 8:11]
    P_minus = branch[:, 11:14]
    momenta = torch.stack([P_plus, P_minus], dim=1)

    n_points = trunk.shape[0]
    delta = torch.eye(3, dtype=torch.float).view(1, 3, 3)
    kbar = torch.zeros(n_points, 3, 3, dtype=torch.float)

    for n in range(0, 2):
        xn = trunk - positions[:, n]
        xi = xn.view(n_points, 3, 1)
        xj = xn.view(n_points, 1, 3)
        rn = torch.linalg.norm(xn, dim=1, dtype=torch.float, keepdim=True).clamp_min(1e-12)
        
        Pn = momenta[:,n]
        Pi = Pn.view(n_points, 3, 1)
        Pj = Pn.view(n_points, 1, 3)

        P_dot_X = torch.sum(xn * Pn.view(n_points, 3), dim=1, dtype=torch.float, keepdim=True).view(n_points, 1, 1)

        momentum_term = xi * Pj + xj * Pi - (delta - xi * xj / rn.pow(2).view(n_points, 1, 1)) * P_dot_X

        kbar = kbar + 3 / (2 * rn.pow(3).view(n_points, 1, 1)) * momentum_term

    return kbar

def PsiSingular(trunk, branch):
    m1 = branch[:, 0:1]
    m2 = branch[:, 1:2]
    masses = torch.hstack([m1, m2])
    x1 = branch[:, 2:5]
    x2 = branch[:, 5:8]
    positions = torch.stack([x1, x2], dim=1)

    psi = torch.ones(trunk.shape[0], 1, dtype=torch.float)
    for n in range(0, 2):
        rn = torch.linalg.norm((trunk - positions[:, n]), dim=1, dtype=torch.float, keepdim=True).clamp_min(1e-12)
        psi = psi + masses[:, n:n+1] / (2 * rn)
    
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

def Laplacian(u, trunk):
    derivative = Grad(u, trunk)
    fx = derivative[:, 0:1]
    fy = derivative[:, 1:2]
    fz = derivative[:, 2:3]

    fxx = Grad(fx, trunk)[:, 0:1]
    fyy = Grad(fy, trunk)[:, 1:2]
    fzz = Grad(fz, trunk)[:, 2:3]

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

def CalcUg(trunk, branch):
    m1 = branch[:, 0:1]
    m2 = branch[:, 1:2]
    masses = torch.hstack([m1, m2])
    x1 = branch[:, 2:5]
    x2 = branch[:, 5:8]
    positions = torch.stack([x1, x2], dim=1)
    P_plus = branch[:, 8:11]
    P_minus = branch[:, 11:14]
    momenta = torch.stack([P_plus, P_minus], dim=1)

    total = torch.zeros(trunk.shape[0], 1, dtype=torch.float)

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

        # Direct expression away from the puncture
        Rl = R[large]
        if Rl.numel() > 0:
            l = 1 / (1 + Rl)
            u0[large] = (5/32* (l - 2*l.pow(2) + 2*l.pow(3) - l.pow(4) + 0.2*l.pow(5)))
            u2[large] = (15*l + 132*l.pow(2) + 53*l.pow(3) + 96*l.pow(4) + 82*l.pow(5) + 84*l.pow(5)/Rl + 84*torch.log(l)/Rl.pow(2)) / (80 * Rl)

        return u0, u2

    for n in range(0, 2):
        m = masses[:, n:n+1]
        P = momenta[:, n]
        P_mag = torch.linalg.norm(P, dim=1, keepdim=True).clamp_min(1e-12)

        xn = trunk - positions[:, n]
        rn = torch.sum(xn.pow(2), dim=1, dtype=torch.float, keepdim=True).pow(1/2).clamp_min(1e-12)
        
        curly_R = (2 * rn / m).clamp_min(1e-12)

        r_hat = xn / rn
        P_hat = P / P_mag
        mu_P = torch.sum(r_hat * P_hat, dim=1, keepdim=True).clamp(-1, 1)
        P2 = 0.5 * (3 * mu_P.pow(2) - 1)

        u0P, u2P = u0P_u2P_stable(curly_R)

        curly_P = 2 * P_mag / m
        total = total + curly_P.pow(2) * (u0P + u2P * P2)

    return total

def WindowFunction(trunk, branch):
    ug = CalcUg(trunk, branch)

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
def Ansatz(h_theta, trunk, branch, c_mag=1):
    ug = CalcUg(trunk, branch)
    W = WindowFunction(trunk, branch)
    u_theta = kappa * ug * (1 + c_mag * W * torch.tanh(h_theta))
    
    return u_theta

def PdeResidual(u_theta, trunk, branch):
    lap_u = Laplacian(u_theta, trunk)
    psi = PsiSingular(trunk, branch) + u_theta
    k_bar = KBar(trunk, branch)
    k2 = k_bar.pow(2).sum(dim=(1, 2)).unsqueeze(1)

    return lap_u + 1/8 * psi.clamp_min(1e-12).pow(-7) * k2

def BoundaryResidual(u_theta, trunk):
    grad_u = Grad(u_theta, trunk)
    n = trunk / 30

    return torch.linalg.vecdot(grad_u, n).unsqueeze(1) + u_theta / 30

def SoftLinfLoss(res, beta=10):
    abs_res = res.abs().reshape(-1)
    n = len(abs_res)

    return (torch.logsumexp(beta * abs_res, dim=0) - np.log(n)) / beta

def RawLoss(u_theta_train, u_theta_data, u_theta_int, u_theta_bound, trunk_train_int, branch_train_int, trunk_train_bound, branch_train_bound):
    res_data = u_theta_train - u_theta_data
    res_int = PdeResidual(u_theta_int, trunk_train_int, branch_train_int)
    res_bound = BoundaryResidual(u_theta_bound, trunk_train_bound, branch_train_bound)

    L_data = torch.mean(res_data.pow(2))
    L2 = torch.mean(res_int.pow(2))
    L_inf = SoftLinfLoss(res_int)
    LBC = torch.mean(res_bound.pow(2))

    return L_data, L2, L_inf, LBC

def ScaledTotalLoss(L_data, L2, L_inf, LBC):
    global ema_L_data, ema_L2, ema_L_inf, ema_LBC
    alpha = 0.95

    with torch.no_grad():
        if ema_L2 is None:
            ema_L_data = L_data.detach()
            ema_L2 = L2.detach()
            ema_L_inf = L_inf.detach()
            ema_LBC = LBC.detach()
        else:
            ema_L_data = alpha * ema_L_data + (1 - alpha) * L_data.detach()
            ema_L2 = alpha * ema_L2 + (1 - alpha) * L2.detach()
            ema_L_inf = alpha * ema_L_inf + (1 - alpha) * L_inf.detach()
            ema_LBC = alpha * ema_LBC + (1 - alpha) * LBC.detach()

    L_data_tilde = L_data / ema_L_data
    L2_tilde = L2 / ema_L2
    L_inf_tilde = L_inf / ema_L_inf
    LBC_tilde = LBC / ema_LBC

    return w_data * L_data_tilde * w2 * L2_tilde + w_inf * L_inf_tilde + w_rob * LBC_tilde

#%% Setup
kappa = 0.635
ug_min = 0.0006
ug_max = 0.0415

branch_net = nn.Sequential(
    nn.Linear(14, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 50)
)

trunk_net = nn.Sequential(
    nn.Linear(3, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 50)
)

output_bias = nn.Parameter(torch.tensor([1.0], dtype=torch.float))

optimizer = torch.optim.Adam(list(branch_net.parameters()) + list(trunk_net.parameters()) + [output_bias], lr=1e-4)

w_data = 1
w2 = 1
w_inf = 0.5
w_rob = 1

L_data_list = []
L2_list = []
L_inf_list = []
LBC_list = []
total_loss_list = []
L2RE_list = []

start_epoch = 1

#%% Model functions
def ModelForward(trunk_net, branch_net, trunk, branch):
    t = trunk_net(trunk)
    b = branch_net(branch)

    h_theta = torch.sum(b * t, dim=1, keepdim=True)
    u_theta = Ansatz(h_theta, trunk, branch)

    return u_theta

def train_epoch(epoch):
    u_pred_data_list, u_data_list = [], []
    trunk_int_list, branch_int_list, u_int_list = [], [], []
    trunk_bound_list, branch_bound_list, u_bound_list = [], [], []

    trunk_net.train()
    branch_net.train()

    for branch_data_train, trunk_data_train, u_data_train in tqdm(dloader_data_train, desc=f"Epoch {epoch} (training data)", leave=False):
        u_pred_train = ModelForward(trunk_net, branch_net, trunk_data_train, branch_data_train)

        u_pred_data_list.append(u_pred_train)
        u_data_list.append(u_data_train)

    u_pred_data_list = torch.concat(u_pred_data_list, dim=0)
    u_data_list = torch.concat(u_data_list, dim=0)

    for branch_int_train, trunk_int_train in tqdm(dloader_int, desc=f"Epoch {epoch} (training interior)", leave=False):
        u_int_train = ModelForward(trunk_net, branch_net, trunk_int_train, branch_int_train)
    
        trunk_int_list.append(trunk_int_train)
        branch_int_list.append(branch_int_train)
        u_int_list.append(u_int_train)
    
    trunk_int_list = torch.concat(trunk_int_list, dim=0)
    branch_int_list = torch.concat(branch_int_list, dim=0)
    u_int_list = torch.concat(u_int_list, dim=0)

    for branch_bound_train, trunk_bound_train in tqdm(dloader_bound, desc=f"Epoch {epoch} (training boundary)", leave=False):
        u_bound_train = ModelForward(trunk_net, branch_net, trunk_bound_train, branch_bound_train)
    
        trunk_bound_list.append(trunk_bound_train)
        branch_bound_list.append(branch_bound_train)
        u_bound_list.append(u_bound_train)
    
    trunk_bound_list = torch.concat(trunk_bound_list, dim=0)
    branch_bound_list = torch.concat(branch_bound_list, dim=0)
    u_bound_list = torch.concat(u_bound_list, dim=0)

    L_data, L2, L_inf, LBC = RawLoss(u_data_train, u_pred_data_list, u_int_list, u_bound_list, trunk_int_list, branch_int_list, trunk_bound_list, branch_bound_list)
    total_loss = ScaledTotalLoss(L_data, L2, L_inf, LBC)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    u_pred_validate_list, u_validate_list = [], []

    trunk_net.eval()
    branch_net.eval()

    with torch.no_grad():
        for branch_data_validate, trunk_data_validate, u_data_validate in tqdm(dloader_data_validate, desc=f"Epoch {epoch} (Validating data)", leave=False):
            u_pred_validate = ModelForward(trunk_net, branch_net, trunk_data_validate, branch_data_validate)

            u_pred_validate_list.append(u_pred_validate)
            u_validate_list.append(u_data_validate)

        u_pred_validate_list = torch.concat(u_pred_validate_list, dim=0)
        u_validate_list = torch.concat(u_validate_list, dim=0)

        u_pred_validate_list = u_pred_validate_list.detach().numpy().flatten()
        u_validate_list = u_validate_list.detach().numpy().flatten()
        L2RE = np.sqrt(((u_pred_validate_list - u_validate_list.flatten())**2).sum() / (u_validate_list.flatten()**2).sum())

    return L_data, L2, L_inf, LBC, L2RE
#%% Training code
n_epoch = 100

for epoch in range(start_epoch, start_epoch+n_epoch):
    L_data, L2, L_inf, LBC, L2RE = train_epoch(epoch)

    L_data_list.append(L_data.detach().item())
    L2_list.append(L2.detach().item())
    L_inf_list.append(L_inf.detach().item())
    LBC_list.append(LBC.detach().item())
    L2RE_list.append(L2RE)

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

total_loss = np.array(L_data_list) + np.array(L2_list) + np.array(L_inf_list) + np.array(LBC_list)

fig2, ax2 = plt.subplots(1,1,figsize = (10,4),dpi = 150)
ax2.plot(epochs, L_data_list, label='L_data', zorder=1)
ax2.plot(epochs, L2_list, label='L2', zorder=1)
ax2.plot(epochs, L_inf_list, label='L_inf', zorder=1)
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