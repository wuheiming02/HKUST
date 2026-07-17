#%% Import modules
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.auto import tqdm

#%% Initialise system
np.random.seed(42)
torch.manual_seed(42)

R_max = 30
puncture_separation = 6

masses = torch.tensor([0.5, 0.5], dtype=torch.float)
positions = torch.tensor([[3, 0, 0], [-3, 0, 0]], dtype=torch.float)
momenta = torch.tensor([[0, 0.2, 0], [0, -0.2, 0]], dtype=torch.float)

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

        u0P = (5/32) * (l-2*l.pow(2) + 2*l.pow(3) - l.pow(4) + 0.2*l.pow(5))
        u2P = (15*l + 132*l.pow(2) + 53*l.pow(3) + 96*l.pow(4) + 82*l.pow(5) + 84*l.pow(5)/curly_R + 84*torch.log(l)/curly_R.pow(2)) / (80 * curly_R)

        curly_P = 2 * P_mag / m

        total = total + curly_P.pow(2) * (u0P + u2P * P2)

    return total
    
def WindowFunction(X):
    ug = CalcUg(X)
    with torch.no_grad():
        u_min = ug.min().detach()
        u_max = ug.max().detach()

    return (ug - u_min) / (u_max - u_min)

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
def Ansatz(h_theta, X, kappa, c_mag=1):
    ug = CalcUg(X)
    W = WindowFunction(X)
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

def SoftLinfLoss(res, beta=2.5):
    abs_res = res.reshape(-1)
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

    with torch.no_grad():
        if ema_L2 is None:
            ema_L2 = L2.detach()
            ema_Linf = Linf.detach() + 1e-12
            ema_LBC = LBC.detach()
        else:
            ema_L2 = 0.99 * ema_L2 + (1 - 0.99) * L2.detach()
            ema_Linf = 0.99 * ema_Linf + (1 - 0.99) * Linf.detach()
            ema_LBC = 0.99 * ema_LBC + (1 - 0.99) * LBC.detach()

    L2_tilde = L2 / (ema_L2 + 1e-12)
    Linf_tilde = Linf / (ema_Linf + 1e-12)
    LBC_tilde = LBC / (ema_LBC + 1e-12)

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

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

kappa = EstimateKappa()

X_int = GenXint().requires_grad_(True)
X_bound = GenXbound().requires_grad_(True)

ema_L2 = None
ema_Linf = None
ema_LBC = None

w2 = 1
w_inf = 0
w_rob = 1

L2_list = []
Linf_list = []
LBC_list = []
total_loss_list = []

start_epoch = 1

#%% Training code
n_epoch = 200

for epoch in tqdm(range(start_epoch, start_epoch+n_epoch)):
    model.train()

    h_theta_int = model(X_int)
    u_theta_int = Ansatz(h_theta_int, X_int, kappa)

    h_theta_bound = model(X_bound)
    u_theta_bound = Ansatz(h_theta_bound, X_bound, kappa)

    L2, Linf, LBC = RawLoss(u_theta_int, u_theta_bound, X_int, X_bound)
    L2_list.append(L2.detach())
    Linf_list.append(Linf.detach())
    LBC_list.append(LBC.detach())

    total_loss = ScaledTotalLoss(L2, Linf, LBC)
    total_loss_list.append(total_loss.detach())

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

fig2, ax2 = plt.subplots(1,1,figsize = (10,4),dpi = 150)
ax2.plot(epochs, L2_list, label='L2', zorder=2)
ax2.plot(epochs, Linf_list, label='L_inf', zorder=2)
ax2.plot(epochs, LBC_list, label='LBC', zorder=2)
ax2.set_xlabel('Epoch',fontsize = 16)
ax2.set_ylabel('Loss',fontsize = 16)
ax2.set_yscale('log')
ax2.set_title('Loss during training',fontsize = 20)
ax2.tick_params(labelsize=12, which='both',top=True, right = True, direction='in')
ax2.grid(color='xkcd:dark blue',alpha = 0.2)
ax2.legend(loc='right',fontsize = 12)
plt.show()
#%% Save checkpoint
checkpoint = {
    "model_state_dict": model.state_dict(),
    "X_int": X_int.detach(),
    "X_bound": X_bound.detach(),

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
}

torch.save(checkpoint, "Simple_BBH_Checkpoint.pt")

print('Checkpoint saved')

#%% Load checkpoint
checkpoint = torch.load("Simple_BBH_Checkpoint.pt")

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

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

kappa = checkpoint["kappa"]

X_int = checkpoint["X_int"].requires_grad_(True)
X_bound = checkpoint["X_bound"].requires_grad_(True)

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

start_epoch = len(total_loss_list) + 1

print('Checkpoint loaded')
print('Start epoch: {}'.format(start_epoch+1))