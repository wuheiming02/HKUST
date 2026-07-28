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

#%% Check CUDA
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

#%% Generate datasets
train_batch_size = 65536
validate_batch_size = 131072
int_batch_size = 8192
bound_batch_size = 1024
test_batch_size = 131072

np.random.seed(42)
torch.manual_seed(42)

def GenBranchSamples(m1):
    N = m1.shape[0]
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

npz_01 = np.load('data/TP01.npz')
coords_01 = npz_01['coords']
u_01 = npz_01['u']
params_01 = GenBranchSamples(torch.tensor([[0.1]] * len(u_01), dtype=torch.float).reshape(-1, 1))

npz_02 = np.load('data/TP02.npz')
coords_02 = npz_02['coords']
u_02 = npz_02['u']
params_02 = GenBranchSamples(torch.tensor([[0.2]] * len(u_02), dtype=torch.float).reshape(-1, 1))

npz_03 = np.load('data/TP03.npz')
coords_03 = npz_03['coords']
u_03 = npz_03['u']
params_03 = GenBranchSamples(torch.tensor([[0.3]] * len(u_03), dtype=torch.float).reshape(-1, 1))

npz_04 = np.load('data/TP04.npz')
coords_04 = npz_04['coords']
u_04 = npz_04['u']
params_04 = GenBranchSamples(torch.tensor([[0.4]] * len(u_04), dtype=torch.float).reshape(-1, 1))

npz_05 = np.load('data/TP05.npz')
coords_05 = npz_05['coords']
u_05 = npz_05['u']
params_05 = GenBranchSamples(torch.tensor([[0.5]] * len(u_05), dtype=torch.float).reshape(-1, 1))

branch_data = torch.concat([params_01, params_02, params_03, params_04, params_05])
trunk_data = np.concat([coords_01, coords_02, coords_03, coords_04, coords_05])
u_data = np.concat([u_01, u_02, u_03, u_04, u_05])

trunk_data = torch.tensor(trunk_data, dtype=torch.float)
u_data = torch.tensor(u_data, dtype=torch.float).reshape(-1, 1)

dataset = TensorDataset(branch_data, trunk_data, u_data)
dataset_train, dataset_validate = random_split(
    dataset = dataset,
    lengths = [0.8, 0.2],
    generator = torch.Generator().manual_seed(42)
)

dloader_data_train = DataLoader(
    dataset = dataset_train,
    batch_size = train_batch_size,
    shuffle = True,
    pin_memory = (device.type == "cuda")
)

dloader_data_validate = DataLoader(
    dataset = dataset_validate,
    batch_size = validate_batch_size,
    shuffle = True,
    pin_memory = (device.type == "cuda")
)

def GenXint(n_points=1e6):
    directions = torch.randn(int(n_points), 3, dtype=torch.float)
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1e-12)
    radii = 10 * torch.rand(int(n_points), 1, dtype=torch.float)
    m1 = torch.rand(int(n_points), 1, dtype=torch.float) * 0.4 + 0.1
    branch_interior = GenBranchSamples(m1)

    return radii * directions, branch_interior

def GenXext(n_points=1e6):
    directions = torch.randn(int(n_points), 3, dtype=torch.float)
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1e-12)
    radii = 20 * torch.rand(int(n_points), 1, dtype=torch.float) + 10
    m1 = torch.rand(int(n_points), 1, dtype=torch.float) * 0.4 + 0.1
    branch_ext = GenBranchSamples(m1)

    return radii * directions, branch_ext

def GenXbound(n_points=2e5):
    directions = torch.randn(int(n_points), 3, dtype=torch.float)
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True).clamp_min(1e-12)
    m1 = torch.rand(int(n_points), 1, dtype=torch.float) * 0.4 + 0.1
    branch_boundary = GenBranchSamples(m1)

    return 30 * directions, branch_boundary

trunk_int, branch_int = GenXint(1e6)
trunk_ext, branch_ext = GenXext(1e6)
trunk_bound, branch_bound = GenXbound(2e5)

trunk_int = torch.concat([trunk_int, trunk_ext])
branch_int = torch.concat([branch_int, branch_ext])

dataset_int = TensorDataset(branch_int, trunk_int)

dloader_int = DataLoader(
    dataset = dataset_int,
    batch_size = int_batch_size,
    shuffle = True,
    pin_memory = (device.type == "cuda")
)

dataset_bound = TensorDataset(branch_bound, trunk_bound)

dloader_bound = DataLoader(
    dataset = dataset_bound,
    batch_size = bound_batch_size,
    shuffle = True,
    pin_memory = (device.type == "cuda")
)

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

dataset_test = TensorDataset(branch_test, trunk_test, u_test)

dloader_test = DataLoader(
    dataset = dataset_test,
    batch_size = 131072,
    shuffle = False,
    drop_last = False,
    pin_memory = (device.type == "cuda")
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
    delta = torch.eye(3, dtype=torch.float, device=device).view(1, 3, 3)
    kbar = torch.zeros(n_points, 3, 3, dtype=torch.float, device=device)

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

    psi = torch.ones(trunk.shape[0], 1, dtype=torch.float, device=device)
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

    total = torch.zeros(trunk.shape[0], 1, dtype=torch.float, device=device)

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

#%% Loss functions
def Ansatz(h_theta, trunk, branch, c_mag=1):
    ug = CalcUg(trunk, branch)
    W = (ug - ug_min) / (ug_max - ug_min)
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
    abs_res = res.abs().reshape(-1) * beta

    return (torch.logsumexp(abs_res, dim=0) - np.log(abs_res.numel())) / beta

def ScaledTotalLoss(L_data, L2, L_inf, LBC):
    global ema_L_data, ema_L2, ema_L_inf, ema_LBC
    alpha = 0.999

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

    L_data_tilde = L_data / ema_L_data.clamp_min(1e-12)
    L2_tilde = L2 / ema_L2.clamp_min(1e-12)
    L_inf_tilde = L_inf / ema_L_inf.clamp_min(1e-12)
    LBC_tilde = LBC / ema_LBC.clamp_min(1e-12)

    return w_data * L_data_tilde + w2 * L2_tilde + w_inf * L_inf_tilde + w_rob * LBC_tilde

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
).to(device)

trunk_net = nn.Sequential(
    nn.Linear(3, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 50)
).to(device)

output_bias = nn.Parameter(torch.tensor([1.0], dtype=torch.float, device=device))

optimizer = torch.optim.Adam(list(branch_net.parameters()) + list(trunk_net.parameters()) + [output_bias], lr=1e-4)

ema_L_data = None
ema_L2 = None
ema_L_inf = None
ema_LBC = None

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

u_pred_test_list = []

start_epoch = 1

#%% Model functions
def ModelForward(trunk_net, branch_net, trunk, branch):
    t = trunk_net(trunk)
    b = branch_net(branch)

    h_theta = torch.sum(b * t, dim=1, keepdim=True) + output_bias
    u_theta = Ansatz(h_theta, trunk, branch)

    return u_theta

def NextBatch(iterator, dloader):
    try:
        batch = next(iterator)

    except StopIteration:
        iterator = iter(dloader)
        batch = next(iterator)

    return batch, iterator

def train_epoch(epoch, beta=10.0):
    trunk_net.train()
    branch_net.train()

    n_data_batches = len(dloader_data_train)
    n_int_batches = len(dloader_int)
    n_bound_batches = len(dloader_bound)
    n_steps = max(n_data_batches, n_int_batches, n_bound_batches)

    data_iterator = iter(dloader_data_train)
    int_iterator = iter(dloader_int)
    bound_iterator = iter(dloader_bound)


    n_data_points, n_int_points, n_linf_points, n_bound_points = 0, 0, 0, 0
    sum_data_squared = 0.0
    sum_int_squared = 0.0
    sum_bound_squared = 0.0
    global_linf_logsumexp = None

    sum_scaled_total_loss = 0.0

    for step in tqdm(range(n_steps), desc=f"Epoch {epoch} (training)", leave=False):
        data_batch, data_iterator = NextBatch(data_iterator, dloader_data_train)
        int_batch, int_iterator = NextBatch(int_iterator, dloader_int)
        bound_batch, bound_iterator = NextBatch(bound_iterator, dloader_bound)

        branch_data_batch, trunk_data_batch, u_data_batch = data_batch
        branch_int_batch, trunk_int_batch = int_batch
        branch_bound_batch, trunk_bound_batch = bound_batch

        branch_data_batch = branch_data_batch.to(device, non_blocking=True)
        trunk_data_batch = trunk_data_batch.to(device, non_blocking=True)
        u_data_batch = u_data_batch.to(device, non_blocking=True).reshape(-1, 1)

        branch_int_batch = branch_int_batch.to(device, non_blocking=True)
        trunk_int_batch = trunk_int_batch.to(device, non_blocking=True).detach().requires_grad_(True)

        branch_bound_batch = branch_bound_batch.to(device, non_blocking=True)
        trunk_bound_batch = trunk_bound_batch.to(device, non_blocking=True).detach().requires_grad_(True)

        u_pred_data = ModelForward(trunk_net, branch_net, trunk_data_batch, branch_data_batch)
        res_data = u_pred_data - u_data_batch
        L_data_batch = res_data.pow(2).mean()

        u_pred_int = ModelForward(trunk_net, branch_net, trunk_int_batch, branch_int_batch)
        res_int = PdeResidual(u_pred_int, trunk_int_batch, branch_int_batch)
        L2_batch = res_int.pow(2).mean()
        L_inf_batch = SoftLinfLoss(res_int, beta=beta)

        u_pred_bound = ModelForward(trunk_net, branch_net, trunk_bound_batch, branch_bound_batch)
        res_bound = BoundaryResidual(u_pred_bound, trunk_bound_batch)
        LBC_batch = res_bound.pow(2).mean()

        total_loss_batch = ScaledTotalLoss(L_data_batch, L2_batch, L_inf_batch, LBC_batch)

        optimizer.zero_grad(set_to_none=True)
        total_loss_batch.backward()
        optimizer.step()
        sum_scaled_total_loss = sum_scaled_total_loss + total_loss_batch.detach().item()

        if step < n_data_batches:
            res_data_detached = res_data.detach()
            sum_data_squared = sum_data_squared + res_data_detached.pow(2).sum().item()
            n_data_points = n_data_points + res_data_detached.numel()

        if step < n_int_batches:
            res_int_detached = res_int.detach()
            sum_int_squared = sum_int_squared + res_int_detached.pow(2).sum().item()
            n_int_points = n_int_points + res_int_detached.numel()

            batch_logsumexp = torch.logsumexp(beta * res_int_detached.abs().reshape(-1),dim=0).cpu()

            if global_linf_logsumexp is None:
                global_linf_logsumexp = batch_logsumexp
            else:
                global_linf_logsumexp = torch.logaddexp(global_linf_logsumexp, batch_logsumexp)
            n_linf_points += res_int_detached.numel()

        if step < n_bound_batches:
            res_bound_detached = res_bound.detach()
            sum_bound_squared = sum_bound_squared + res_bound_detached.pow(2).sum().item()
            n_bound_points += res_bound_detached.numel()

    epoch_L_data = sum_data_squared / max(n_data_points, 1)
    epoch_L2 = sum_int_squared / max(n_int_points, 1)
    epoch_LBC = sum_bound_squared / max(n_bound_points, 1)
    epoch_L_inf = (global_linf_logsumexp.item() - np.log(max(n_linf_points, 1))) / beta
    epoch_total_loss = sum_scaled_total_loss / n_steps

    trunk_net.eval()
    branch_net.eval()

    sum_u_res = torch.zeros((), dtype=torch.float, device=device)
    sum_u_batch = torch.zeros((), dtype=torch.float, device=device)

    with torch.no_grad():
        for branch_valid_batch, trunk_valid_batch, u_valid_batch in tqdm(dloader_data_validate, desc=f"Epoch {epoch} (validating data)", leave=False):
            branch_valid_batch = branch_valid_batch.to(device, non_blocking=True)
            trunk_valid_batch = trunk_valid_batch.to(device, non_blocking=True)
            u_valid_batch = u_valid_batch.to(device, non_blocking=True)
    
            u_pred_valid = ModelForward(trunk_net, branch_net, trunk_valid_batch, branch_valid_batch)

            sum_u_res = sum_u_res + ((u_pred_valid - u_valid_batch).pow(2)).sum()
            sum_u_batch = sum_u_batch + (u_valid_batch.pow(2)).sum()

        L2RE = torch.sqrt(sum_u_res / sum_u_batch.clamp_min(1e-24)).item()

    return epoch_L_data, epoch_L2, epoch_L_inf, epoch_LBC, epoch_total_loss, L2RE

#%% Training code
n_epoch = 5000
patience = 200
best_epoch = start_epoch - 1
min_L2RE = np.inf

for epoch in range(start_epoch, start_epoch+n_epoch):
    L_data, L2, L_inf, LBC, total_loss, L2RE = train_epoch(epoch, beta=10)

    L_data_list.append(L_data)
    L2_list.append(L2)
    L_inf_list.append(L_inf)
    LBC_list.append(LBC)
    total_loss_list.append(total_loss)
    L2RE_list.append(L2RE)

    if L2RE < min_L2RE:
        best_epoch = epoch
        min_L2RE = L2RE

    elif best_epoch <= epoch - patience and epoch > 300:
        break

print('Training finished')

#%% Plot losses
epochs = np.linspace(1, len(L_data_list), len(L_data_list))

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

#%% Test dataset code
with torch.no_grad():
        for branch_test_batch, trunk_test_batch, u_test_batch in tqdm(dloader_test, desc=f"Testing", leave=False):
            branch_test_batch = branch_test_batch.to(device, non_blocking=True)
            trunk_test_batch = trunk_test_batch.to(device, non_blocking=True)
            u_test_batch = u_test_batch.to(device, non_blocking=True)
    
            u_pred_test = ModelForward(trunk_net, branch_net, trunk_test_batch, branch_test_batch)
            u_pred_test_list.append(u_pred_test.detach().cpu())

#%% Save checkpoint
checkpoint = {
    "branch_net_state_dict": branch_net.state_dict(),
    "trunk_net_state_dict": trunk_net.state_dict(),
    "output_bias": output_bias.detach(),
    "optimizer_state_dict":optimizer.state_dict(),

    "L_data_list": list(L_data_list),
    "L2_list": list(L2_list),
    "L_inf_list": list(L_inf_list),
    "LBC_list": list(LBC_list),
    "total_loss_list": list(total_loss_list),
    "L2RE_list": list(L2RE_list),

    "ema_L_data":ema_L_data,
    "ema_L2": ema_L2,
    "ema_L_inf": ema_L_inf,
    "ema_LBC": ema_LBC,

    "u_pred_test_list":u_pred_test_list
}

torch.save(checkpoint, "Full_PIDONet_Checkpoint.pt")

print('Checkpoint saved')

#%% Load checkpoint
checkpoint = torch.load("Full_PIDONet_Checkpoint.pt", weights_only=False, map_location=device)

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
).to(device)

trunk_net = nn.Sequential(
    nn.Linear(3, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 50)
).to(device)

branch_net.load_state_dict(checkpoint["branch_net_state_dict"])
trunk_net.load_state_dict(checkpoint["trunk_net_state_dict"])
output_bias = nn.Parameter(checkpoint["output_bias"])

optimizer = torch.optim.Adam(list(branch_net.parameters()) + list(trunk_net.parameters()) + [output_bias], lr=1e-4)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

kappa = 0.635
ug_min = 0.0006
ug_max = 0.0415

ema_L_data = checkpoint["ema_L_data"]
ema_L2 = checkpoint["ema_L2"]
ema_L_inf = checkpoint["ema_L_inf"]
ema_LBC = checkpoint["ema_LBC"]

w_data = 1
w2 = 1
w_inf = 0.5
w_rob = 1

L_data_list = checkpoint["L_data_list"]
L2_list = checkpoint["L2_list"]
L_inf_list = checkpoint["L_inf_list"]
LBC_list = checkpoint["LBC_list"]
total_loss_list = checkpoint["total_loss_list"]
L2RE_list = checkpoint["L2RE_list"]

u_pred_test_list = checkpoint["u_pred_test_list"]

start_epoch = len(total_loss_list) + 1

print('Checkpoint loaded')
print('Start epoch: {}'.format(start_epoch))

#%%
