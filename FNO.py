#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

#%% Generate datasets and dataloaders
np.random.seed(42)
torch.manual_seed(42)

alpha = 0.01
T_final = 1.0
k_max = 10

n_samples = 2000
n_super_res = 200

s_train = 64
s_super_res = 256

def fourier_coefficients(n_samples, k_max, coeff_decay=1.5):
    k = np.arange(1, k_max+1)

    c0 = 0.2 * np.random.randn(n_samples, 1)
    a = np.random.rand(n_samples, k_max) / k[None, :]**coeff_decay
    b = np.random.rand(n_samples, k_max) / k[None, :]**coeff_decay

    return c0, a, b

def heat_solution(c0, a, b, s, alpha, T_final):
    n_samples = c0.shape[0]
    max_k = a.shape[1]

    X = np.linspace(0, 1, s, endpoint=False)
    u0 = np.zeros((n_samples, s))
    uT = np.zeros((n_samples, s))

    u0 = u0 + c0
    uT = uT + c0

    for i in range(max_k):
        k = i + 1
        sin_k = np.sin(2 * np.pi * k * X)[None, :]
        cos_k = np.cos(2 * np.pi * k * X)[None, :]

        mode = a[:, i:i+1] * sin_k + b[:, i:i+1] * cos_k
        decay = np.exp(-alpha * (2 * np.pi * k)**2 * T_final)

        u0 = u0 + mode
        uT = uT + mode * decay

    return X, u0, uT

def gen_dataset(coeffs, s, alpha, T_final):
    c0, a, b = coeffs
    X, u0, uT = heat_solution(c0, a, b, s, alpha, T_final)

    n_samples = u0.shape[0]
    X_grid = np.tile(X[None, :], (n_samples, 1))

    input_data = np.stack([u0, X_grid], axis=-1)
    target_data = uT[:, :, None]

    input_data = torch.tensor(input_data, dtype=torch.float)
    target_data = torch.tensor(target_data, dtype=torch.float)

    return input_data, target_data, X

samples_coeff = fourier_coefficients(n_samples, k_max)
super_res_coeff = fourier_coefficients(n_super_res, k_max)
X_sample, y_sample, x_sample = gen_dataset(samples_coeff, s_train, alpha, T_final)
X_super_res, y_super_res, x_super_res = gen_dataset(super_res_coeff, s_super_res, alpha, T_final)

dataset = TensorDataset(X_sample, y_sample)
dataset_train, dataset_validate, dataset_test = random_split(
    dataset = dataset,
    lengths = [0.6, 0.2, 0.2],
    generator = torch.Generator().manual_seed(42)
)

train_indices = dataset_train.indices
validate_indices = dataset_validate.indices
test_indices = dataset_test.indices

X_train = X_sample[train_indices]
y_train = y_sample[train_indices]
X_validate = X_sample[validate_indices]
y_validate = y_sample[validate_indices]
X_test = X_sample[test_indices]
y_test = y_sample[test_indices]

X_train_mean = X_train[:, :, 0:1].mean()
X_train_std = X_train[:, :, 0:1].std()
y_train_mean = y_train.mean()
y_train_std = y_train.std()

X_train_norm = X_train.clone()
X_validate_norm = X_validate.clone()
X_test_norm = X_test.clone()
X_super_res_norm = X_super_res.clone()
y_train_norm = y_train.clone()
y_validate_norm = y_validate.clone()
y_test_norm = y_test.clone()
y_super_res_norm = y_super_res.clone()

X_train_norm[:, :, 0:1] = (X_train[:, :, 0:1] - X_train_mean) / X_train_std
X_validate_norm[:, :, 0:1] = (X_validate[:, :, 0:1] - X_train_mean) / X_train_std
X_test_norm[:, :, 0:1] = (X_test[:, :, 0:1] - X_train_mean) / X_train_std
X_super_res_norm[:, :, 0:1] = (X_super_res[:, :, 0:1] - X_train_mean) / X_train_std
y_train_norm = (y_train - y_train_mean) / y_train_std
y_validate_norm = (y_validate - y_train_mean) / y_train_std
y_test_norm = (y_test - y_train_mean) / y_train_std
y_super_res_norm = (y_super_res - y_train_mean) / y_train_std

dataset_train_norm = TensorDataset(X_train_norm, y_train_norm)
dataset_validate_norm = TensorDataset(X_validate_norm, y_validate_norm)
dataset_test_norm = TensorDataset(X_test_norm, y_test_norm)
dataset_super_res_norm = TensorDataset(X_super_res_norm, y_super_res_norm)

dloader_train = DataLoader(
    dataset = dataset_train_norm,
    batch_size = 32,
    shuffle = True,
    drop_last = True
)

dloader_validate = DataLoader(
    dataset = dataset_validate_norm,
    batch_size = 16,
    shuffle = True
)

dloader_test = DataLoader(
    dataset = dataset_test_norm,
    batch_size = 16,
    shuffle = True
)

dloader_test_super_res = DataLoader(
    dataset = dataset_super_res_norm,
    batch_size = 16,
    shuffle = True
)

#%% Define FNO model
width = 32
modes = 12
n_layers = 4

lift = nn.Linear(2, width)

scale = 1 / (width * width)
spectral_weights = nn.ParameterList([
    nn.Parameter(scale * torch.randn((width, width, modes), dtype=torch.cfloat)) for _ in range(n_layers)
])

local_paths = nn.ModuleList([
    nn.Conv1d(
        in_channels = width,
        out_channels = width,
        kernel_size = 1
    ) for _ in range(n_layers)
])

def global_path(X, weights, modes):
    batch_size = X.shape[0]
    width = X.shape[1]
    s = X.shape[2]

    X_ft = torch.fft.rfft(X, dim=-1)
    out_ft = torch.zeros((batch_size, width, X_ft.shape[-1]), dtype=torch.cfloat)
    X_ft_clipped = X_ft[:, :, :modes]
    out_ft[:, : , :modes] = torch.einsum('bim,iom->bom', X_ft_clipped, weights)
    X_out = torch.fft.irfft(out_ft, n=s, dim=-1)

    return X_out

project = nn.Sequential(
    nn.Linear(width, 128),
    nn.GELU(),
    nn.Linear(128, 1)
)

def fno_forward(X):
    x = lift(X)
    x = x.permute(0, 2, 1)

    for layer in range(n_layers):
        x_global = global_path(x, spectral_weights[layer], modes)
        x_local = local_paths[layer](x)
        x = nn.functional.gelu(x_global+x_local)

    x = x.permute(0, 2, 1)
    y_pred = project(x)

    return y_pred

#%% Define loss function
def loss_fcn(y_pred, y_true):
    batch_size = y_true.shape[0]

    diff_norm = torch.linalg.norm((y_pred - y_true).reshape(batch_size, -1), dim=1)
    true_norm = torch.linalg.norm(y_true.reshape(batch_size, -1), dim=1)

    return torch.mean(diff_norm/true_norm)

#%% Define optimizer
parameters_list = (
    list(lift.parameters()) + 
    list(spectral_weights.parameters()) + 
    list(local_paths.parameters()) + 
    list(project.parameters())
)

adam_optimizer = torch.optim.Adam(parameters_list, lr=1e-3)

#%% Define training cycle
def train_epoch(optimizer, loss_fcn, epoch):
    tot_loss = 0
    valid_loss = 0

    lift.train()
    local_paths.train()
    project.train()
    for X_train, y_train in tqdm(dloader_train, desc=f"Epoch {epoch+1} (training)", leave=False):
        y_pred = fno_forward(X_train)
        optimizer.zero_grad()
        loss = loss_fcn(y_pred, y_train)
        tot_loss = tot_loss + loss.item()
        loss.backward()
        optimizer.step()

    lift.eval()
    local_paths.eval()
    project.eval()
    with torch.no_grad():
        for X_valid, y_valid in tqdm(dloader_validate, desc=f"Epoch {epoch+1} (validation)", leave=False):
            y_pred_v = fno_forward(X_valid)
            vloss = loss_fcn(y_pred_v, y_valid)
            valid_loss = valid_loss + vloss.item()

    return tot_loss/len(dloader_train), valid_loss/len(dloader_validate)

#%% Training cycle
n_epochs = 300

training_loss_list = []
validation_loss_list = []

for epoch in range(n_epochs):
    training_loss, valid_loss = train_epoch(adam_optimizer, loss_fcn, epoch)
    training_loss_list.append(training_loss)
    validation_loss_list.append(valid_loss)

    if epoch % 50 == 0:
        print(f'epoch:', epoch)

print("Training finished")

#%% Plot losses
fig, ax = plt.subplots(1,1,figsize = (8,6),dpi = 150)
ax.plot(training_loss_list, color='black',label='Training loss')
ax.plot(validation_loss_list, color='#D55E00',label='Validation loss')
ax.set_xlabel('Epoch',fontsize = 16)
ax.set_ylabel('L2 relative loss',fontsize = 16)
ax.set_yscale('log')
ax.set_title('Loss during training',fontsize = 20)
ax.tick_params(labelsize=12, which='both',top=True, right = True, direction='in')
ax.grid(color='xkcd:dark blue',alpha = 0.2)
ax.legend(loc='upper right',fontsize = 12)
plt.show()

#%% Save training progress
checkpoint = {
    "lift_state_dict": lift.state_dict(),
    "local_paths_state_dict": local_paths.state_dict(),
    "project_state_dict": project.state_dict(),

    "spectral_weights": [w.detach() for w in spectral_weights],

    "alpha": alpha,
    "T_final": T_final,
    "width": width,
    "modes": modes,
    "n_layers": n_layers,

    "X_train_mean": X_train_mean.detach(),
    "X_train_std": X_train_std.detach(),
    "y_train_mean": y_train_mean.detach(),
    "y_train_std": y_train_std.detach(),

    "training_loss_list": training_loss_list,
    "validation_loss_list": validation_loss_list
}

torch.save(checkpoint, "fno_heat_equation_checkpoint.pt")

#%% Load checkpoint

checkpoint = torch.load("fno_heat_equation_checkpoint.pt")

width = checkpoint["width"]
modes = checkpoint["modes"]
n_layers = checkpoint["n_layers"]

lift = nn.Linear(2, width)

spectral_weights = nn.ParameterList([
    nn.Parameter(scale * torch.randn((width, width, modes), dtype=torch.cfloat)) for _ in range(n_layers)
])

local_paths = nn.ModuleList([
    nn.Conv1d(
        in_channels = width,
        out_channels = width,
        kernel_size = 1
    ) for _ in range(n_layers)
])

project = nn.Sequential(
    nn.Linear(width, 128),
    nn.GELU(),
    nn.Linear(128, 1)
)

lift.load_state_dict(checkpoint["lift_state_dict"])
local_paths.load_state_dict(checkpoint["local_paths_state_dict"])
project.load_state_dict(checkpoint["project_state_dict"])

for i in range(n_layers):
    spectral_weights[i].data = checkpoint["spectral_weights"][i]

X_train_mean = checkpoint["X_train_mean"]
X_train_std = checkpoint["X_train_std"]
y_train_mean = checkpoint["y_train_mean"]
y_train_std = checkpoint["y_train_std"]

training_loss_list = checkpoint["training_loss_list"]
validation_loss_list = checkpoint["validation_loss_list"]

print("Checkpoint loaded")
