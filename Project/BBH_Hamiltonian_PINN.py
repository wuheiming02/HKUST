#%% Import modules
# This script follows the section-by-section style of your NavierStokes.py file:
# imports -> sampling/data -> model -> residuals -> loss -> training -> plots -> checkpoint.
# Physics target: baseline equal-mass, non-spinning, boosted binary black-hole initial data
# in Zhou et al., "Solving Hamiltonian Constraint Equation with Physics-Informed Neural Networks".

import os
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

#%% Reproducibility and device
# Same idea as your Navier-Stokes code: fix NumPy and PyTorch seeds before sampling/training.
np.random.seed(42)
torch.manual_seed(42)

# Use CUDA if available. CPU training is possible but slow because the loss needs second derivatives.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
print(f"Using device: {device}")

#%% Problem setup: simplest example in the paper
# Paper baseline example, Fig. 2 and Table I:
#   m+ = m- = 0.5
#   punctures on x-axis at x = ±3
#   Py+ = -Py- = 0.2
#   spins are zero
# This is non-spinning but not time-symmetric, because the holes have nonzero linear momenta.

R_MAX = 30.0                 # Paper Eq. (8): spherical domain boundary r = Rmax, with Rmax = 30.
PUNCTURE_SEPARATION = 6.0    # x = ±3, so coordinate separation is 6.

masses_np = np.array([0.5, 0.5], dtype=np.float32)
positions_np = np.array([
    [ 3.0, 0.0, 0.0],       # plus puncture
    [-3.0, 0.0, 0.0],       # minus puncture
], dtype=np.float32)

momenta_np = np.array([
    [0.0,  0.2, 0.0],       # P_plus^y = +0.2
    [0.0, -0.2, 0.0],       # P_minus^y = -0.2
], dtype=np.float32)

spins_np = np.zeros((2, 3), dtype=np.float32)   # non-spinning case

masses = torch.tensor(masses_np, dtype=dtype, device=device)
positions = torch.tensor(positions_np, dtype=dtype, device=device)
momenta = torch.tensor(momenta_np, dtype=dtype, device=device)
spins = torch.tensor(spins_np, dtype=dtype, device=device)

#%% Run mode and hyperparameters
# The paper uses N_Ω = 20000 interior points, N_∂Ω = 8000 boundary points,
# 5000 Adam steps, 3 hidden layers, 64 neurons, SiLU activation, lr = 3e-4,
# c = 0.2, w2 = 1, winf = 0, wrob = 1 for the baseline example.
#
# Full paper-size training is expensive. Start with RUN_MODE = "debug" to check the pipeline,
# then switch to RUN_MODE = "paper" for a faithful run.

RUN_MODE = "debug"     # "debug" or "paper"

if RUN_MODE == "paper":
    N_INTERIOR = 20_000
    N_BOUNDARY = 8_000
    N_EPOCHS = 5_000
    PRINT_EVERY = 100
    KAPPA_VOLUME_POINTS = 200_000     # Paper: 5e6. This value is cheaper but still quasi-MC.
    KAPPA_BOUNDARY_POINTS = 20_000    # Paper: 5e4.
else:
    N_INTERIOR = 4_000
    N_BOUNDARY = 1_500
    N_EPOCHS = 500
    PRINT_EVERY = 25
    KAPPA_VOLUME_POINTS = 30_000
    KAPPA_BOUNDARY_POINTS = 5_000

LEARNING_RATE = 3.0e-4
C_CORRECTION = 0.2       # c in paper Eq. (5), baseline example.
W2 = 1.0                 # w2 in paper Eq. (12), baseline example.
W_INF = 0.0              # winf = 0 for Fig. 2 baseline; code supports nonzero values.
BETA = 2.5               # beta in paper Eq. (14); irrelevant when W_INF = 0.
W_ROB = 1.0              # wrob in paper Eq. (15), baseline example.
EMA_ALPHA = 0.99         # alpha in paper Eq. (16). The paper defines it as tunable.
EMA_EPS = 1.0e-12

# Numerical regularisation used only to avoid division by exactly zero at a puncture.
# Random collocation almost never hits a puncture exactly, but this makes the code robust.
R_EPS = 1.0e-6

# Optional evaluation against a TwoPunctures dataset from the previous tutorial.
# Expected .npz fields: coords with shape (N,3), u with shape (N,).
TP_REFERENCE_FILE = "twopunctures_paper_simple.npz"

#%% Sampling functions
# Paper methodology: choose random interior points in Ω and random boundary points on ∂Ω.
# These are collocation points, not supervised training pairs.

def sample_uniform_ball(n_points, radius, device, dtype):
    """Uniform random samples in a 3D ball of radius R.

    This implements uniform volume sampling in Ω. The r ~ U^(1/3) factor is needed
    because the volume element is r^2 dr dΩ.
    """
    dirs = torch.randn(n_points, 3, device=device, dtype=dtype)
    dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True).clamp_min(R_EPS)
    radii = radius * torch.rand(n_points, 1, device=device, dtype=dtype).pow(1.0 / 3.0)
    return radii * dirs


def sample_uniform_sphere(n_points, radius, device, dtype):
    """Uniform random samples on the spherical boundary ∂Ω."""
    dirs = torch.randn(n_points, 3, device=device, dtype=dtype)
    dirs = dirs / torch.linalg.norm(dirs, dim=1, keepdim=True).clamp_min(R_EPS)
    return radius * dirs


def sobol_ball(n_points, radius, device, dtype, seed=1234):
    """Sobol quasi-Monte Carlo samples in the ball, used for kappa as in paper Eq. (10)."""
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    u = engine.draw(n_points).to(device=device, dtype=dtype).clamp(1.0e-12, 1.0 - 1.0e-12)

    r = radius * u[:, 0:1].pow(1.0 / 3.0)
    cos_theta = 2.0 * u[:, 1:2] - 1.0
    sin_theta = torch.sqrt((1.0 - cos_theta**2).clamp_min(0.0))
    phi = 2.0 * math.pi * u[:, 2:3]

    x = r * sin_theta * torch.cos(phi)
    y = r * sin_theta * torch.sin(phi)
    z = r * cos_theta
    return torch.cat([x, y, z], dim=1)


def sobol_sphere(n_points, radius, device, dtype, seed=5678):
    """Sobol quasi-Monte Carlo samples on the sphere, used for kappa as in paper Eq. (10)."""
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    u = engine.draw(n_points).to(device=device, dtype=dtype).clamp(1.0e-12, 1.0 - 1.0e-12)

    cos_theta = 2.0 * u[:, 0:1] - 1.0
    sin_theta = torch.sqrt((1.0 - cos_theta**2).clamp_min(0.0))
    phi = 2.0 * math.pi * u[:, 1:2]

    x = radius * sin_theta * torch.cos(phi)
    y = radius * sin_theta * torch.sin(phi)
    z = radius * cos_theta
    return torch.cat([x, y, z], dim=1)

#%% Generate collocation points
# Paper: before training, choose N_Ω = 20000 inner points and N_∂Ω = 8000 boundary points.
# Your Navier-Stokes code sampled a subset of training points before the training loop;
# this block plays the same role, except the samples are physics collocation points.

X_interior_base = sample_uniform_ball(N_INTERIOR, R_MAX, device, dtype).detach()
X_boundary_base = sample_uniform_sphere(N_BOUNDARY, R_MAX, device, dtype).detach()

# Scale bounds for the MLP input. The physical coordinates are x,y,z, but the network
# sees x/Rmax, y/Rmax, z/Rmax, which keeps inputs O(1).
lbs = torch.tensor([-R_MAX, -R_MAX, -R_MAX], dtype=dtype, device=device)
ubs = torch.tensor([ R_MAX,  R_MAX,  R_MAX], dtype=dtype, device=device)

#%% Tensor calculus helpers
# Same role as grad() in your Navier-Stokes file: a thin wrapper around torch.autograd.grad.
# Here we need first derivatives for the Robin boundary condition and second derivatives
# for the Laplacian Δu in paper Eq. (4).

def grad(outputs, inputs, create_graph=True):
    derivative = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return derivative


def laplacian(scalar_field, coords):
    """Compute Δf = f_xx + f_yy + f_zz for scalar_field(coords)."""
    g = grad(scalar_field, coords, create_graph=True)
    f_x = g[:, 0:1]
    f_y = g[:, 1:2]
    f_z = g[:, 2:3]

    f_xx = grad(f_x, coords, create_graph=True)[:, 0:1]
    f_yy = grad(f_y, coords, create_graph=True)[:, 1:2]
    f_zz = grad(f_z, coords, create_graph=True)[:, 2:3]
    return f_xx + f_yy + f_zz

#%% Physics: puncture distances, singular conformal factor and Bowen-York Kbar
# Paper Eqs. (1)-(4): assume conformal flatness γ_ij = ψ^4 f_ij and maximal slicing K=0.
# The conformal factor is ψ = 1 + Σ m_n/(2 r_n) + u.
# The Bowen-York extrinsic curvature Kbar_ij solves the momentum constraint analytically.

def safe_radius(vec):
    return torch.sqrt(torch.sum(vec**2, dim=1, keepdim=True).clamp_min(R_EPS**2))


def psi_singular(coords):
    """ψ_sing = 1 + Σ m_n/(2 r_n), paper Eq. (3) without u."""
    psi = torch.ones(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)
    for n in range(2):
        r_n = safe_radius(coords - positions[n])
        psi = psi + masses[n] / (2.0 * r_n)
    return psi


def bowen_york_kbar(coords):
    """Compute Kbar_ij from paper Eq. (2).

    For this specific baseline case spins are zero, but the spin term is included so the
    function also matches the notation of the paper for later experiments.
    """
    n_points = coords.shape[0]
    delta = torch.eye(3, dtype=coords.dtype, device=coords.device).view(1, 3, 3)
    kbar = torch.zeros(n_points, 3, 3, dtype=coords.dtype, device=coords.device)

    for n in range(2):
        x_n = coords - positions[n]                         # vector from puncture to field point
        r_n = safe_radius(x_n)
        r2 = r_n**2

        P_n = momenta[n]
        S_n = spins[n]

        P_dot_x = torch.sum(x_n * P_n.view(1, 3), dim=1, keepdim=True)

        x_i = x_n.view(n_points, 3, 1)
        x_j = x_n.view(n_points, 1, 3)
        P_i = P_n.view(1, 3, 1)
        P_j = P_n.view(1, 1, 3)

        # Linear momentum part of paper Eq. (2):
        # 3/(2 r^3) [x_i P_j + x_j P_i - (δ_ij - x_i x_j/r^2)(P·x)]
        momentum_term = (
            x_i * P_j
            + x_j * P_i
            - (delta - x_i * x_j / r2.view(n_points, 1, 1)) * P_dot_x.view(n_points, 1, 1)
        )
        kbar = kbar + (3.0 / (2.0 * r_n.view(n_points, 1, 1)**3)) * momentum_term

        # Spin part of paper Eq. (2):
        # 3/r^5 [(S × x)_i x_j + (S × x)_j x_i]
        if torch.linalg.norm(S_n).item() > 0.0:
            S_cross_x = torch.cross(S_n.view(1, 3).expand_as(x_n), x_n, dim=1)
            spin_i = S_cross_x.view(n_points, 3, 1)
            spin_j = S_cross_x.view(n_points, 1, 3)
            spin_term = spin_i * x_j + spin_j * x_i
            kbar = kbar + (3.0 / r_n.view(n_points, 1, 1)**5) * spin_term

    return kbar


def kbar_squared(coords):
    """Kbar_ij Kbar^ij using the flat conformal metric for index contraction."""
    kbar = bowen_york_kbar(coords)
    return torch.sum(kbar * kbar, dim=(1, 2), keepdim=True)

#%% Analytical guidance u_g = u_P + u_J + u_c
# Paper Eq. (5)-(7): the network does not learn u directly. It learns h_theta, which
# modulates an approximate analytical solution u_g.
#
# For the non-spinning case, u_J = 0 and u_c = 0, so u_g is the superposition of the
# boosted-puncture term u_P for the two holes. The formula below is from the paper's Ref. [40]
# and is the same approximation the paper says it uses for guidance.

def boosted_puncture_guide(coords):
    """Approximate analytical guide u_g for non-spinning boosted punctures.

    Implements u_g = Σ u_P for the two punctures. In this baseline case S=0, so the
    spin contribution u_J and boost-spin cross term u_c vanish.
    """
    total = torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)

    for n in range(2):
        m = masses[n]
        P = momenta[n]
        P_mag = torch.linalg.norm(P)

        # If a puncture has zero momentum, its boost guide contribution is zero.
        if P_mag.item() == 0.0:
            continue

        x_n = coords - positions[n]
        r_n = safe_radius(x_n)

        # Dimensionless variables used in the analytical approximation.
        R = (2.0 * r_n / m).clamp_min(1.0e-5)
        ell = 1.0 / (1.0 + R)

        r_hat = x_n / r_n
        P_hat = P / P_mag
        mu_P = torch.sum(r_hat * P_hat.view(1, 3), dim=1, keepdim=True).clamp(-1.0, 1.0)
        legendre_P2 = 0.5 * (3.0 * mu_P**2 - 1.0)

        # u_P = Ptilde^2 [u0^P + u2^P P2(mu_P)]
        P_tilde = 2.0 * P_mag / m

        u0_P = (5.0 / 32.0) * (
            ell
            - 2.0 * ell**2
            + 2.0 * ell**3
            - ell**4
            + 0.2 * ell**5
        )

        numerator = (
            15.0 * ell
            + 132.0 * ell**2
            + 53.0 * ell**3
            + 96.0 * ell**4
            + 82.0 * ell**5
            + 84.0 * ell**5 / R
            + 84.0 * torch.log(ell) / R**2
        )
        u2_P = numerator / (80.0 * R)

        total = total + P_tilde**2 * (u0_P + u2_P * legendre_P2)

    return total


def compute_window_constants():
    """Estimate u_min and u_max in paper Eq. (7) using the fixed training points."""
    with torch.no_grad():
        guide_values = boosted_puncture_guide(torch.cat([X_interior_base, X_boundary_base], dim=0))
        u_min = guide_values.min().detach()
        u_max = guide_values.max().detach()
    return u_min, u_max

u_g_min, u_g_max = compute_window_constants()
print(f"Estimated guide range for window W: u_min = {u_g_min.item():.6e}, u_max = {u_g_max.item():.6e}")


def window_function(coords):
    """Paper Eq. (7): W(x) = (u_g - u_min)/(u_max - u_min)."""
    ug = boosted_puncture_guide(coords)
    denom = (u_g_max - u_g_min).clamp_min(1.0e-12)
    return (ug - u_g_min) / denom

#%% Estimate kappa from the divergence-theorem condition
# Paper Eqs. (9)-(10): kappa adjusts the global scale of u_g before training h_theta.
# The paper uses Sobol quasi-Monte Carlo with 5e6 volume points and 5e4 boundary points.
# For a personal machine, the defaults here are smaller; increase the constants above for accuracy.

def estimate_kappa():
    print("Estimating kappa from paper Eq. (10)...")

    # Boundary integral: -∮ ∇u_g · n dS
    Xb = sobol_sphere(KAPPA_BOUNDARY_POINTS, R_MAX, device, dtype, seed=1001).detach().requires_grad_(True)
    ug_b = boosted_puncture_guide(Xb)
    grad_ug_b = grad(ug_b, Xb, create_graph=False)
    n_vec = Xb / R_MAX
    du_dn = torch.sum(grad_ug_b * n_vec, dim=1, keepdim=True)
    surface_area = 4.0 * math.pi * R_MAX**2
    boundary_integral = surface_area * du_dn.mean()
    left_coeff = (-boundary_integral).detach()

    # Volume samples for RHS(kappa).
    Xv = sobol_ball(KAPPA_VOLUME_POINTS, R_MAX, device, dtype, seed=2002).detach()
    with torch.no_grad():
        ug_v = boosted_puncture_guide(Xv)
        psi_sing_v = psi_singular(Xv)
        k2_v = kbar_squared(Xv)
        volume = (4.0 / 3.0) * math.pi * R_MAX**3

    def rhs(kappa_value):
        k = torch.tensor(float(kappa_value), dtype=dtype, device=device)
        with torch.no_grad():
            psi_trial = (psi_sing_v + k * ug_v).clamp_min(1.0e-8)
            source = 0.125 * k2_v * psi_trial.pow(-7.0)
            return volume * source.mean()

    def f(kappa_value):
        return (torch.tensor(float(kappa_value), dtype=dtype, device=device) * left_coeff - rhs(kappa_value)).item()

    # Robust scalar bracketing/bisection. kappa is only a structural scale, not a trainable parameter.
    lo, hi = 0.0, 1.0
    f_lo, f_hi = f(lo), f(hi)
    while f_hi < 0.0 and hi < 1.0e3:
        hi *= 2.0
        f_hi = f(hi)

    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_hi < 0.0:
        print("Warning: kappa root was not bracketed. Falling back to kappa = 1.0.")
        return torch.tensor(1.0, dtype=dtype, device=device)

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if f_mid > 0.0:
            hi = mid
        else:
            lo = mid

    kappa = 0.5 * (lo + hi)
    print(f"Estimated kappa = {kappa:.6e}")
    return torch.tensor(kappa, dtype=dtype, device=device)

kappa = estimate_kappa()

#%% Define model
# Paper strategy summary: 3 hidden layers, 64 neurons per layer, SiLU activation.
# Input: scaled spatial coordinates (x/Rmax, y/Rmax, z/Rmax).
# Output: h_theta(x), a latent scalar correction field. The physical prediction u_theta is built by Eq. (5).

model = nn.Sequential(
    nn.Linear(3, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 1),
).to(device=device, dtype=dtype)


def scale_coords(coords):
    """Scale physical coordinates to [-1,1], analogous to your Navier-Stokes scaling block."""
    return 2.0 * (coords - lbs) / (ubs - lbs) - 1.0


def hard_enforcement_ansatz(coords):
    """Paper Eq. (5): u_theta = kappa u_g [1 + c W tanh(h_theta)]."""
    h_theta = model(scale_coords(coords))
    ug = boosted_puncture_guide(coords)
    W = window_function(coords)
    u_theta = kappa * ug * (1.0 + C_CORRECTION * W * torch.tanh(h_theta))
    psi_theta = psi_singular(coords) + u_theta
    return h_theta, u_theta, psi_theta

#%% Compute residuals
# Paper Eq. (4): residual R = Δu + 1/8 ψ^{-7} Kbar_ij Kbar^ij.
# Paper Eq. (8): boundary residual R_B = n·∇u + u/r on the spherical boundary.

def pde_residual(coords):
    """Interior Hamiltonian-constraint residual."""
    coords = coords.detach().clone().requires_grad_(True)
    _, u_theta, psi_theta = hard_enforcement_ansatz(coords)

    lap_u = laplacian(u_theta, coords)
    source = 0.125 * psi_theta.clamp_min(1.0e-8).pow(-7.0) * kbar_squared(coords)
    residual = lap_u + source
    return residual, u_theta, psi_theta


def boundary_residual(coords):
    """Robin boundary residual on r = Rmax."""
    coords = coords.detach().clone().requires_grad_(True)
    _, u_theta, _ = hard_enforcement_ansatz(coords)
    grad_u = grad(u_theta, coords, create_graph=True)

    r = safe_radius(coords)
    n_vec = coords / r
    du_dn = torch.sum(grad_u * n_vec, dim=1, keepdim=True)

    # On a spherical boundary, ∂r/∂n = 1, so Eq. (8) becomes n·∇u + u/r = 0.
    residual_b = du_dn + u_theta / r
    return residual_b

#%% Define loss function
# Paper Eqs. (11)-(15): total loss = PDE residual loss + Robin boundary loss.
# PDE loss combines L2 and soft-L∞. The baseline paper example sets W_INF = 0, so only L2 is active.

def soft_linf_loss(residual, beta):
    """Paper Eq. (14): smooth log-sum-exp approximation to L∞."""
    abs_r = torch.abs(residual).reshape(-1)
    n = abs_r.numel()
    return (torch.logsumexp(beta * abs_r, dim=0) - math.log(n)) / beta


def raw_loss_terms(X_interior, X_boundary):
    """Return the three raw terms L2, soft-L∞, LBC before EMA balancing."""
    R, _, _ = pde_residual(X_interior)
    Rb = boundary_residual(X_boundary)

    L2 = torch.mean(R**2)
    LINF = soft_linf_loss(R, BETA) if W_INF != 0.0 else torch.zeros_like(L2)
    LBC = torch.mean(Rb**2)
    return L2, LINF, LBC

#%% Loss balancing strategy
# Paper Eqs. (16)-(17): each loss component is divided by an exponential moving average
# to prevent one large-magnitude term from dominating training.

ema_L2 = None
ema_LINF = None
ema_LBC = None


def balanced_total_loss(L2, LINF, LBC):
    """Apply EMA loss balancing and return the weighted total loss."""
    global ema_L2, ema_LINF, ema_LBC

    with torch.no_grad():
        if ema_L2 is None:
            ema_L2 = L2.detach()
            ema_LINF = LINF.detach() + EMA_EPS
            ema_LBC = LBC.detach()
        else:
            ema_L2 = EMA_ALPHA * ema_L2 + (1.0 - EMA_ALPHA) * L2.detach()
            ema_LINF = EMA_ALPHA * ema_LINF + (1.0 - EMA_ALPHA) * LINF.detach()
            ema_LBC = EMA_ALPHA * ema_LBC + (1.0 - EMA_ALPHA) * LBC.detach()

    L2_tilde = L2 / (ema_L2 + EMA_EPS)
    LINF_tilde = LINF / (ema_LINF + EMA_EPS)
    LBC_tilde = LBC / (ema_LBC + EMA_EPS)

    total = W2 * L2_tilde + W_INF * LINF_tilde + W_ROB * LBC_tilde
    return total, L2_tilde, LINF_tilde, LBC_tilde

#%% Training cycle
# This mirrors your Navier-Stokes training loop: forward pass -> loss -> zero_grad -> backward -> step -> record lists.
# Difference: there is no supervised data loss. The comparison target during training is zero residual.

L2_list = []
Linf_list = []
Lbc_list = []
total_loss_list = []
L2_balanced_list = []
Lbc_balanced_list = []

adam_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(N_EPOCHS):
    model.train()

    # Paper uses fixed random collocation points. We reuse the same sampled points each epoch.
    X_interior = X_interior_base
    X_boundary = X_boundary_base

    L2, LINF, LBC = raw_loss_terms(X_interior, X_boundary)
    total_loss, L2_tilde, LINF_tilde, LBC_tilde = balanced_total_loss(L2, LINF, LBC)

    adam_optimizer.zero_grad()
    total_loss.backward()
    adam_optimizer.step()

    L2_list.append(L2.item())
    Linf_list.append(LINF.item())
    Lbc_list.append(LBC.item())
    total_loss_list.append(total_loss.item())
    L2_balanced_list.append(L2_tilde.item())
    Lbc_balanced_list.append(LBC_tilde.item())

    if epoch % PRINT_EVERY == 0:
        print(
            f"epoch {epoch:6d} | "
            f"total={total_loss.item():.4e} | "
            f"L2={L2.item():.4e} | "
            f"LBC={LBC.item():.4e} | "
            f"kappa={kappa.item():.4e}"
        )

print("Adam finished")

#%% Optional evaluation against TwoPunctures reference data
# The paper does not train on TwoPunctures data. It only validates by comparing u_PINN
# with u_TP on reference grid points using L2RE, paper Eq. (18).

def evaluate_against_twopunctures(npz_path):
    path = Path(npz_path)
    if not path.exists():
        print(f"No TwoPunctures reference file found at {path}. Skipping L2RE evaluation.")
        return None

    data = np.load(path)
    coords_np = data["coords"].astype(np.float32)
    u_tp_np = data["u"].astype(np.float32).reshape(-1, 1)

    # Keep evaluation memory controlled.
    max_eval = min(coords_np.shape[0], 50_000)
    idx = np.linspace(0, coords_np.shape[0] - 1, max_eval, dtype=int)

    coords = torch.tensor(coords_np[idx], dtype=dtype, device=device)
    u_tp = torch.tensor(u_tp_np[idx], dtype=dtype, device=device)

    model.eval()
    with torch.no_grad():
        _, u_pred, _ = hard_enforcement_ansatz(coords)
        l2re = torch.sqrt(torch.sum((u_pred - u_tp)**2) / torch.sum(u_tp**2))

    print(f"L2RE against TwoPunctures reference = {l2re.item():.6e}")
    return l2re.item()

l2re_tp = evaluate_against_twopunctures(TP_REFERENCE_FILE)

#%% Plot losses
# Same style as your Navier-Stokes plots, but the terms are Hamiltonian residual and Robin BC losses.

epochs = np.arange(1, len(total_loss_list) + 1)

fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=150)
ax.plot(epochs, total_loss_list, label="Balanced total loss", zorder=2)
ax.plot(epochs, L2_list, label="Raw PDE L2 loss", alpha=0.6, zorder=1)
ax.plot(epochs, Lbc_list, label="Raw Robin BC loss", alpha=0.6, zorder=1)
if W_INF != 0.0:
    ax.plot(epochs, Linf_list, label="Raw soft-Linf loss", alpha=0.6, zorder=1)
ax.set_xlabel("Epoch", fontsize=14)
ax.set_ylabel("Loss", fontsize=14)
ax.set_yscale("log")
ax.set_title("Hamiltonian-constraint PINN training", fontsize=16)
ax.tick_params(labelsize=11, which="both", top=True, right=True, direction="in")
ax.grid(alpha=0.25)
ax.legend(loc="best", fontsize=10)
plt.tight_layout()
plt.savefig("bbh_pinn_losses.png", dpi=200)
plt.show()

#%% Plot predicted u along the x-axis
# This reproduces the kind of 1D diagnostic shown in paper Fig. 2(c), but without needing TP data.

x_line = torch.linspace(-R_MAX, R_MAX, 600, dtype=dtype, device=device).view(-1, 1)
y_line = torch.zeros_like(x_line)
z_line = torch.zeros_like(x_line)
coords_line = torch.cat([x_line, y_line, z_line], dim=1)

# Avoid evaluating exactly at the punctures x = ±3.
mask = (torch.abs(x_line[:, 0] - 3.0) > 1.0e-3) & (torch.abs(x_line[:, 0] + 3.0) > 1.0e-3)
coords_line = coords_line[mask]

model.eval()
with torch.no_grad():
    _, u_line, psi_line = hard_enforcement_ansatz(coords_line)

fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=150)
ax.plot(coords_line[:, 0].detach().cpu().numpy(), u_line[:, 0].detach().cpu().numpy(), label="PINN $u_\\theta$")
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("u", fontsize=14)
ax.set_title("PINN correction field along x-axis", fontsize=16)
ax.tick_params(labelsize=11, which="both", top=True, right=True, direction="in")
ax.grid(alpha=0.25)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("bbh_pinn_u_x_axis.png", dpi=200)
plt.show()

#%% Save training progress
# Same idea as your Navier-Stokes checkpoint block: save model, constants, losses and hyperparameters.

checkpoint = {
    "model_state_dict": model.state_dict(),
    "kappa": kappa.detach().cpu(),
    "u_g_min": u_g_min.detach().cpu(),
    "u_g_max": u_g_max.detach().cpu(),
    "lbs": lbs.detach().cpu(),
    "ubs": ubs.detach().cpu(),
    "masses": masses.detach().cpu(),
    "positions": positions.detach().cpu(),
    "momenta": momenta.detach().cpu(),
    "spins": spins.detach().cpu(),
    "R_MAX": R_MAX,
    "C_CORRECTION": C_CORRECTION,
    "W2": W2,
    "W_INF": W_INF,
    "BETA": BETA,
    "W_ROB": W_ROB,
    "EMA_ALPHA": EMA_ALPHA,
    "RUN_MODE": RUN_MODE,
    "N_INTERIOR": N_INTERIOR,
    "N_BOUNDARY": N_BOUNDARY,
    "N_EPOCHS": N_EPOCHS,
    "L2_list": L2_list,
    "Linf_list": Linf_list,
    "Lbc_list": Lbc_list,
    "total_loss_list": total_loss_list,
    "L2_balanced_list": L2_balanced_list,
    "Lbc_balanced_list": Lbc_balanced_list,
    "l2re_tp": l2re_tp,
}

torch.save(checkpoint, "bbh_hamiltonian_pinn_checkpoint.pt")
print("Saved checkpoint: bbh_hamiltonian_pinn_checkpoint.pt")
print("Saved plots: bbh_pinn_losses.png, bbh_pinn_u_x_axis.png")

#%% Load checkpoint template
# This block is deliberately similar to your Navier-Stokes load-checkpoint section.
# Uncomment and run in a fresh session if you want to reload the trained model.

"""
checkpoint = torch.load("bbh_hamiltonian_pinn_checkpoint.pt", map_location=device)

model = nn.Sequential(
    nn.Linear(3, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 64),
    nn.SiLU(),
    nn.Linear(64, 1),
).to(device=device, dtype=dtype)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

kappa = checkpoint["kappa"].to(device=device, dtype=dtype)
u_g_min = checkpoint["u_g_min"].to(device=device, dtype=dtype)
u_g_max = checkpoint["u_g_max"].to(device=device, dtype=dtype)

print("Reloaded model. kappa =", kappa.item())
"""
