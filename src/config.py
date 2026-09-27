"""
config.py
Single source of truth for all project parameters.
Modify here only; do not hard-code values elsewhere.

Project: PINN for 2D Transient Heat Equation
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = PROJECT_ROOT / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORT_DIR   = PROJECT_ROOT / "report"

# Create directories if they do not exist
for _d in [RESULTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR, REPORT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Physical Parameters (fixed for ALL experiments)

ALPHA   = 0.01   # Thermal diffusivity [m²/s] (or dimensionless)
T_FINAL = 0.5    # Final simulation time

# Gaussian hot-spot initial condition:  u(x,y,0) = A * exp(-((x-x0)²+(y-y0)²)/(2σ²))
IC_AMPLITUDE = 1.0   # A
IC_CENTER_X  = 0.5   # x0
IC_CENTER_Y  = 0.5   # y0
IC_SIGMA     = 0.1   # σ

# Boundary conditions: homogeneous Dirichlet (u = 0 on all walls)

# Characteristic diffusion length check (informational):
# sqrt(4 * ALPHA * T_FINAL) ≈ 0.141  → Gaussian spreads ~1.4σ; stays well away from walls
DIFFUSION_LENGTH = (4 * ALPHA * T_FINAL) ** 0.5  # ≈ 0.141

# Finite-Difference Solver Parameters
FD_NX     = 64          # Spatial grid points in x (including boundaries)
FD_NY     = 64          # Spatial grid points in y (including boundaries)
FD_SAFETY = 0.9         # Safety factor on CFL/stability timestep

# Derived FD quantities (computed here for reference; fd_solver.py also recomputes)
_dx = 1.0 / (FD_NX - 1)
_dy = 1.0 / (FD_NY - 1)
FD_DX = _dx
FD_DY = _dy
# FTCS 2D stability: r = alpha * dt / dx² ≤ 0.25
FD_DT_MAX  = 0.25 * _dx ** 2 / ALPHA          # Maximum stable dt
FD_DT      = FD_SAFETY * FD_DT_MAX            # Actual dt with safety factor
FD_N_STEPS = int(T_FINAL / FD_DT) + 1         # Number of time steps

# Time snapshots at which to save/compare solutions
EVAL_TIMES = [0.0, 0.1, 0.2, 0.3, 0.5]

# Validation grid sizes for convergence test
FD_VALIDATION_GRIDS = [32, 64, 96]

# PINN Architecture
PINN_INPUT_DIM   = 3          # (x, y, t)
PINN_OUTPUT_DIM  = 1          # û(x,y,t)
PINN_HIDDEN_DIMS = [64, 64, 64, 64]   # 4 hidden layers × 64 neurons
PINN_ACTIVATION  = "tanh"

# Collocation Point Counts (baseline)

N_COLLOC_INTERIOR = 5000   # Interior PDE residual points
N_COLLOC_IC       = 1000   # Initial condition points  (t = 0)
N_COLLOC_BC       = 1000   # Boundary condition points (250 per wall)

# Loss Weights
LAMBDA_PDE = 1.0
LAMBDA_IC  = 10.0
LAMBDA_BC  = 10.0

# Training Parameters

TRAIN_OPTIMIZER  = "Adam"
TRAIN_LR_INIT    = 1e-3       # Initial learning rate
TRAIN_LR_FINAL   = 1e-5       # Final LR (cosine decay target)
TRAIN_EPOCHS     = 20_000
TRAIN_SEED       = 42
TRAIN_CHECKPOINT_INTERVAL = 1_000   # Save checkpoint every N epochs
TRAIN_PRINT_INTERVAL      = 500     # Print loss every N epochs

# Early stopping: if total loss falls below this, stop training
TRAIN_EARLY_STOP_LOSS = 1e-6

# Sensitivity Experiment Settings (Tier 2)

SENSITIVITY_COLLOC_COUNTS = [2500, 5000, 10000]   # Interior PDE point counts to sweep
SENSITIVITY_ARCHITECTURES = [
    [32, 32, 32],         # Small: 3 × 32
    [64, 64, 64, 64],     # Baseline: 4 × 64
    [64, 64, 64, 64, 64], # Deep: 5 × 64
]

# Reproducibility
RANDOM_SEED = 42

# Hardware / Environment Note

HARDWARE_NOTE = "CPU-only (no CUDA). All experiments run on CPU."
