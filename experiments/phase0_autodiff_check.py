"""
phase0_autodiff_check.py
Phase 0 -- Autograd Sanity Check

Before implementing the PDE residual, verify that PyTorch autograd
produces correct second-order derivatives on known analytic functions.

Tests:
    1. f(x) = sin(x)   -> d2f/dx2 = -sin(x)
    2. f(x,y) = sin(pi*x)*sin(pi*y)  -> Laplacian(f) = -2*pi^2*f
    3. Mixed: f(x,y,t) = sin(pi*x)*sin(pi*y)*exp(-t)
              -> df/dt = -f,  Laplacian(f) = -2*pi^2*f
    4. PDE residual == 0 for the exact Fourier-mode solution

Run this file first. All tests must PASS before touching the PDE code.

Usage:
    python experiments/phase0_autodiff_check.py
"""

import sys
import torch
import numpy as np

PASS_SYMBOL = "[PASS]"
FAIL_SYMBOL = "[FAIL]"


def check(name: str, computed: torch.Tensor, expected: torch.Tensor,
          tol: float = 1e-5) -> bool:
    """Compare computed vs expected and print result."""
    err = (computed - expected).abs().max().item()
    ok = err < tol
    symbol = PASS_SYMBOL if ok else FAIL_SYMBOL
    print(f"  {symbol}  {name:55s}  max_err = {err:.2e}  (tol={tol:.0e})")
    return ok


# Test 1: d2(sin(x))/dx2 = -sin(x)   (1D)

def test_1d_second_derivative():
    print("\n--- Test 1: d2(sin(x))/dx2 = -sin(x) ---")
    x = torch.linspace(0, 2 * float(np.pi), 200, dtype=torch.float64).unsqueeze(1)
    x.requires_grad_(True)

    f = torch.sin(x)  # (N, 1)

    # First derivative via autograd
    df_dx, = torch.autograd.grad(f, x,
                                  grad_outputs=torch.ones_like(f),
                                  create_graph=True)
    # Second derivative
    d2f_dx2, = torch.autograd.grad(df_dx, x,
                                    grad_outputs=torch.ones_like(df_dx),
                                    create_graph=False)

    expected = -torch.sin(x.detach())
    return check("d2(sin(x))/dx2 == -sin(x)", d2f_dx2.detach(), expected)


# Test 2: Laplacian of sin(pi*x)*sin(pi*y) = -2*pi^2*sin(pi*x)*sin(pi*y)

def test_2d_laplacian():
    print("\n--- Test 2: Laplacian[sin(pi*x)*sin(pi*y)] = -2*pi^2*f ---")
    n = 100
    x_np, y_np = np.meshgrid(np.linspace(0.01, 0.99, n),
                               np.linspace(0.01, 0.99, n))
    x = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float64, requires_grad=True)
    y = torch.tensor(y_np.reshape(-1, 1), dtype=torch.float64, requires_grad=True)

    pi = torch.tensor(float(np.pi), dtype=torch.float64)
    f = torch.sin(pi * x) * torch.sin(pi * y)

    # df/dx and df/dy
    ones = torch.ones_like(f)
    df_dx, = torch.autograd.grad(f, x, grad_outputs=ones, create_graph=True)
    df_dy, = torch.autograd.grad(f, y, grad_outputs=ones, create_graph=True)

    # Second derivatives
    d2f_dxx, = torch.autograd.grad(df_dx, x, grad_outputs=torch.ones_like(df_dx), create_graph=False)
    d2f_dyy, = torch.autograd.grad(df_dy, y, grad_outputs=torch.ones_like(df_dy), create_graph=False)

    laplacian = d2f_dxx + d2f_dyy
    expected  = -2.0 * pi ** 2 * torch.sin(pi * x.detach()) * torch.sin(pi * y.detach())

    return check("Laplacian[sin(pi*x)*sin(pi*y)] == -2*pi^2*f",
                 laplacian.detach(), expected, tol=1e-4)


# Test 3: Mixed (x,y,t) -- verify df/dt and Laplacian separately
#         f = sin(pi*x) * sin(pi*y) * exp(-t)
#         df/dt     = -f
#         Laplacian = -2*pi^2*f

def test_3d_mixed():
    print("\n--- Test 3: Mixed (x,y,t): df/dt and Laplacian on sin(pi*x)*sin(pi*y)*exp(-t) ---")
    n = 50
    pts = np.random.default_rng(0).uniform(0.05, 0.95, (n * n, 3))
    x = torch.tensor(pts[:, 0:1], dtype=torch.float64, requires_grad=True)
    y = torch.tensor(pts[:, 1:2], dtype=torch.float64, requires_grad=True)
    t = torch.tensor(pts[:, 2:3], dtype=torch.float64, requires_grad=True)

    pi = torch.tensor(float(np.pi), dtype=torch.float64)
    f  = torch.sin(pi * x) * torch.sin(pi * y) * torch.exp(-t)

    ones = torch.ones_like(f)

    # df/dt
    df_dt, = torch.autograd.grad(f, t, grad_outputs=ones, create_graph=True)
    # df/dx, df/dy
    df_dx, = torch.autograd.grad(f, x, grad_outputs=ones, create_graph=True)
    df_dy, = torch.autograd.grad(f, y, grad_outputs=ones, create_graph=True)
    # Second spatial derivatives
    d2f_dxx, = torch.autograd.grad(df_dx, x, grad_outputs=torch.ones_like(df_dx), create_graph=False)
    d2f_dyy, = torch.autograd.grad(df_dy, y, grad_outputs=torch.ones_like(df_dy), create_graph=False)

    f_det = f.detach()
    ok1 = check("df/dt[sin(pi*x)*sin(pi*y)*exp(-t)] == -f",
                df_dt.detach(), -f_det, tol=1e-4)
    laplacian = d2f_dxx + d2f_dyy
    ok2 = check("Laplacian[sin(pi*x)*sin(pi*y)*exp(-t)] == -2*pi^2*f",
                laplacian.detach(), -2.0 * pi ** 2 * f_det, tol=1e-4)
    return ok1 and ok2


# Test 4: PDE residual == 0 for the exact Fourier-mode solution
#
#   u_exact(x,y,t) = exp(-2*pi^2*alpha*t) * sin(pi*x) * sin(pi*y)
#   du/dt - alpha*(d2u/dx2 + d2u/dy2) = 0  (by construction)
#
# This directly validates the autograd pattern used in losses.py.

def test_pde_residual_exact_solution():
    print("\n--- Test 4: PDE residual == 0 for exact Fourier-mode solution ---")
    ALPHA = 0.01
    n = 200
    rng = np.random.default_rng(1)
    pts = rng.uniform(0.05, 0.95, (n, 3))
    x = torch.tensor(pts[:, 0:1], dtype=torch.float64, requires_grad=True)
    y = torch.tensor(pts[:, 1:2], dtype=torch.float64, requires_grad=True)
    t = torch.tensor(pts[:, 2:3], dtype=torch.float64, requires_grad=True)

    pi    = torch.tensor(float(np.pi), dtype=torch.float64)
    alpha = torch.tensor(ALPHA,        dtype=torch.float64)

    u = torch.exp(-2.0 * pi**2 * alpha * t) * torch.sin(pi * x) * torch.sin(pi * y)

    ones = torch.ones_like(u)
    du_dt,  = torch.autograd.grad(u,    t, grad_outputs=ones, create_graph=True)
    du_dx,  = torch.autograd.grad(u,    x, grad_outputs=ones, create_graph=True)
    du_dy,  = torch.autograd.grad(u,    y, grad_outputs=ones, create_graph=True)
    d2u_dxx,= torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=False)
    d2u_dyy,= torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), create_graph=False)

    residual = du_dt - alpha * (d2u_dxx + d2u_dyy)
    expected = torch.zeros_like(residual)

    return check("PDE residual == 0 for exact solution", residual.detach(), expected, tol=1e-6)


# Main

if __name__ == "__main__":
    print("=" * 65)
    print("Phase 0 -- PyTorch Autograd Sanity Check")
    print("=" * 65)

    results = [
        test_1d_second_derivative(),
        test_2d_laplacian(),
        test_3d_mixed(),
        test_pde_residual_exact_solution(),
    ]

    print("\n" + "=" * 65)
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"Results: {n_pass}/{len(results)} tests PASSED, {n_fail} FAILED")
    print("=" * 65)

    if n_fail > 0:
        print("\nAUTODIFF CHECK FAILED -- do NOT proceed to PDE implementation.")
        sys.exit(1)
    else:
        print("\nAll autograd checks PASSED -- safe to implement PDE losses.")
        sys.exit(0)
