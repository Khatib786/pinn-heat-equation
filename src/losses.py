"""
losses.py
"""

import torch
from typing import Optional


def _grad(output: torch.Tensor, inp: torch.Tensor,
          create_graph: bool = True) -> torch.Tensor:
    """
    Compute ∂output/∂inp using torch.autograd.grad.

    Parameters
    ----------
    output       : scalar or (N,1) tensor — the quantity to differentiate
    inp          : tensor with requires_grad=True
    create_graph : True during training (allows higher-order grads)

    Returns
    -------
    Gradient tensor, same shape as inp.
    """
    grad, = torch.autograd.grad(
        outputs=output,
        inputs=inp,
        grad_outputs=torch.ones_like(output),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )
    return grad


def compute_pde_residual(model,
                          x_r: torch.Tensor,
                          y_r: torch.Tensor,
                          t_r: torch.Tensor,
                          alpha: float,
                          create_graph: bool = True) -> torch.Tensor:
    """
    Compute the PDE residual at interior collocation points.

        r(x,y,t) = ∂û/∂t - α(∂²û/∂x² + ∂²û/∂y²)

    For t normalized as t̃ = t / T_FINAL inside model.forward():
        ∂û/∂t   = (1/T_FINAL) × ∂û/∂t̃
        ∂²û/∂x², ∂²û/∂y²  — no scaling needed (x,y ∈ [0,1])

    Parameters
    ----------
    model   : HeatPINN (or any model with .forward(x, y, t) interface)
    x_r, y_r, t_r : collocation point coordinates, shape (N, 1),
                    all with requires_grad=True
    alpha   : thermal diffusivity
    create_graph : keep graph for backward (True during training)

    Returns
    -------
    residual : tensor of shape (N, 1)
    """
    u = model(x_r, y_r, t_r)          # (N, 1)  — forward pass

    # First derivatives
    u_t = _grad(u, t_r, create_graph)     # ∂û/∂t (raw, w.r.t. physical t)
    u_x = _grad(u, x_r, create_graph)    # ∂û/∂x
    u_y = _grad(u, y_r, create_graph)    # ∂û/∂y

    # Second spatial derivatives
    u_xx = _grad(u_x, x_r, create_graph)  # ∂²û/∂x²
    u_yy = _grad(u_y, y_r, create_graph)  # ∂²û/∂y²

    # Note: model normalizes t → t̃ = t/T internally, so ∂û/∂t here is already
    # w.r.t. physical t (the chain rule is absorbed into autograd since t is the
    # actual input we differentiate through). No manual scaling needed.

    residual = u_t - alpha * (u_xx + u_yy)
    return residual


def compute_ic_loss(model,
                    x_ic: torch.Tensor,
                    y_ic: torch.Tensor,
                    t_ic: torch.Tensor,
                    u0_target: torch.Tensor) -> torch.Tensor:
    """
    Initial condition loss:
        L_IC = (1/N_IC) Σ (û(x,y,0) - u0(x,y))²

    Parameters
    ----------
    x_ic, y_ic, t_ic : IC collocation points (t_ic should be all zeros)
    u0_target         : ground-truth IC values, shape (N_IC, 1)

    Returns
    -------
    Scalar MSE loss.
    """
    u_pred = model(x_ic, y_ic, t_ic)   # (N_IC, 1)
    return torch.mean((u_pred - u0_target) ** 2)


def compute_bc_loss(model,
                    x_bc: torch.Tensor,
                    y_bc: torch.Tensor,
                    t_bc: torch.Tensor) -> torch.Tensor:
    """
    Homogeneous Dirichlet BC loss:
        L_BC = (1/N_BC) Σ û(x_bc, y_bc, t_bc)²

    Points on all four walls; û should be zero there.

    Returns
    -------
    Scalar MSE loss.
    """
    u_pred = model(x_bc, y_bc, t_bc)   # (N_BC, 1)
    return torch.mean(u_pred ** 2)


def compute_total_loss(model,
                        x_r:  torch.Tensor, y_r:  torch.Tensor, t_r:  torch.Tensor,
                        x_ic: torch.Tensor, y_ic: torch.Tensor, t_ic: torch.Tensor,
                        u0_target: torch.Tensor,
                        x_bc: torch.Tensor, y_bc: torch.Tensor, t_bc: torch.Tensor,
                        alpha: float,
                        lambda_pde: float = 1.0,
                        lambda_ic:  float = 10.0,
                        lambda_bc:  float = 10.0) -> dict:
    """
    Compute the composite PINN loss:
        L = λ_PDE * L_PDE + λ_IC * L_IC + λ_BC * L_BC

    Returns a dict with individual components (for logging) and total.
    All returned tensors are scalar; call .item() to get Python floats.
    """
    l_pde = torch.mean(compute_pde_residual(model, x_r, y_r, t_r, alpha) ** 2)
    l_ic  = compute_ic_loss(model, x_ic, y_ic, t_ic, u0_target)
    l_bc  = compute_bc_loss(model, x_bc, y_bc, t_bc)

    total = lambda_pde * l_pde + lambda_ic * l_ic + lambda_bc * l_bc

    return {
        "total"   : total,
        "pde"     : l_pde,
        "ic"      : l_ic,
        "bc"      : l_bc,
    }

# Collocation point samplers


def sample_interior_points(n: int, t_final: float,
                            seed: Optional[int] = None) -> tuple:
    """
    Sample N interior collocation points uniformly in [0,1]² × [0,T_FINAL].
    Returns (x, y, t) tensors with requires_grad=True.
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    x = torch.rand(n, 1, generator=rng, dtype=torch.float32, requires_grad=False)
    y = torch.rand(n, 1, generator=rng, dtype=torch.float32, requires_grad=False)
    t = torch.rand(n, 1, generator=rng, dtype=torch.float32, requires_grad=False) * t_final

    # Enable grads for autograd differentiation
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    return x, y, t


def sample_ic_points(n: int, seed: Optional[int] = None) -> tuple:
    """
    Sample N initial-condition points: (x,y) uniform in [0,1]², t=0.
    Returns (x_ic, y_ic, t_ic, u0_func_placeholder) — caller must compute u0.
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed + 1000)

    x = torch.rand(n, 1, generator=rng, dtype=torch.float32)
    y = torch.rand(n, 1, generator=rng, dtype=torch.float32)
    t = torch.zeros(n, 1, dtype=torch.float32)
    return x, y, t


def sample_bc_points(n_per_wall: int, t_final: float,
                     seed: Optional[int] = None) -> tuple:
    """
    Sample N_per_wall points on each of the four domain walls.
    Total BC points = 4 × n_per_wall.

    Walls:
        x=0 : (0, y, t)    — left
        x=1 : (1, y, t)    — right
        y=0 : (x, 0, t)    — bottom
        y=1 : (x, 1, t)    — top

    Returns (x_bc, y_bc, t_bc) tensors, shape (4*n_per_wall, 1).
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed + 2000)

    def _rand(n):
        return torch.rand(n, 1, generator=rng, dtype=torch.float32)

    n = n_per_wall
    # Left wall:  x=0, y∈[0,1], t∈[0,T]
    x_left = torch.zeros(n, 1);  y_left = _rand(n);  t_left = _rand(n) * t_final
    # Right wall: x=1
    x_right = torch.ones(n, 1);  y_right = _rand(n); t_right = _rand(n) * t_final
    # Bottom wall: y=0
    x_bot = _rand(n);  y_bot = torch.zeros(n, 1);    t_bot = _rand(n) * t_final
    # Top wall: y=1
    x_top = _rand(n);  y_top = torch.ones(n, 1);     t_top = _rand(n) * t_final

    x_bc = torch.cat([x_left, x_right, x_bot, x_top], dim=0)
    y_bc = torch.cat([y_left, y_right, y_bot, y_top], dim=0)
    t_bc = torch.cat([t_left, t_right, t_bot, t_top], dim=0)

    return x_bc, y_bc, t_bc


def gaussian_ic_torch(x: torch.Tensor, y: torch.Tensor,
                       amplitude: float, cx: float, cy: float,
                       sigma: float) -> torch.Tensor:
    """Gaussian IC evaluated on torch tensors (no grad needed)."""
    return amplitude * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
