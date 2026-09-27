"""
evaluation.py
Quantitative error metrics comparing PINN predictions to the FD reference.
Metrics reported per time snapshot:
    - Relative L2 error       : ||û - u_FD||₂ / ||u_FD||₂
    - RMSE                    : sqrt(mean((û - u_FD)²))
    - Max absolute error      : max|û - u_FD|
    - PDE residual norm       : sqrt(mean(r²)) on a test grid
    - IC relative L2 error    : at t=0
    - BC max absolute error   : on boundary nodes
"""

import time
import numpy as np
import torch
from typing import Optional

# Basic scalar metrics


def relative_l2(u_pred: np.ndarray, u_ref: np.ndarray,
                eps: float = 1e-14) -> float:
    """Relative L2 error: ||u_pred - u_ref||₂ / ||u_ref||₂"""
    return float(np.linalg.norm(u_pred - u_ref) /
                 (np.linalg.norm(u_ref) + eps))


def rmse(u_pred: np.ndarray, u_ref: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((u_pred - u_ref) ** 2)))


def max_abs_error(u_pred: np.ndarray, u_ref: np.ndarray) -> float:
    """Maximum absolute pointwise error."""
    return float(np.max(np.abs(u_pred - u_ref)))

# PINN evaluation on FD grid


def evaluate_pinn_on_fd_grid(model: torch.nn.Module,
                              x_fd: np.ndarray,
                              y_fd: np.ndarray,
                              t_val: float,
                              device: str = "cpu") -> np.ndarray:
    """
    Evaluate the PINN on the same (Ny × Nx) grid used by the FD solver.

    Parameters
    ----------
    model  : trained HeatPINN
    x_fd   : 1D array of x-coordinates (length Nx)
    y_fd   : 1D array of y-coordinates (length Ny)
    t_val  : scalar time

    Returns
    -------
    u_pinn : 2D array of shape (Ny, Nx)
    """
    Nx, Ny = len(x_fd), len(y_fd)
    X, Y = np.meshgrid(x_fd, y_fd)   # each (Ny, Nx)
    x_flat = X.reshape(-1, 1).astype(np.float32)
    y_flat = Y.reshape(-1, 1).astype(np.float32)
    t_flat = np.full_like(x_flat, t_val)

    x_t = torch.tensor(x_flat, dtype=torch.float32)
    y_t = torch.tensor(y_flat, dtype=torch.float32)
    t_t = torch.tensor(t_flat, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        u_pred = model(x_t, y_t, t_t).numpy()   # (Ny*Nx, 1)

    return u_pred.reshape(Ny, Nx)


def evaluate_pde_residual_on_grid(model: torch.nn.Module,
                                   x_fd: np.ndarray,
                                   y_fd: np.ndarray,
                                   t_val: float,
                                   alpha: float) -> dict:
    """


    dict with 'residual_rms' (sqrt(mean(r^2))) and 'residual_max' (max|r|).
    """

    from losses import compute_pde_residual

    X, Y = np.meshgrid(x_fd, y_fd)   # (Ny, Nx)
    x_flat = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    y_flat = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
    t_flat = torch.full_like(x_flat, t_val, requires_grad=True)

    model.eval()   # eval mode (e.g. disables dropout if any) — grad still flows
    residual = compute_pde_residual(
        model, x_flat, y_flat, t_flat, alpha, create_graph=True
    ).detach().numpy()

    return {
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "residual_max": float(np.max(np.abs(residual))),
    }


def evaluate_at_times(model: torch.nn.Module,
                      fd_result: dict,
                      eval_times: list[float],
                      alpha: float,
                      device: str = "cpu") -> dict:
    """
    Compare PINN to FD reference at a list of time snapshots.

    """
    x_fd = fd_result["x"]
    y_fd = fd_result["y"]
    snapshots = fd_result["snapshots"]
    results = {}

    for t_val in eval_times:
        if t_val not in snapshots:
            print(f"[eval] Warning: t={t_val} not in FD snapshots; skipping.")
            continue

        u_fd_arr   = snapshots[t_val]                                  # (Ny, Nx) numpy
        u_pinn_arr = evaluate_pinn_on_fd_grid(model, x_fd, y_fd, t_val)  # (Ny, Nx) numpy
        pde_res    = evaluate_pde_residual_on_grid(model, x_fd, y_fd, t_val, alpha)

        err = {
            "t"            : t_val,
            "l2_rel"       : relative_l2(u_pinn_arr, u_fd_arr),
            "rmse"         : rmse(u_pinn_arr, u_fd_arr),
            "max_ae"       : max_abs_error(u_pinn_arr, u_fd_arr),
            "residual_rms" : pde_res["residual_rms"],
            "residual_max" : pde_res["residual_max"],
            "u_pinn"       : u_pinn_arr,
            "u_fd"         : u_fd_arr,
            "error"        : u_pinn_arr - u_fd_arr,
        }
        results[t_val] = err
        print(f"  t={t_val:.2f}: L2_rel={err['l2_rel']:.4e}  "
              f"RMSE={err['rmse']:.4e}  MaxAE={err['max_ae']:.4e}  "
              f"PDE_resid_rms={err['residual_rms']:.4e}")

    return results

# IC / BC specific checks


def evaluate_ic_error(model: torch.nn.Module,
                      x_fd: np.ndarray, y_fd: np.ndarray,
                      amplitude: float, cx: float, cy: float,
                      sigma: float) -> dict:
    """
    Evaluate PINN IC satisfaction: compare û(x,y,0) to the Gaussian IC.

    Returns dict with l2_rel, rmse, max_ae arrays.
    """
    X, Y = np.meshgrid(x_fd, y_fd)
    u0_exact = amplitude * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2))
    u_pinn_0 = evaluate_pinn_on_fd_grid(model, x_fd, y_fd, t_val=0.0)

    return {
        "l2_rel"  : relative_l2(u_pinn_0, u0_exact),
        "rmse"    : rmse(u_pinn_0, u0_exact),
        "max_ae"  : max_abs_error(u_pinn_0, u0_exact),
        "u_pinn_0": u_pinn_0,
        "u0_exact": u0_exact,
    }


def evaluate_bc_error(model: torch.nn.Module,
                      x_fd: np.ndarray, y_fd: np.ndarray,
                      eval_times: list[float]) -> dict:
    """
    Evaluate PINN BC satisfaction: max |û| on boundary over all eval times.

    Returns dict with 'max_bc_error' and per-wall errors.
    """
    model.eval()
    all_bc_vals = []

    for t_val in eval_times:
        # Evaluate on boundary nodes
        n_pts = len(x_fd)
        t_arr = np.full(n_pts, t_val, dtype=np.float32)

        # Left wall (x=0): û(0, y, t)
        u_left = _eval_boundary(model, np.zeros(n_pts, dtype=np.float32),
                                 y_fd.astype(np.float32), t_arr)
        # Right wall (x=1): û(1, y, t)
        u_right = _eval_boundary(model, np.ones(n_pts, dtype=np.float32),
                                  y_fd.astype(np.float32), t_arr)
        # Bottom wall (y=0): û(x, 0, t)
        u_bot = _eval_boundary(model, x_fd.astype(np.float32),
                                np.zeros(n_pts, dtype=np.float32), t_arr)
        # Top wall (y=1): û(x, 1, t)
        u_top = _eval_boundary(model, x_fd.astype(np.float32),
                                np.ones(n_pts, dtype=np.float32), t_arr)

        all_bc_vals.extend([u_left, u_right, u_bot, u_top])

    all_bc_vals = np.concatenate(all_bc_vals)
    return {
        "max_bc_error"  : float(np.max(np.abs(all_bc_vals))),
        "mean_bc_error" : float(np.mean(np.abs(all_bc_vals))),
        "rms_bc_error"  : float(np.sqrt(np.mean(all_bc_vals ** 2))),
    }


def _eval_boundary(model, x_arr, y_arr, t_arr) -> np.ndarray:
    """Helper: evaluate model on 1D boundary arrays."""
    x_t = torch.tensor(x_arr.reshape(-1, 1), dtype=torch.float32)
    y_t = torch.tensor(y_arr.reshape(-1, 1), dtype=torch.float32)
    t_t = torch.tensor(t_arr.reshape(-1, 1), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return model(x_t, y_t, t_t).numpy().flatten()

# Inference timing

def time_inference(model: torch.nn.Module,
                   n_query_points: int = 10_000,
                   n_repeats: int = 5,
                   t_final: float = 0.5) -> dict:
    """
    Measure PINN inference time for a given number of query points.

    Runs n_repeats evaluations and reports mean/std wall time.

    Parameters
    ----------
    model          : trained HeatPINN
    n_query_points : number of (x,y,t) points per inference call
    n_repeats      : number of timing repetitions

    Returns
    -------
    dict with 'mean_s', 'std_s', 'n_points'
    """
    x_q = torch.rand(n_query_points, 1, dtype=torch.float32)
    y_q = torch.rand(n_query_points, 1, dtype=torch.float32)
    t_q = torch.rand(n_query_points, 1, dtype=torch.float32) * t_final

    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats + 1):   # warm-up + repeats
            t0 = time.perf_counter()
            _ = model(x_q, y_q, t_q)
            times.append(time.perf_counter() - t0)

    times = times[1:]   # discard warm-up
    return {
        "n_points": n_query_points,
        "mean_s"  : float(np.mean(times)),
        "std_s"   : float(np.std(times)),
        "min_s"   : float(np.min(times)),
    }
