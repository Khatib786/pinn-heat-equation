"""
fd_solver.py

"""

import time
import numpy as np
from pathlib import Path


def make_gaussian_ic(x: np.ndarray, y: np.ndarray,
                     amplitude: float, cx: float, cy: float,
                     sigma: float) -> np.ndarray:
    """
    Evaluate the Gaussian hot-spot initial condition on a 2D grid.

    Parameters
    ----------
    x : 1D array of x-coordinates (length Nx)
    y : 1D array of y-coordinates (length Ny)
    amplitude, cx, cy, sigma : IC parameters (see config.py)

    Returns
    -------
    u0 : 2D array, shape (Ny, Nx)
         u0[j, i] = amplitude * exp(-((x[i]-cx)² + (y[j]-cy)²) / (2σ²))
    """
    X, Y = np.meshgrid(x, y)  # (Ny, Nx)
    return amplitude * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2))


def make_fourier_ic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Single-mode Fourier initial condition for analytical validation:
        u(x,y,0) = sin(πx) sin(πy)

    Exact solution: u(x,y,t) = exp(-2π²αt) sin(πx) sin(πy)
    """
    X, Y = np.meshgrid(x, y)
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def fourier_analytical(x: np.ndarray, y: np.ndarray,
                        t: float, alpha: float) -> np.ndarray:
    """Exact solution for the Fourier-mode IC at time t."""
    X, Y = np.meshgrid(x, y)
    decay = np.exp(-2.0 * np.pi ** 2 * alpha * t)
    return decay * np.sin(np.pi * X) * np.sin(np.pi * Y)


class FDSolver:
    """
    Explicit FTCS solver for ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²).

    Parameters
    ----------
    nx, ny   : number of grid points in x, y (including boundary nodes)
    alpha    : thermal diffusivity
    t_final  : simulation end time
    dt       : time step (must satisfy stability condition)
    safety   : safety factor applied to dt (informational; dt must already satisfy stability)
    """

    def __init__(self, nx: int, ny: int, alpha: float,
                 t_final: float, dt: float, safety: float = 0.9):
        self.nx      = nx
        self.ny      = ny
        self.alpha   = alpha
        self.t_final = t_final
        self.dt      = dt
        self.safety  = safety

        self.x = np.linspace(0.0, 1.0, nx)
        self.y = np.linspace(0.0, 1.0, ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Stability parameters
        self.rx = alpha * dt / self.dx ** 2   # should be ≤ 0.25 (2D)
        self.ry = alpha * dt / self.dy ** 2

        self._check_stability()

    def _check_stability(self):
        """Verify von Neumann stability condition and raise if violated."""
        r_total = self.rx + self.ry   # 2D: must be ≤ 0.5 (equivalent to each ≤ 0.25 for uniform grid)
        limit   = 0.5
        print(f"[FD] Stability check: r_x = {self.rx:.5f}, r_y = {self.ry:.5f}, "
              f"r_x + r_y = {r_total:.5f} (limit = {limit})")
        if r_total > limit:
            raise ValueError(
                f"FD stability condition violated: r_x + r_y = {r_total:.4f} > {limit}. "
                f"Reduce dt or increase grid spacing."
            )
        print(f"[FD] Stability condition satisfied (safety factor applied).")

    def solve(self, u0: np.ndarray,
              save_times: list[float] | None = None,
              verbose: bool = True) -> dict:
        """
        Advance the solution from t=0 to t_final using explicit FTCS.

        Parameters
        ----------
        u0         : Initial condition, shape (ny, nx)
        save_times : List of times at which to store snapshots (including 0.0)
        verbose    : Print progress

        Returns
        -------
        result : dict with keys:
            'snapshots'   : {t: u_array} for each saved time
            'energy'      : list of (t, total_energy) tuples
            'solve_time'  : wall-clock solve time in seconds
            'n_steps'     : total number of time steps taken
        """
        if u0.shape != (self.ny, self.nx):
            raise ValueError(f"u0 shape {u0.shape} != expected ({self.ny}, {self.nx})")

        if save_times is None:
            save_times = [0.0, self.t_final]
        save_times_sorted = sorted(set(save_times))

        u = u0.copy().astype(np.float64)
        snapshots = {}
        energy    = []

        # Save t=0
        if 0.0 in save_times_sorted:
            snapshots[0.0] = u.copy()
        energy.append((0.0, self._compute_energy(u)))

        t = 0.0
        n_steps = 0
        save_idx = 0
        # Move pointer past t=0 if already saved
        while save_idx < len(save_times_sorted) and save_times_sorted[save_idx] <= 1e-14:
            save_idx += 1

        t_start = time.perf_counter()

        while t < self.t_final - 1e-12:
            # Adjust last step to land exactly on t_final
            dt = min(self.dt, self.t_final - t)

            # FTCS update (vectorized):
            # u_new[j,i] = u[j,i] + rx*(u[j,i+1] - 2u[j,i] + u[j,i-1])
            #                       + ry*(u[j+1,i] - 2u[j,i] + u[j-1,i])
            # Interior nodes only (boundary stays 0)
            u_new = u.copy()
            rx = self.alpha * dt / self.dx ** 2
            ry = self.alpha * dt / self.dy ** 2

            u_new[1:-1, 1:-1] = (
                u[1:-1, 1:-1]
                + rx * (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2])
                + ry * (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1])
            )

            # Enforce Dirichlet BC (should already be 0, but numerical safety)
            u_new[0, :]  = 0.0
            u_new[-1, :] = 0.0
            u_new[:, 0]  = 0.0
            u_new[:, -1] = 0.0

            u = u_new
            t += dt
            n_steps += 1

            # Record energy every 10 steps (cheap)
            if n_steps % 10 == 0:
                energy.append((t, self._compute_energy(u)))

            # Save snapshot if we passed a requested time
            while (save_idx < len(save_times_sorted) and
                   t >= save_times_sorted[save_idx] - 1e-10):
                t_snap = save_times_sorted[save_idx]
                snapshots[t_snap] = u.copy()
                save_idx += 1

        solve_time = time.perf_counter() - t_start

        if verbose:
            print(f"[FD] Solve complete: {n_steps} steps, "
                  f"final t = {t:.6f}, wall time = {solve_time:.4f} s")

        return {
            "snapshots" : snapshots,
            "energy"    : energy,
            "solve_time": solve_time,
            "n_steps"   : n_steps,
            "x"         : self.x,
            "y"         : self.y,
        }

    def _compute_energy(self, u: np.ndarray) -> float:
        """
        Approximate ∫∫ u dA using the trapezoidal rule, correctly weighted
        by the actual grid spacing (dx, dy).

        NOTE: np.trapz() defaults to unit spacing when no `x`/`dx` argument
        is given. The previous version omitted dx/dy entirely, which
        silently computed a raw grid-point sum (∝ 1/(dx*dy) times too
        large) rather than the physical integral. For this grid
        (dx=dy≈1/63), that inflated the reported "energy" by a factor of
        ≈3970 — exactly matching the discrepancy seen in
        fd_energy_decay.png (plotted value ≈249 vs. true continuum
        integral of the Gaussian IC, 2*pi*A*sigma^2 ≈ 0.063).
        For Dirichlet BC with positive IC, this should decay monotonically.
        """
        # NumPy 2.0 removed np.trapz in favor of np.trapezoid; this was a
        # latent bug in the original code (present even before the dx/dy fix
        # above) that would crash on any environment running NumPy >= 2.0.
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(_trapz(_trapz(u, dx=self.dx, axis=1), dx=self.dy, axis=0))


# Standalone validation helper


def validate_against_analytical(nx: int, alpha: float, t_final: float,
                                 dt_safety: float = 0.9) -> dict:
    """
    Validate FD solver against the exact Fourier-mode solution:
        u(x,y,t) = exp(-2π²αt) sin(πx) sin(πy)

    Returns dict with L2 error and RMSE at t_final.
    """
    dx = 1.0 / (nx - 1)
    dt = dt_safety * 0.25 * dx ** 2 / alpha
    solver = FDSolver(nx=nx, ny=nx, alpha=alpha, t_final=t_final, dt=dt)

    x = solver.x
    y = solver.y
    u0 = make_fourier_ic(x, y)

    result = solver.solve(u0, save_times=[0.0, t_final], verbose=False)
    u_fd   = result["snapshots"][t_final]
    u_exact = fourier_analytical(x, y, t_final, alpha)

    diff   = u_fd - u_exact
    l2_rel = float(np.linalg.norm(diff) / (np.linalg.norm(u_exact) + 1e-14))
    rmse   = float(np.sqrt(np.mean(diff ** 2)))
    max_ae = float(np.max(np.abs(diff)))

    return {
        "nx"    : nx,
        "l2_rel": l2_rel,
        "rmse"  : rmse,
        "max_ae": max_ae,
    }


def convergence_test(grid_sizes: list[int], alpha: float,
                     t_final: float, dt_safety: float = 0.9) -> list[dict]:
    """
    Run FD validation at multiple grid resolutions.
    Returns list of result dicts from validate_against_analytical.
    """
    results = []
    for nx in grid_sizes:
        r = validate_against_analytical(nx, alpha, t_final, dt_safety)
        print(f"  nx={nx:3d}: L2_rel={r['l2_rel']:.2e}  RMSE={r['rmse']:.2e}  "
              f"MaxAE={r['max_ae']:.2e}")
        results.append(r)
    return results
