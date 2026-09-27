"""
visualization.py

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from typing import Optional


# Helpers

def _save(fig: plt.Figure, filename: str, figures_dir: Path,
          dpi: int = 150) -> None:
    """Save figure to FIGURES_DIR and close it."""
    path = figures_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[vis] Saved {path}")


# Single-field heat map

def plot_field(u: np.ndarray, x: np.ndarray, y: np.ndarray,
               title: str = "", figures_dir: Path = Path("figures"),
               filename: str = "field.png",
               vmin: Optional[float] = None,
               vmax: Optional[float] = None,
               cmap: str = "hot") -> plt.Figure:
    """
    Plot a single 2D temperature field as a filled contour / imshow.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 51)
    else:
        levels = 50
    im = ax.contourf(x, y, u, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label="u")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    _save(fig, filename, figures_dir)
    return fig


# Comparison rows: FD | PINN | Error

def plot_comparison_row(eval_results: dict,
                         eval_times: list,
                         x_fd: np.ndarray,
                         y_fd: np.ndarray,
                         figures_dir: Path = Path("figures"),
                         filename: str = "comparison_row.png") -> plt.Figure:
    """
    Plot FD field, PINN field, and absolute error side-by-side for each time.

    Layout: rows = time snapshots, cols = [FD | PINN | |error|]
    """
    n_times = len(eval_times)
    fig = plt.figure(figsize=(14, 3.5 * n_times))
    gs  = gridspec.GridSpec(n_times, 3, figure=fig, hspace=0.4, wspace=0.35)

    for row_idx, t_val in enumerate(eval_times):
        if t_val not in eval_results:
            continue
        res   = eval_results[t_val]
        u_fd  = res["u_fd"]
        u_pinn= res["u_pinn"]
        err   = np.abs(res["error"])

        # Shared scale for FD and PINN columns
        vmax_shared = max(u_fd.max(), u_pinn.max(), 1e-10)
        vmin_shared = min(u_fd.min(), u_pinn.min(), 0.0)
        # Explicit levels anchored to the shared scale — see plot_field()
        # for why a bare integer `levels` count silently ignores vmin/vmax.
        shared_levels = np.linspace(vmin_shared, vmax_shared, 51)

        for col_idx, (data, label) in enumerate([
            (u_fd,   f"FD  t={t_val:.2f}"),
            (u_pinn, f"PINN t={t_val:.2f}"),
            (err,    f"|Error| t={t_val:.2f}"),
        ]):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if col_idx < 2:
                im = ax.contourf(x_fd, y_fd, data, levels=shared_levels,
                                 cmap="hot", vmin=vmin_shared, vmax=vmax_shared)
            else:
                # Error: use a sequential colormap from 0
                im = ax.contourf(x_fd, y_fd, data, levels=50, cmap="Reds")
            fig.colorbar(im, ax=ax, shrink=0.85)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8)
            ax.set_aspect("equal")
            # Annotate with error metrics (error column only)
            if col_idx == 2:
                l2r = eval_results[t_val]["l2_rel"]
                ax.text(0.05, 0.05, f"L2_rel={l2r:.2e}",
                        transform=ax.transAxes, fontsize=7,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    fig.suptitle("FD Reference vs. PINN Prediction", fontsize=12, y=1.01)
    _save(fig, filename, figures_dir)
    return fig


# Error map at a single time

def plot_error_map(error: np.ndarray, x_fd: np.ndarray, y_fd: np.ndarray,
                   t_val: float, l2_rel: float,
                   figures_dir: Path = Path("figures"),
                   filename: str = "error_map.png") -> plt.Figure:
    """Signed error map (PINN - FD) with diverging colormap."""
    abs_max = max(np.abs(error).max(), 1e-10)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.contourf(x_fd, y_fd, error, levels=100, cmap="RdBu_r", norm=norm)
    fig.colorbar(im, ax=ax, label="PINN − FD")
    ax.set_title(f"Signed Error  t={t_val:.2f}  (L2_rel={l2_rel:.2e})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    _save(fig, filename, figures_dir)
    return fig


# Cross-section at y = 0.5

def plot_cross_section(eval_results: dict,
                        eval_times: list,
                        x_fd: np.ndarray,
                        y_fd: np.ndarray,
                        y_slice: float = 0.5,
                        figures_dir: Path = Path("figures"),
                        filename: str = "cross_section.png") -> plt.Figure:
    """
    Plot 1D cross-section of FD and PINN at y ≈ y_slice for each eval time.
    """
    # Find nearest y index
    j_idx = int(np.argmin(np.abs(y_fd - y_slice)))
    y_actual = y_fd[j_idx]

    n_times = len(eval_times)
    cols = min(n_times, 4)
    rows = (n_times + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                              squeeze=False)
    axes_flat = axes.flatten()

    for idx, t_val in enumerate(eval_times):
        ax = axes_flat[idx]
        if t_val not in eval_results:
            ax.set_visible(False)
            continue
        res = eval_results[t_val]
        ax.plot(x_fd, res["u_fd"][j_idx, :],    "b-",  lw=2, label="FD")
        ax.plot(x_fd, res["u_pinn"][j_idx, :],  "r--", lw=2, label="PINN")
        ax.set_title(f"t = {t_val:.2f}  (y ≈ {y_actual:.3f})", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("u"); ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    # Hide unused panels
    for idx in range(len(eval_times), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Cross-sections at y ≈ {y_actual:.3f}", fontsize=11)
    fig.tight_layout()
    _save(fig, filename, figures_dir)
    return fig


# Training loss curve

def plot_loss_curve(history: dict,
                    figures_dir: Path = Path("figures"),
                    filename: str = "loss_curve.png") -> plt.Figure:
    """
    Plot training loss curves: total, PDE, IC, BC on a log scale.
    """
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: all components
    ax = axes[0]
    ax.semilogy(epochs, history["total"], "k-",  lw=1.5, label="Total")
    ax.semilogy(epochs, history["pde"],   "b--", lw=1.2, label="PDE")
    ax.semilogy(epochs, history["ic"],    "g-.", lw=1.2, label="IC")
    ax.semilogy(epochs, history["bc"],    "r:",  lw=1.2, label="BC")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss Components")
    ax.legend(); ax.grid(True, alpha=0.4)

    # Right: LR schedule
    ax2 = axes[1]
    ax2.semilogy(epochs, history["lr"], "m-", lw=1.5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate (Cosine Decay)")
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    _save(fig, filename, figures_dir)
    return fig


# Amortized cost analysis

def plot_amortization(t_fd: float, t_train: float, t_infer_per_eval: float,
                       n_max: int = 200,
                       figures_dir: Path = Path("figures"),
                       filename: str = "amortization.png") -> plt.Figure:
    """
    Plot amortized cost: C_FD(N) = N*t_fd vs. C_PINN(N) = t_train + N*t_infer.

    Parameters
    ----------
    t_fd             : wall time for one full FD solve (seconds)
    t_train          : PINN training wall time (seconds)
    t_infer_per_eval : PINN inference time per full-grid evaluation (seconds)
    n_max            : maximum N to plot
    """
    N = np.arange(0, n_max + 1)
    c_fd   = N * t_fd
    c_pinn = t_train + N * t_infer_per_eval

    # Break-even
    denom = t_fd - t_infer_per_eval
    if denom > 0:
        n_star = t_train / denom
        n_star_label = f"N* ≈ {n_star:.1f}"
    else:
        n_star = None
        n_star_label = "No crossover (PINN inference ≥ FD)"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N, c_fd,   "b-",  lw=2, label=f"FD:   N × {t_fd:.3f} s")
    ax.plot(N, c_pinn, "r-",  lw=2,
            label=f"PINN: {t_train:.1f} s + N × {t_infer_per_eval:.4f} s")

    if n_star is not None and 0 < n_star < n_max:
        ax.axvline(n_star, color="gray", ls="--", lw=1.5, label=n_star_label)
        ax.axhline(t_train + n_star * t_infer_per_eval,
                   color="gray", ls=":", lw=1)

    ax.set_xlabel("Number of evaluations N")
    ax.set_ylabel("Total wall-clock time (s)")
    ax.set_title(f"Amortized Computational Cost\n{n_star_label}")
    ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout()
    _save(fig, filename, figures_dir)
    return fig


# FD energy decay plot

def plot_energy_decay(energy_log: list[tuple],
                       figures_dir: Path = Path("figures"),
                       filename: str = "fd_energy_decay.png") -> plt.Figure:
    """
    Plot total 'energy' ∫∫u dA vs. time (should decrease monotonically for Dirichlet BC).
    """
    times = [e[0] for e in energy_log]
    energies = [e[1] for e in energy_log]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, energies, "b-", lw=2)
    ax.set_xlabel("t"); ax.set_ylabel("∫∫ u dA")
    ax.set_title("FD Energy Decay (Dirichlet BC drains heat)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    _save(fig, filename, figures_dir)
    return fig


# Convergence test plot

def plot_convergence(conv_results: list[dict],
                      figures_dir: Path = Path("figures"),
                      filename: str = "fd_convergence.png") -> plt.Figure:
    """
    Log-log convergence plot for FD solver (L2_rel vs. Δx).
    """
    nx_vals = [r["nx"]     for r in conv_results]
    l2_vals = [r["l2_rel"] for r in conv_results]
    dx_vals = [1.0 / (nx - 1) for nx in nx_vals]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(dx_vals, l2_vals, "bo-", lw=2, ms=7, label="FD L2_rel error")
    # Reference slope: O(Δx²)
    dx_arr = np.array(dx_vals)
    ref = l2_vals[0] * (dx_arr / dx_arr[0]) ** 2
    ax.loglog(dx_arr, ref, "k--", lw=1.5, label="O(Δx²) reference")
    ax.set_xlabel("Grid spacing Δx"); ax.set_ylabel("Relative L2 Error")
    ax.set_title("FD Solver Convergence (vs. Analytical Fourier Solution)")
    ax.legend(); ax.grid(True, alpha=0.4, which="both")
    fig.tight_layout()
    _save(fig, filename, figures_dir)
    return fig


# Summary error bar chart across time

def plot_error_vs_time(eval_results: dict,
                        figures_dir: Path = Path("figures"),
                        filename: str = "error_vs_time.png") -> plt.Figure:
    """
    Bar chart of L2_rel error at each evaluation time.
    """
    times  = sorted(eval_results.keys())
    l2_vals= [eval_results[t]["l2_rel"] for t in times]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([f"t={t:.2f}" for t in times], l2_vals,
                  color="steelblue", edgecolor="navy", alpha=0.8)
    ax.bar_label(bars, fmt="%.2e", padding=3, fontsize=9)
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("PINN vs. FD: Relative L2 Error at Evaluation Times")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    _save(fig, filename, figures_dir)
    return fig
