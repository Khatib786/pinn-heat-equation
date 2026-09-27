"""
sensitivity.py

Tier 2 - Sensitivity Analysis

Experiments:
    A. Collocation count sensitivity:
       Vary N_interior ∈ {2500, 5000, 10000} — retrain from scratch each time
    B. Architecture sensitivity:
       Compare [32×3], [64×4] (baseline), [64×5] — retrain each

For each configuration:
    - Train for TRAIN_EPOCHS epochs (same as baseline)
    - Evaluate on FD reference grid at all EVAL_TIMES
    - Record: final losses, L2_rel at each time, training wall time

Output:
    results/sensitivity_colloc.json
    results/sensitivity_arch.json
    figures/sensitivity_colloc.png
    figures/sensitivity_arch.png

Usage:
    python experiments/sensitivity.py --exp colloc   # collocation sweep only
    python experiments/sensitivity.py --exp arch     # architecture sweep only
    python experiments/sensitivity.py --exp all      # both (default)

WARNING: Each run takes as long as baseline training. Run only after Tier 1 is stable.
"""

import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (
    ALPHA, T_FINAL, IC_AMPLITUDE, IC_CENTER_X, IC_CENTER_Y, IC_SIGMA,
    FD_NX, FD_NY, FD_DT, FD_SAFETY,
    PINN_HIDDEN_DIMS, PINN_ACTIVATION, TRAIN_SEED,
    TRAIN_LR_INIT, TRAIN_LR_FINAL, TRAIN_EPOCHS,
    TRAIN_CHECKPOINT_INTERVAL, TRAIN_PRINT_INTERVAL, TRAIN_EARLY_STOP_LOSS,
    LAMBDA_PDE, LAMBDA_IC, LAMBDA_BC,
    N_COLLOC_INTERIOR, N_COLLOC_IC, N_COLLOC_BC,
    SENSITIVITY_COLLOC_COUNTS, SENSITIVITY_ARCHITECTURES,
    EVAL_TIMES,
    RESULTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR,
)
from fd_solver import FDSolver, make_gaussian_ic
from pinn_model import HeatPINN
from losses import (
    compute_total_loss,
    sample_interior_points, sample_ic_points, sample_bc_points,
    gaussian_ic_torch,
)
from training import Trainer
from evaluation import evaluate_at_times
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Sensitivity analysis")
    parser.add_argument("--exp", choices=["colloc", "arch", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs (e.g. use fewer for a quick test)")
    return parser.parse_args()


# Helper: train one configuration and evaluate

def train_and_evaluate(hidden_dims: list, n_interior: int,
                        fd_result: dict, n_epochs: int,
                        run_tag: str, seed: int = TRAIN_SEED) -> dict:
    """
    Train a PINN with given architecture and collocation count.
    Evaluate on FD reference. Return summary dict.
    """
    print(f"\n{'─'*55}")
    print(f"Config: arch={hidden_dims}  n_interior={n_interior}  tag={run_tag}")
    print(f"{'─'*55}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HeatPINN(hidden_dims=hidden_dims, t_final=T_FINAL,
                     activation=PINN_ACTIVATION)

    x_r, y_r, t_r = sample_interior_points(n_interior, T_FINAL, seed=seed)
    x_ic, y_ic, t_ic = sample_ic_points(N_COLLOC_IC, seed=seed)
    x_bc, y_bc, t_bc = sample_bc_points(N_COLLOC_BC // 4, T_FINAL, seed=seed)
    u0_target = gaussian_ic_torch(x_ic, y_ic,
                                   IC_AMPLITUDE, IC_CENTER_X, IC_CENTER_Y, IC_SIGMA)

    def loss_fn(m):
        return compute_total_loss(
            m, x_r, y_r, t_r, x_ic, y_ic, t_ic, u0_target,
            x_bc, y_bc, t_bc,
            alpha=ALPHA,
            lambda_pde=LAMBDA_PDE, lambda_ic=LAMBDA_IC, lambda_bc=LAMBDA_BC,
        )

    ckpt_dir = CHECKPOINTS_DIR / f"sensitivity_{run_tag}"
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        lr_init=TRAIN_LR_INIT, lr_final=TRAIN_LR_FINAL,
        n_epochs=n_epochs,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_interval=max(1000, n_epochs // 10),
        print_interval=max(500, n_epochs // 20),
        early_stop_loss=TRAIN_EARLY_STOP_LOSS,
        seed=seed,
    )
    history = trainer.train()
    train_time = trainer.train_wall_time

    # Evaluate
    # Evaluate at whichever configured EVAL_TIMES are present in this FD run's
    # snapshots (previously a hardcoded [0.1,0.2,0.3,0.5] list independent of
    # config.EVAL_TIMES, which risked silently diverging from the Tier 1
    # baseline's own evaluation times).
    eval_times = [t for t in EVAL_TIMES if t in fd_result["snapshots"]]
    eval_results = evaluate_at_times(model, fd_result, eval_times, ALPHA)

    # Compile summary
    summary = {
        "tag"          : run_tag,
        "hidden_dims"  : list(hidden_dims),
        "n_interior"   : n_interior,
        "n_params"     : sum(p.numel() for p in model.parameters()),
        "n_epochs"     : len(history["epoch"]),
        "train_time_s" : train_time,
        "final_total"  : history["total"][-1],
        "final_pde"    : history["pde"][-1],
        "final_ic"     : history["ic"][-1],
        "final_bc"     : history["bc"][-1],
        "eval": {
            str(t): {
                "l2_rel": eval_results[t]["l2_rel"],
                "rmse"  : eval_results[t]["rmse"],
                "max_ae": eval_results[t]["max_ae"],
            }
            for t in eval_results
        }
    }
    return summary


# Experiment A: Collocation sensitivity

def run_colloc_sensitivity(fd_result: dict, n_epochs: int) -> list[dict]:
    print("\n" + "=" * 60)
    print("SENSITIVITY A: Collocation Count")
    print(f"  Counts to test: {SENSITIVITY_COLLOC_COUNTS}")
    print(f"  Epochs per run: {n_epochs}")
    print("=" * 60)

    results = []
    for n_int in SENSITIVITY_COLLOC_COUNTS:
        tag = f"colloc_{n_int}"
        r = train_and_evaluate(
            hidden_dims=PINN_HIDDEN_DIMS,   # baseline architecture — from config, not hardcoded
            n_interior=n_int,
            fd_result=fd_result,
            n_epochs=n_epochs,
            run_tag=tag,
        )
        results.append(r)

    # Save
    path = RESULTS_DIR / "sensitivity_colloc.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Sensitivity] Collocation results saved to {path}")

    # Plot: L2_rel at t=0.5 vs. collocation count
    _plot_sensitivity_bar(
        results,
        x_key=lambda r: r["n_interior"],
        x_label="N_interior",
        title="Collocation Count Sensitivity (t=0.5)",
        filename="sensitivity_colloc.png",
    )
    return results


# Experiment B: Architecture sensitivity

def run_arch_sensitivity(fd_result: dict, n_epochs: int) -> list[dict]:
    print("\n" + "=" * 60)
    print("SENSITIVITY B: Architecture")
    print(f"  Architectures: {SENSITIVITY_ARCHITECTURES}")
    print(f"  Epochs per run: {n_epochs}")
    print("=" * 60)

    results = []
    for arch in SENSITIVITY_ARCHITECTURES:
        tag = f"arch_{'x'.join(map(str, arch))}_n{len(arch)}"
        r = train_and_evaluate(
            hidden_dims=arch,
            n_interior=N_COLLOC_INTERIOR,   # baseline collocation count
            fd_result=fd_result,
            n_epochs=n_epochs,
            run_tag=tag,
        )
        results.append(r)

    # Save
    path = RESULTS_DIR / "sensitivity_arch.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Sensitivity] Architecture results saved to {path}")

    _plot_sensitivity_bar(
        results,
        x_key=lambda r: str(r["hidden_dims"]),
        x_label="Architecture",
        title="Architecture Sensitivity (t=0.5)",
        filename="sensitivity_arch.png",
    )
    return results


# Bar chart helper

def _plot_sensitivity_bar(results: list[dict],
                           x_key, x_label: str, title: str,
                           filename: str, t_eval: float = T_FINAL) -> None:
    x_vals  = [x_key(r) for r in results]
    l2_vals = [r["eval"].get(str(t_eval), {}).get("l2_rel", float("nan"))
               for r in results]
    t_vals  = [r["train_time_s"] / 60.0 for r in results]   # minutes

    x_str = [str(v) for v in x_vals]
    x_pos = np.arange(len(x_str))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    bars = ax1.bar(x_pos, l2_vals, color="steelblue", edgecolor="navy", alpha=0.8)
    ax1.bar_label(bars, fmt="%.2e", padding=3, fontsize=9)
    ax1.set_xticks(x_pos); ax1.set_xticklabels(x_str, rotation=15, ha="right")
    ax1.set_xlabel(x_label); ax1.set_ylabel("Relative L2 Error (t=0.5)")
    ax1.set_title(f"{title}\nAccuracy")
    ax1.grid(True, axis="y", alpha=0.4)

    bars2 = ax2.bar(x_pos, t_vals, color="salmon", edgecolor="darkred", alpha=0.8)
    ax2.bar_label(bars2, fmt="%.1f min", padding=3, fontsize=9)
    ax2.set_xticks(x_pos); ax2.set_xticklabels(x_str, rotation=15, ha="right")
    ax2.set_xlabel(x_label); ax2.set_ylabel("Training Time (min)")
    ax2.set_title(f"{title}\nTraining Cost")
    ax2.grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Sensitivity] Saved {path}")


# Main

if __name__ == "__main__":
    args = parse_args()
    n_epochs = args.epochs or TRAIN_EPOCHS

    print("\n" + "=" * 60)
    print("TIER 2 SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"WARNING: Each configuration trains for {n_epochs} epochs.")
    print(f"Ensure baseline.py has run first (need FD reference + results/).\n")

    # Load FD reference
    fd_npz = RESULTS_DIR / "fd_reference.npz"
    if not fd_npz.exists():
        print("ERROR: results/fd_reference.npz not found. Run baseline.py first.")
        sys.exit(1)

    data = np.load(fd_npz)
    x_fd = data["x"]
    y_fd = data["y"]
    # Reconstruct snapshots dict
    snapshots = {}
    for key in data.files:
        if key.startswith("u_t"):
            t_str = key[3:].replace("p", ".")
            snapshots[float(t_str)] = data[key]
    fd_result = {"x": x_fd, "y": y_fd, "snapshots": snapshots,
                 "solve_time": float(data.get("solve_time", np.array([0.0]))[0])}

    if args.exp in ("colloc", "all"):
        run_colloc_sensitivity(fd_result, n_epochs)

    if args.exp in ("arch", "all"):
        run_arch_sensitivity(fd_result, n_epochs)

    print("\nSensitivity analysis complete.")
