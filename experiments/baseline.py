
# Master baseline experiment script.
# Steps:
# 1. Validate the FD solver using an analytical Fourier-mode solution and convergence test.
# 2. Generate the FD reference solution from the Gaussian initial condition and save snapshots.
# 3. Train the PINN.
# 4. Compare PINN results with the FD reference solution.
# 5. Generate the required figures.
# Usage:
# python experiments/baseline.py              # Full run
# python experiments/baseline.py --skip-train # Use existing checkpoint
# python experiments/baseline.py --fd-only    # FD validation and reference only
# Results:     results/baseline_results.json
# Figures:     figures/
# Checkpoint:  checkpoints/pinn_checkpoint_final.pt

import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (
    ALPHA, T_FINAL, IC_AMPLITUDE, IC_CENTER_X, IC_CENTER_Y, IC_SIGMA,
    FD_NX, FD_NY, FD_DT, FD_DT_MAX, FD_SAFETY,
    PINN_HIDDEN_DIMS, PINN_ACTIVATION,
    N_COLLOC_INTERIOR, N_COLLOC_IC, N_COLLOC_BC,
    LAMBDA_PDE, LAMBDA_IC, LAMBDA_BC,
    TRAIN_LR_INIT, TRAIN_LR_FINAL, TRAIN_EPOCHS, TRAIN_SEED,
    TRAIN_CHECKPOINT_INTERVAL, TRAIN_PRINT_INTERVAL, TRAIN_EARLY_STOP_LOSS,
    EVAL_TIMES, FD_VALIDATION_GRIDS,
    RESULTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR,
)
from fd_solver import (
    FDSolver, make_gaussian_ic, make_fourier_ic,
    convergence_test, validate_against_analytical,
)
from pinn_model import HeatPINN
from losses import (
    compute_total_loss,
    sample_interior_points, sample_ic_points, sample_bc_points,
    gaussian_ic_torch,
)
from training import Trainer
from evaluation import (
    evaluate_at_times, evaluate_ic_error, evaluate_bc_error,
    time_inference,
)
from visualization import (
    plot_field, plot_comparison_row, plot_cross_section,
    plot_loss_curve, plot_energy_decay, plot_convergence,
    plot_error_vs_time, plot_error_map,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline PINN vs. FD experiment")
    parser.add_argument("--skip-train", action="store_true",
                        help="Load existing checkpoint; skip PINN training")
    parser.add_argument("--fd-only", action="store_true",
                        help="Run FD validation and reference only; stop before PINN")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific checkpoint to load (overrides default)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs from config")
    return parser.parse_args()


# Phase FD Validation

def run_fd_validation():
    print("\n" + "=" * 60)
    print("PHASE 1+2: FD SOLVER VALIDATION")
    print("=" * 60)

    # Convergence test against analytical Fourier-mode solution
    print("\n[FD Validation] Convergence test (Fourier-mode IC):")
    conv_results = convergence_test(FD_VALIDATION_GRIDS, ALPHA, T_FINAL, FD_SAFETY)

    # Plot convergence
    plot_convergence(conv_results, figures_dir=FIGURES_DIR,
                     filename="fd_convergence.png")

    # Report stability parameter for baseline grid
    dx = 1.0 / (FD_NX - 1)
    r  = ALPHA * FD_DT / dx ** 2
    print(f"\n[FD Validation] Baseline grid: Nx={FD_NX}, Δx={dx:.5f}, "
          f"Δt={FD_DT:.5e}, r={r:.4f} (limit=0.25)")

    # Analytical check for baseline grid
    print("\n[FD Validation] Baseline grid analytical error:")
    baseline_valid = validate_against_analytical(FD_NX, ALPHA, T_FINAL, FD_SAFETY)
    print(f"  nx={FD_NX}: L2_rel={baseline_valid['l2_rel']:.2e}  "
          f"RMSE={baseline_valid['rmse']:.2e}  MaxAE={baseline_valid['max_ae']:.2e}")

    return conv_results, baseline_valid


# FD Reference Solution

def run_fd_reference():
    print("\n" + "=" * 60)
    print("PHASE 3: FD REFERENCE SOLUTION (Gaussian IC)")
    print("=" * 60)

    solver = FDSolver(nx=FD_NX, ny=FD_NY, alpha=ALPHA,
                      t_final=T_FINAL, dt=FD_DT, safety=FD_SAFETY)
    x, y = solver.x, solver.y
    u0 = make_gaussian_ic(x, y, IC_AMPLITUDE, IC_CENTER_X, IC_CENTER_Y, IC_SIGMA)

    # Plot IC
    plot_field(u0, x, y,
               title=f"Initial Condition: Gaussian (A={IC_AMPLITUDE}, σ={IC_SIGMA})",
               figures_dir=FIGURES_DIR, filename="fd_ic.png", cmap="hot")

    print(f"\n[FD Reference] Solving with Nx={FD_NX}, Δt={FD_DT:.2e}, "
          f"save_times={EVAL_TIMES} ...")
    fd_result = solver.solve(u0, save_times=EVAL_TIMES, verbose=True)
    print(f"[FD Reference] Solve time: {fd_result['solve_time']:.4f} s")

    # Plot energy decay
    plot_energy_decay(fd_result["energy"], figures_dir=FIGURES_DIR,
                      filename="fd_energy_decay.png")

    # Plot FD snapshots
    for t_val in EVAL_TIMES:
        if t_val in fd_result["snapshots"]:
            u_snap = fd_result["snapshots"][t_val]
            vmax = fd_result["snapshots"][0.0].max()
            plot_field(u_snap, x, y,
                       title=f"FD Reference: t={t_val:.2f}",
                       figures_dir=FIGURES_DIR,
                       filename=f"fd_snapshot_t{t_val:.2f}.png",
                       vmin=0, vmax=vmax, cmap="hot")

    # Save FD result to disk (compressed)
    np.savez_compressed(
        str(RESULTS_DIR / "fd_reference.npz"),
        x=x, y=y,
        **{f"u_t{str(t_val).replace('.','p')}": fd_result["snapshots"][t_val]
           for t_val in EVAL_TIMES if t_val in fd_result["snapshots"]},
        solve_time=np.array([fd_result["solve_time"]]),
    )
    print(f"[FD Reference] Snapshots saved to results/fd_reference.npz")

    return fd_result


#  PINN Setup + Training

def run_pinn_training(n_epochs: int = None):
    print("\n" + "=" * 60)
    print("PHASE 4-7: PINN IMPLEMENTATION + TRAINING")
    print("=" * 60)

    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)

    n_epochs = n_epochs or TRAIN_EPOCHS

    # Build model
    model = HeatPINN(hidden_dims=PINN_HIDDEN_DIMS,
                     t_final=T_FINAL,
                     activation=PINN_ACTIVATION)

    # Sample collocation points (fixed - sampled once, not per epoch)
    print(f"\n[PINN] Sampling collocation points...")
    x_r, y_r, t_r = sample_interior_points(N_COLLOC_INTERIOR, T_FINAL, seed=TRAIN_SEED)
    x_ic, y_ic, t_ic = sample_ic_points(N_COLLOC_IC, seed=TRAIN_SEED)
    x_bc, y_bc, t_bc = sample_bc_points(N_COLLOC_BC // 4, T_FINAL, seed=TRAIN_SEED)

    # Compute IC target values (Gaussian)
    u0_target = gaussian_ic_torch(x_ic, y_ic,
                                   IC_AMPLITUDE, IC_CENTER_X, IC_CENTER_Y, IC_SIGMA)

    print(f"  Interior PDE points : {x_r.shape[0]}")
    print(f"  IC points           : {x_ic.shape[0]}")
    print(f"  BC points           : {x_bc.shape[0]} (4 × {N_COLLOC_BC // 4})")

    # Loss closure - captures all collocation tensors
    def loss_fn(m):
        return compute_total_loss(
            m,
            x_r, y_r, t_r,
            x_ic, y_ic, t_ic, u0_target,
            x_bc, y_bc, t_bc,
            alpha=ALPHA,
            lambda_pde=LAMBDA_PDE,
            lambda_ic=LAMBDA_IC,
            lambda_bc=LAMBDA_BC,
        )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        lr_init=TRAIN_LR_INIT,
        lr_final=TRAIN_LR_FINAL,
        n_epochs=n_epochs,
        checkpoint_dir=str(CHECKPOINTS_DIR),
        checkpoint_interval=TRAIN_CHECKPOINT_INTERVAL,
        print_interval=TRAIN_PRINT_INTERVAL,
        early_stop_loss=TRAIN_EARLY_STOP_LOSS,
        seed=TRAIN_SEED,
    )

    history = trainer.train()
    train_time = trainer.train_wall_time

    # Plot loss curve
    plot_loss_curve(history, figures_dir=FIGURES_DIR, filename="loss_curve.png")

    print(f"\n[PINN] Training wall time: {train_time:.2f} s ({train_time/60:.2f} min)")
    return model, history, train_time


def load_pinn_from_checkpoint(checkpoint_path: str = None):
    """Load a trained PINN from disk. Returns (model, history, train_time_s)."""
    if checkpoint_path is None:
        checkpoint_path = str(CHECKPOINTS_DIR / "pinn_checkpoint_final.pt")

    model = HeatPINN(hidden_dims=PINN_HIDDEN_DIMS,
                     t_final=T_FINAL,
                     activation=PINN_ACTIVATION)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    history = ckpt.get("history", {})
    train_time = ckpt.get("train_time_s", None)
    print(f"[PINN] Loaded checkpoint from {checkpoint_path}")

    if train_time is None:
        print("[PINN] WARNING: this checkpoint predates train_time_s tracking — "
              "training time is unknown. Amortized-cost analysis (benchmarking.py) "
              "will be unable to compute a break-even point until you retrain "
              "(without --skip-train) to produce a checkpoint with this field.")
        train_time = float("nan")
    else:
        print(f"[PINN] Recovered training time from checkpoint: {train_time:.2f} s "
              f"({train_time/60:.2f} min)")

    if history:
        plot_loss_curve(history, figures_dir=FIGURES_DIR, filename="loss_curve.png")

    return model, history, train_time

def run_evaluation(model, fd_result):
    print("\n" + "=" * 60)
    print("PHASE 8: QUANTITATIVE EVALUATION (PINN vs. FD)")
    print("=" * 60)

    x_fd = fd_result["x"]
    y_fd = fd_result["y"]
    eval_times_present = [t for t in EVAL_TIMES if t in fd_result["snapshots"]]

    print("\n[Eval] Error metrics at each evaluation time:")
    eval_results = evaluate_at_times(model, fd_result, eval_times_present, ALPHA)

    # IC error
    print("\n[Eval] IC error (t=0):")
    ic_err = evaluate_ic_error(model, x_fd, y_fd,
                                IC_AMPLITUDE, IC_CENTER_X, IC_CENTER_Y, IC_SIGMA)
    print(f"  IC L2_rel={ic_err['l2_rel']:.4e}  RMSE={ic_err['rmse']:.4e}  "
          f"MaxAE={ic_err['max_ae']:.4e}")

    # BC error
    print("\n[Eval] BC error (all walls, all eval times):")
    bc_err = evaluate_bc_error(model, x_fd, y_fd, eval_times_present)
    print(f"  Max|û_BC|={bc_err['max_bc_error']:.4e}  "
          f"Mean|û_BC|={bc_err['mean_bc_error']:.4e}  "
          f"RMS|û_BC|={bc_err['rms_bc_error']:.4e}")

    # Inference timing
    print("\n[Eval] PINN inference timing:")
    infer_timing = time_inference(model, n_query_points=FD_NX * FD_NY,
                                   n_repeats=10, t_final=T_FINAL)
    print(f"  n_points={infer_timing['n_points']}: "
          f"mean={infer_timing['mean_s']*1000:.3f} ms, "
          f"min={infer_timing['min_s']*1000:.3f} ms")

    return eval_results, ic_err, bc_err, infer_timing


# Phase: Visualization

def run_visualization(eval_results, fd_result):
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)

    x_fd = fd_result["x"]
    y_fd = fd_result["y"]
    eval_times_present = sorted(eval_results.keys())

    # Main comparison grid (FD | PINN | Error)
    plot_comparison_row(eval_results, eval_times_present, x_fd, y_fd,
                        figures_dir=FIGURES_DIR, filename="comparison_all_times.png")

    # Cross-sections at y=0.5
    plot_cross_section(eval_results, eval_times_present, x_fd, y_fd,
                       y_slice=0.5, figures_dir=FIGURES_DIR,
                       filename="cross_sections.png")

    # Error vs. time bar chart
    plot_error_vs_time(eval_results, figures_dir=FIGURES_DIR,
                       filename="error_vs_time.png")

    # Individual signed error maps
    for t_val in eval_times_present:
        if t_val == 0.0:
            continue
        plot_error_map(eval_results[t_val]["error"], x_fd, y_fd, t_val,
                       eval_results[t_val]["l2_rel"],
                       figures_dir=FIGURES_DIR,
                       filename=f"error_map_t{t_val:.2f}.png")

    print("[Vis] All figures saved to figures/")


# Save summary results


def save_summary(conv_results, fd_result, history, eval_results,
                 ic_err, bc_err, infer_timing, train_time):
    """Save all numerical results as JSON for reproducibility."""
    summary = {
        "fd_validation": {
            "convergence": [{"nx": r["nx"], "l2_rel": r["l2_rel"],
                              "rmse": r["rmse"], "max_ae": r["max_ae"]}
                            for r in conv_results],
            "solve_time_s": fd_result["solve_time"],
        },
        "pinn_training": {
            "train_time_s"   : train_time,
            "final_total_loss": history["total"][-1] if history.get("total") else None,
            "final_pde_loss"  : history["pde"][-1]   if history.get("pde")   else None,
            "final_ic_loss"   : history["ic"][-1]    if history.get("ic")    else None,
            "final_bc_loss"   : history["bc"][-1]    if history.get("bc")    else None,
            "n_epochs"        : len(history.get("epoch", [])),
        },
        "evaluation": {
            str(t): {k: v for k, v in res.items()
                     if k not in ("u_pinn", "u_fd", "error")}
            for t, res in eval_results.items()
        },
        "ic_error": {k: v for k, v in ic_err.items()
                     if k not in ("u_pinn_0", "u0_exact")},
        "bc_error": bc_err,
        "inference_timing": infer_timing,
    }

    path = RESULTS_DIR / "baseline_results.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Summary] Results saved to {path}")
    return summary


# Main


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 60)
    print("PINN vs. FD BASELINE EXPERIMENT")
    print("2D Transient Heat Equation")
    print("=" * 60)
    print(f"  alpha={ALPHA}, T={T_FINAL}, Gaussian IC (A={IC_AMPLITUDE}, "
          f"x0={IC_CENTER_X}, y0={IC_CENTER_Y}, sigma={IC_SIGMA})")
    print(f"  FD grid: {FD_NX}×{FD_NY}, Δt={FD_DT:.2e}")
    print(f"  PINN: {PINN_HIDDEN_DIMS} (tanh), seed={TRAIN_SEED}")

    total_start = time.perf_counter()

    #  FD validation
    conv_results, _ = run_fd_validation()

    # FD reference
    fd_result = run_fd_reference()

    if args.fd_only:
        print("\n[Main] --fd-only flag set. Stopping before PINN.")
        sys.exit(0)

    # PhasePINN training (or load checkpoint)
    train_time = 0.0
    history = {}
    if args.skip_train:
        model, history, train_time = load_pinn_from_checkpoint(args.checkpoint)
    else:
        model, history, train_time = run_pinn_training(
            n_epochs=args.epochs
        )

    #  Evaluation
    eval_results, ic_err, bc_err, infer_timing = run_evaluation(model, fd_result)

    # Visualization
    run_visualization(eval_results, fd_result)

    # Save summary
    save_summary(conv_results, fd_result, history, eval_results,
                 ic_err, bc_err, infer_timing, train_time)

    total_wall = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"Baseline experiment complete. Total wall time: {total_wall:.1f} s "
          f"({total_wall/60:.1f} min)")
    print(f"{'='*60}")
