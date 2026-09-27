"""
 Tier 2: Computational benchmarking and amortized cost analysis.
 Measures:
 1. FD reference solve time using the Gaussian initial condition.
 2. PINN training time from saved checkpoint metadata.
 3. PINN inference time for different numbers of query points.
 4. FD solve time for repeated simulations.
 5. Amortized cost and crossover analysis.
 Outputs:
 results/benchmarking_results.json
       Usage:
python experiments/benchmarking.py [--checkpoint PATH]
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
    RESULTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR,
)
from fd_solver import FDSolver, make_gaussian_ic
from pinn_model import HeatPINN
from evaluation import time_inference
from visualization import plot_amortization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking + amortized cost")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n-fd-repeats", type=int, default=5,
                        help="Number of FD timing repetitions")
    return parser.parse_args()


# 1. FD timing

def time_fd_solve(n_repeats: int = 5) -> dict:
    """Time the full FD solve (Gaussian IC, same params as baseline)."""
    solver = FDSolver(nx=FD_NX, ny=FD_NY, alpha=ALPHA,
                      t_final=T_FINAL, dt=FD_DT, safety=FD_SAFETY)
    x, y = solver.x, solver.y
    u0 = make_gaussian_ic(x, y, IC_AMPLITUDE, IC_CENTER_X, IC_CENTER_Y, IC_SIGMA)

    times = []
    print(f"\n[Benchmark] Timing FD solve ({n_repeats} repetitions)...")
    for i in range(n_repeats):
        t0 = time.perf_counter()
        solver.solve(u0, save_times=[0.0, T_FINAL], verbose=False)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Rep {i+1}: {elapsed:.4f} s")

    return {
        "mean_s": float(np.mean(times)),
        "std_s" : float(np.std(times)),
        "min_s" : float(np.min(times)),
        "times" : times,
    }


# 2. Load PINN + get training time from baseline results
def load_model_and_train_time(checkpoint_path: str = None) -> tuple:
    """Load trained PINN. Try to read training time from baseline_results.json."""
    if checkpoint_path is None:
        checkpoint_path = str(CHECKPOINTS_DIR / "pinn_checkpoint_final.pt")

    model = HeatPINN(hidden_dims=PINN_HIDDEN_DIMS,
                     t_final=T_FINAL, activation=PINN_ACTIVATION)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Try to recover training time from saved results
    train_time = None
    baseline_json = RESULTS_DIR / "baseline_results.json"
    if baseline_json.exists():
        with open(baseline_json) as f:
            baseline = json.load(f)
        train_time = baseline.get("pinn_training", {}).get("train_time_s", None)
        # NOTE: check `is None`, not truthiness - a real train_time_s of 0.0
        # (e.g. written by a skip-train run) is falsy but NOT missing data,
        # and must still trigger the warning below rather than being silently
        # accepted as "the PINN trained in 0 seconds".
        if train_time is not None:
            print(f"[Benchmark] PINN training time from baseline: {train_time:.2f} s "
                  f"({train_time/60:.2f} min)")

    if train_time is None or train_time <= 0.0:
        print("[Benchmark] WARNING: valid training time not found in baseline_results.json "
              "(missing, null, or 0.0 — likely from a --skip-train run that never recorded "
              "the real training time). Amortized-cost analysis cannot proceed without it. "
              "Set train_time_s manually or re-run baseline.py WITHOUT --skip-train first.")
        train_time = float("nan")

    return model, train_time


# 3. PINN inference timing at multiple point counts

def time_pinn_inference_scaling(model, point_counts: list[int],
                                 n_repeats: int = 10) -> list[dict]:
    """Time PINN inference at multiple query sizes."""
    results = []
    print(f"\n[Benchmark] PINN inference timing at multiple query sizes...")
    for n_pts in point_counts:
        r = time_inference(model, n_query_points=n_pts,
                           n_repeats=n_repeats, t_final=T_FINAL)
        results.append(r)
        print(f"  N={n_pts:7d}: mean={r['mean_s']*1000:.3f} ms  "
              f"min={r['min_s']*1000:.3f} ms")
    return results


# 4. Amortized cost analysis

def compute_amortized(t_fd: float, t_train: float,
                       t_infer_per_eval: float,
                       n_max: int = 300) -> dict:
    """
    Compute amortized cost curves and break-even point N*.

    C_FD(N)   = N × t_fd
    C_PINN(N) = t_train + N × t_infer_per_eval

    N* = t_train / (t_fd - t_infer_per_eval)   if denominator > 0
    """
    if np.isnan(t_train):
        # No valid training time available - cannot compute a meaningful
        # break-even point. Fail loudly rather than reporting a fabricated
        # or misleading crossover.
        n_star = None
        crossover_exists = False
    else:
        denom = t_fd - t_infer_per_eval
        if denom > 0:
            n_star = t_train / denom
            crossover_exists = True
        else:
            n_star = None
            crossover_exists = False

    N_arr = np.arange(0, n_max + 1)
    c_fd   = N_arr * t_fd
    c_pinn = t_train + N_arr * t_infer_per_eval

    return {
        "n_star"           : float(n_star) if n_star else None,
        "crossover_exists" : crossover_exists,
        "t_fd_s"           : t_fd,
        "t_train_s"        : t_train,
        "t_infer_per_eval_s": t_infer_per_eval,
        "N_arr"            : N_arr.tolist(),
        "c_fd"             : c_fd.tolist(),
        "c_pinn"           : c_pinn.tolist(),
    }


# 5. Inference scaling plot

def plot_inference_scaling(infer_results: list[dict],
                            figures_dir: Path) -> None:
    """Log-log plot of inference time vs. number of query points."""
    n_pts  = [r["n_points"] for r in infer_results]
    mean_s = [r["mean_s"]   for r in infer_results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(n_pts, mean_s, "ro-", lw=2, ms=7, label="PINN inference")
    ax.set_xlabel("Number of query points N")
    ax.set_ylabel("Inference time (s)")
    ax.set_title("PINN Inference Time vs. Query Size")
    ax.legend(); ax.grid(True, alpha=0.4, which="both")
    fig.tight_layout()
    path = figures_dir / "inference_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Benchmark] Saved {path}")


# Main

if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 60)
    print("BENCHMARKING + AMORTIZED COST ANALYSIS")
    print("=" * 60)

    # 1. FD timing
    fd_timing = time_fd_solve(n_repeats=args.n_fd_repeats)
    print(f"\n[Benchmark] FD solve: mean={fd_timing['mean_s']:.4f} s  "
          f"std={fd_timing['std_s']:.4f} s  min={fd_timing['min_s']:.4f} s")

    # 2. Load PINN
    model, train_time = load_model_and_train_time(args.checkpoint)

    # 3. Inference timing at various query sizes
    # The "equivalent FD evaluation" = full 64×64 grid = FD_NX × FD_NY points
    grid_size = FD_NX * FD_NY
    point_counts = [100, 500, 1000, grid_size, 10 * grid_size]
    infer_results = time_pinn_inference_scaling(model, point_counts, n_repeats=10)

    # The "fair" comparison: PINN evaluated on the full FD grid
    grid_infer = next(r for r in infer_results if r["n_points"] == grid_size)
    t_infer_per_eval = grid_infer["mean_s"]

    # 4. Amortized analysis
    print(f"\n[Benchmark] Computing amortized cost...")
    print(f"  t_FD          = {fd_timing['mean_s']:.4f} s")
    print(f"  t_train       = {train_time:.2f} s")
    print(f"  t_infer/eval  = {t_infer_per_eval*1000:.3f} ms")

    amort = compute_amortized(
        t_fd=fd_timing["mean_s"],
        t_train=train_time,
        t_infer_per_eval=t_infer_per_eval,
        n_max=500,
    )

    if amort["crossover_exists"]:
        print(f"\n[Benchmark] Break-even N* = {amort['n_star']:.1f}")
        print(f"  After {amort['n_star']:.0f} evaluations, PINN amortizes its training cost.")
    elif np.isnan(train_time):
        print(f"\n[Benchmark] No amortized-cost result: training time is unavailable (NaN). "
              f"Re-run baseline.py without --skip-train before drawing any conclusion here.")
    else:
        print(f"\n[Benchmark] No crossover: PINN inference ({t_infer_per_eval*1000:.3f} ms) "
              f">= FD solve ({fd_timing['mean_s']*1000:.3f} ms).")
        print("  The PINN does not offer a computational advantage on this problem/hardware.")

    # 5. Plots
    plot_amortization(
        t_fd=fd_timing["mean_s"],
        t_train=train_time,
        t_infer_per_eval=t_infer_per_eval,
        n_max=min(500, max(200, int(3 * (amort["n_star"] or 500)))),
        figures_dir=FIGURES_DIR,
        filename="amortization.png",
    )
    plot_inference_scaling(infer_results, FIGURES_DIR)

    # 6. Save results
    results = {
        "fd_timing"   : {k: v for k, v in fd_timing.items() if k != "times"},
        "train_time_s": train_time,
        "infer_timing": infer_results,
        "amortization": {k: v for k, v in amort.items()
                         if k not in ("N_arr", "c_fd", "c_pinn")},
    }
    path = RESULTS_DIR / "benchmarking_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Benchmark] Results saved to {path}")
    print("\nBenchmarking complete.")
