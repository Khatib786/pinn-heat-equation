"""
training.py
-----------
Trainer class for the PINN: Adam optimizer, cosine LR schedule,
loss history logging, and checkpoint save/load.
"""

import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Callable


class Trainer:
    """
    Full training loop for HeatPINN.

    Parameters
    ----------
    model        : HeatPINN instance
    loss_fn      : callable(model) → dict with keys 'total','pde','ic','bc'
                   (i.e., a closure that captures collocation points)
    lr_init      : initial Adam learning rate
    lr_final     : final LR after cosine decay
    n_epochs     : total training epochs
    checkpoint_dir : directory to save/load checkpoints
    checkpoint_interval : save every N epochs
    print_interval      : print loss every N epochs
    early_stop_loss     : stop if total loss < this threshold
    seed         : torch manual seed
    """

    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: Callable,
                 lr_init: float          = 1e-3,
                 lr_final: float         = 1e-5,
                 n_epochs: int           = 20_000,
                 checkpoint_dir: str     = "checkpoints",
                 checkpoint_interval: int = 1_000,
                 print_interval: int      = 500,
                 early_stop_loss: float   = 1e-6,
                 seed: int                = 42):

        self.model              = model
        self.loss_fn            = loss_fn
        self.lr_init            = lr_init
        self.lr_final           = lr_final
        self.n_epochs           = n_epochs
        self.checkpoint_dir     = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.print_interval     = print_interval
        self.early_stop_loss    = early_stop_loss
        self.seed               = seed

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=n_epochs,
            eta_min=lr_final,
        )

        # History — separate lists for each loss component
        self.history = {
            "epoch"   : [],
            "total"   : [],
            "pde"     : [],
            "ic"      : [],
            "bc"      : [],
            "lr"      : [],
        }

        self._train_start_time  = None
        self._train_wall_time   = 0.0   # accumulated wall time

    def train(self) -> dict:
        """
        Run the training loop.

        Returns
        -------
        history : dict of loss curves (can be used for plotting)
        """
        model = self.model
        model.train()

        print(f"\n[Trainer] Starting training: {self.n_epochs} epochs, "
              f"Adam lr {self.lr_init} → {self.lr_final} (cosine)\n")
        print(f"{'Epoch':>8}  {'Total':>12}  {'PDE':>12}  "
              f"{'IC':>12}  {'BC':>12}  {'LR':>10}")
        print("-" * 72)

        self._train_start_time = time.perf_counter()

        for epoch in range(1, self.n_epochs + 1):
            self.optimizer.zero_grad()

            losses = self.loss_fn(model)
            total  = losses["total"]

            total.backward()
            self.optimizer.step()
            self.scheduler.step()

            # --- Logging ---
            l_total = total.item()
            l_pde   = losses["pde"].item()
            l_ic    = losses["ic"].item()
            l_bc    = losses["bc"].item()
            lr_now  = self.scheduler.get_last_lr()[0]

            self.history["epoch"].append(epoch)
            self.history["total"].append(l_total)
            self.history["pde"].append(l_pde)
            self.history["ic"].append(l_ic)
            self.history["bc"].append(l_bc)
            self.history["lr"].append(lr_now)

            # --- Print ---
            if epoch % self.print_interval == 0 or epoch == 1:
                elapsed = time.perf_counter() - self._train_start_time
                print(f"{epoch:>8d}  {l_total:>12.4e}  {l_pde:>12.4e}  "
                      f"{l_ic:>12.4e}  {l_bc:>12.4e}  {lr_now:>10.2e}  "
                      f"[{elapsed:.0f}s]")

            # --- Checkpoint ---
            if epoch % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch)

            # --- Early stopping ---
            if l_total < self.early_stop_loss:
                print(f"\n[Trainer] Early stop at epoch {epoch}: "
                      f"total loss = {l_total:.2e} < {self.early_stop_loss:.2e}")
                self._save_checkpoint(epoch, tag="early_stop")
                break

        self._train_wall_time = time.perf_counter() - self._train_start_time
        print(f"\n[Trainer] Training complete. "
              f"Total wall time: {self._train_wall_time:.2f} s "
              f"({self._train_wall_time/60:.1f} min)")

        # Save final checkpoint
        self._save_checkpoint(epoch, tag="final")
        self._save_history()

        return self.history

    def _save_checkpoint(self, epoch: int, tag: str = "") -> None:
        """Save model state dict, optimizer state, and elapsed training time."""
        suffix = f"_ep{epoch}" if not tag else f"_{tag}"
        path = self.checkpoint_dir / f"pinn_checkpoint{suffix}.pt"
        
        elapsed = (time.perf_counter() - self._train_start_time
                   if self._train_start_time is not None else 0.0)
        torch.save({
            "epoch"           : epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state" : self.optimizer.state_dict(),
            "history"         : self.history,
            "train_time_s"    : elapsed,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """
        Load a saved checkpoint into model and optimizer.
        Returns the epoch at which training was saved.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.history = ckpt.get("history", self.history)
        self._train_wall_time = ckpt.get("train_time_s", 0.0)
        epoch = ckpt["epoch"]
        print(f"[Trainer] Loaded checkpoint from '{path}' (epoch {epoch})")
        return epoch

    def _save_history(self) -> None:
        """Save loss history as JSON for later analysis."""
        path = self.checkpoint_dir / "loss_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Trainer] Loss history saved to {path}")

    @property
    def train_wall_time(self) -> float:
        """Total training wall-clock time in seconds."""
        return self._train_wall_time
