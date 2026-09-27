"""
pinn_model.py
"""

import torch
import torch.nn as nn
from typing import Sequence


class HeatPINN(nn.Module):
    """
    Feed-forward neural network surrogate for the 2D transient heat equation.

    Parameters
    ----------
    hidden_dims : sequence of ints — number of neurons per hidden layer
    t_final     : final simulation time (used for t-normalization inside forward)
    activation  : activation function string; currently only 'tanh' supported
    """

    def __init__(self,
                 hidden_dims: Sequence[int] = (64, 64, 64, 64),
                 t_final: float = 0.5,
                 activation: str = "tanh"):
        super().__init__()

        self.t_final = t_final

        if activation == "tanh":
            act_cls = nn.Tanh
        elif activation == "sin":
            # Sinusoidal activation (alternative, not the baseline)
            act_cls = lambda: _SinActivation()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build layers
        layers = []
        in_dim = 3   # (x, y, t)
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # output layer (no activation)

        self.net = nn.Sequential(*layers)

        # Xavier initialization
        self._init_weights()

        # Count and report parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[PINN] Architecture: 3 → {list(hidden_dims)} → 1  "
              f"({n_params:,} trainable parameters)")

    def _init_weights(self):
        """Xavier uniform initialization for all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate û(x, y, t).

        Parameters
        ----------
        x, y, t : tensors of shape (N, 1) or (N,)
                  All must be on the same device.

        Returns
        -------
        u_hat : tensor of shape (N, 1)
        """
        # Ensure 2D column tensors
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        t = t.view(-1, 1)

        # Normalize t to [0, 1]; x and y are already in [0, 1]
        t_norm = t / self.t_final

        inp = torch.cat([x, y, t_norm], dim=1)   # (N, 3)
        return self.net(inp)                       # (N, 1)

    def predict_grid(self, x_grid: torch.Tensor, y_grid: torch.Tensor,
                     t_val: float) -> torch.Tensor:
        """
        Convenience method: evaluate û on a 2D meshgrid at a single time.

        Parameters
        ----------
        x_grid : 1D tensor of x-values (length Nx)
        y_grid : 1D tensor of y-values (length Ny)
        t_val  : scalar time value

        Returns
        -------
        u_hat : 2D tensor of shape (Ny, Nx) — same layout as FD solver output
        """
        Nx = x_grid.shape[0]
        Ny = y_grid.shape[0]

        # Build meshgrid
        X, Y = torch.meshgrid(x_grid, y_grid, indexing="xy")  # both (Nx, Ny) — need to transpose
        # Note: FD uses (Ny, Nx) layout (row=y, col=x); match that here
        X = X.T.reshape(-1, 1)   # (Ny*Nx, 1)
        Y = Y.T.reshape(-1, 1)
        T = torch.full_like(X, t_val)

        with torch.no_grad():
            u_flat = self.forward(X, Y, T)   # (Ny*Nx, 1)

        return u_flat.reshape(Ny, Nx)   # (Ny, Nx)


class _SinActivation(nn.Module):
    """Optional sinusoidal activation (not used in baseline)."""
    def forward(self, x):
        return torch.sin(x)
