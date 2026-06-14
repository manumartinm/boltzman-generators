"""Continuous Normalizing Flow trained by Conditional Flow Matching.

Refs:
- Lipman, Chen, Ben-Hamu, Nickel, Le 2023, "Flow Matching for Generative Modeling", ICLR.
- Chen et al. 2018, "Neural ODEs" — the underlying generative model.

CFM with linear (optimal-transport) paths:
    x_t = (1 - t) z_0 + t x_1,  u_t(x_t | z_0, x_1) = x_1 - z_0

Training is plain MSE regression of the model's velocity onto (x_1 - z_0).
Sampling integrates the learned ODE with Euler (kept dependency-free; for
high-precision use torchdiffeq.odeint).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _fourier_features(t: Tensor, num_freqs: int = 8) -> Tensor:
    """Encode t ∈ [0,1] with sin/cos at increasing frequencies."""
    # t: (n,)
    freqs = 2 * math.pi * torch.arange(1, num_freqs + 1, device=t.device, dtype=t.dtype)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args), t[:, None]], dim=-1)


class VelocityField(nn.Module):
    """MLP v_theta(x, t) -> R^d. Concatenates Fourier-featured t with x."""

    def __init__(self, dim: int, hidden_dim: int = 128, num_hidden: int = 4, num_freqs: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.num_freqs = num_freqs
        t_dim = 2 * num_freqs + 1
        in_dim = dim + t_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_feat = _fourier_features(t, self.num_freqs)
        return self.net(torch.cat([x, t_feat], dim=-1))


class CNFModel(nn.Module):
    """Continuous normalizing flow + standard-normal prior.

    `sample(n, n_steps)` integrates the ODE with Euler from t=0 to t=1.
    """

    def __init__(self, velocity: VelocityField) -> None:
        super().__init__()
        self.v = velocity
        self.dim = velocity.dim

    def sample(self, n: int, device: torch.device | str = "cpu", n_steps: int = 100) -> Tensor:
        x = torch.randn(n, self.dim, device=device)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((n,), k * dt, device=device)
            v = self.v(x, t)
            x = x + dt * v
        return x

    def cfm_loss(self, x1: Tensor) -> Tensor:
        """Conditional flow matching loss with linear OT path."""
        n = x1.shape[0]
        t = torch.rand(n, device=x1.device)
        z0 = torch.randn_like(x1)
        x_t = (1 - t)[:, None] * z0 + t[:, None] * x1
        u_target = x1 - z0
        v_pred = self.v(x_t, t)
        return (v_pred - u_target).pow(2).sum(dim=-1).mean()
