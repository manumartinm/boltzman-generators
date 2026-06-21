"""Base classes for normalizing flows.

Convention used everywhere in this package:

- `forward(z)` maps prior space → data space (sampling direction).
  Returns `(x, log_det)` where `log_det = log|det df/dz|`.
- `inverse(x)` maps data space → prior space (density direction).
  Returns `(z, log_det)` where `log_det = log|det df^-1/dx| = -log|det df/dz|`.

With this convention:
    log p_X(x) = log p_Z(z) + log|det df^-1/dx|   # the inverse log-det
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from ..base.density import BaseDensityModel


class Flow(nn.Module, ABC):
    """Invertible transformation z <-> x with tractable log-determinant."""

    @abstractmethod
    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def inverse(self, x: Tensor) -> tuple[Tensor, Tensor]: ...


class GaussianPrior(nn.Module):
    """Standard normal prior N(0, I) of given dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("_log_norm", torch.tensor(0.5 * dim * math.log(2 * math.pi)))

    def sample(self, n: int, device: torch.device | str = "cpu") -> Tensor:
        return torch.randn(n, self.dim, device=device)

    def log_prob(self, z: Tensor) -> Tensor:
        return -0.5 * z.pow(2).sum(dim=-1) - self._log_norm


class FlowModel(BaseDensityModel):
    """Flow stack + prior. Provides sample, log_prob, and forward KL loss."""

    def __init__(self, prior: GaussianPrior, flow: Flow) -> None:
        super().__init__()
        self.prior = prior
        self.flow = flow

    def sample(self, n: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
        """Draw n samples in data space. Returns (x, log_prob_x)."""
        z = self.prior.sample(n, device=device)
        log_pz = self.prior.log_prob(z)
        x, log_det_fwd = self.flow.forward(z)
        log_px = log_pz - log_det_fwd
        return x, log_px

    def log_prob(self, x: Tensor) -> Tensor:
        z, log_det_inv = self.flow.inverse(x)
        return self.prior.log_prob(z) + log_det_inv
