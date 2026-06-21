"""Abstract base class for Boltzmann generator density models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@runtime_checkable
class DensityModel(Protocol):
    """Structural typing contract for density models used in losses and sampling."""

    def sample(self, n: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
        """Draw ``n`` samples; return ``(x, log_q(x))``."""

    def log_prob(self, x: Tensor) -> Tensor:
        """Log-density ``log q(x)`` for batch ``x``."""


class BaseDensityModel(nn.Module, ABC):
    """PyTorch module implementing a tractable approximate Boltzmann density."""

    @abstractmethod
    def sample(self, n: int, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
        """Draw ``n`` samples in data space; return ``(x, log_q(x))``."""

    @abstractmethod
    def log_prob(self, x: Tensor) -> Tensor:
        """Log-density ``log q(x)`` for batch ``x``."""

    def nll(self, x: Tensor) -> Tensor:
        """Negative log-likelihood (forward KL up to constant). Mean over batch."""
        return -self.log_prob(x).mean()
