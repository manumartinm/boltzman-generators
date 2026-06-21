"""Abstract base class for benchmark energy functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class EnergyModel(ABC):
    """Reduced energy u(x) = U(x) / (kT) in dimension ``dim``."""

    dim: int

    @abstractmethod
    def energy(self, x: Tensor) -> Tensor:
        """Evaluate reduced energy on batch ``x`` of shape ``(..., dim)``."""

    def __call__(self, x: Tensor) -> Tensor:
        return self.energy(x)
