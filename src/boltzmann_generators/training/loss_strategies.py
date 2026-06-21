"""Loss strategy classes for Boltzmann Generator training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import Tensor

from ..base.density import DensityModel
from ..base.energy import EnergyModel

EnergyFn = Callable[[Tensor], Tensor]


class LossStrategy(ABC):
    """Compute a scalar training loss and per-component diagnostics."""

    @abstractmethod
    def compute(
        self,
        model: DensityModel,
        energy_fn: EnergyModel | EnergyFn,
        x_data: Tensor | None,
        *,
        n_samples: int,
        device: torch.device | str,
    ) -> tuple[Tensor, dict[str, float]]: ...


class ForwardKLLoss(LossStrategy):
    """Maximum-likelihood / forward-KL loss on target samples."""

    def compute(
        self,
        model: DensityModel,
        energy_fn: EnergyModel | EnergyFn,
        x_data: Tensor | None,
        *,
        n_samples: int,
        device: torch.device | str,
    ) -> tuple[Tensor, dict[str, float]]:
        del energy_fn, n_samples, device
        if x_data is None:
            raise ValueError("ForwardKLLoss requires x_data.")
        loss = -model.log_prob(x_data).mean()
        return loss, {"kl_x": float(loss.item())}


class ReverseKLLoss(LossStrategy):
    """Reverse-KL loss using only the energy function."""

    def __init__(self, energy_max: float | None = None) -> None:
        self.energy_max = energy_max

    def compute(
        self,
        model: DensityModel,
        energy_fn: EnergyModel | EnergyFn,
        x_data: Tensor | None,
        *,
        n_samples: int,
        device: torch.device | str,
    ) -> tuple[Tensor, dict[str, float]]:
        del x_data
        x, log_q = model.sample(n_samples, device=device)
        u = energy_fn(x)
        if self.energy_max is not None:
            u = torch.clamp(u, max=self.energy_max)
        loss = (u + log_q).mean()
        return loss, {"kl_z": float(loss.item())}


class MixedLossStrategy(LossStrategy):
    """Weighted combination of forward- and reverse-KL terms."""

    def __init__(
        self,
        *,
        w_ml: float = 1.0,
        w_kl: float = 1.0,
        energy_max: float | None = None,
    ) -> None:
        self.w_ml = w_ml
        self.w_kl = w_kl
        self._forward = ForwardKLLoss()
        self._reverse = ReverseKLLoss(energy_max=energy_max)

    def compute(
        self,
        model: DensityModel,
        energy_fn: EnergyModel | EnergyFn,
        x_data: Tensor | None,
        *,
        n_samples: int,
        device: torch.device | str,
    ) -> tuple[Tensor, dict[str, float]]:
        parts: dict[str, float] = {}
        loss = torch.zeros((), device=device)
        if self.w_ml > 0 and x_data is not None:
            l_ml, p_ml = self._forward.compute(
                model, energy_fn, x_data, n_samples=n_samples, device=device
            )
            loss = loss + self.w_ml * l_ml
            parts.update(p_ml)
        if self.w_kl > 0:
            l_kl, p_kl = self._reverse.compute(
                model, energy_fn, x_data, n_samples=n_samples, device=device
            )
            loss = loss + self.w_kl * l_kl
            parts.update(p_kl)
        return loss, parts
