"""Boltzmann Generator losses.

`kl_by_example` = NLL on samples (forward KL, mode-covering).
`kl_by_energy` = reverse KL using only the energy function (mode-seeking).
`mixed_loss` = weighted combination, optionally with energy clipping for
stability in early training.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from .base.density import DensityModel
from .base.energy import EnergyModel
from .training.loss_strategies import (
    ForwardKLLoss,
    MixedLossStrategy,
    ReverseKLLoss,
)

EnergyFn = Callable[[Tensor], Tensor]

__all__ = [
    "DensityModel",
    "EnergyFn",
    "ForwardKLLoss",
    "MixedLossStrategy",
    "ReverseKLLoss",
    "kl_by_energy",
    "kl_by_example",
    "mixed_loss",
]


def kl_by_example(model: DensityModel, x: Tensor) -> Tensor:
    """Forward-KL training (= MLE up to constant). Needs target samples."""
    return -model.log_prob(x).mean()


def kl_by_energy(
    model: DensityModel,
    energy_fn: EnergyModel | EnergyFn,
    n_samples: int,
    device: torch.device | str,
    energy_max: float | None = None,
) -> Tensor:
    """Reverse-KL training. Needs only the energy function."""
    loss, _ = ReverseKLLoss(energy_max=energy_max).compute(
        model, energy_fn, None, n_samples=n_samples, device=device
    )
    return loss


def mixed_loss(
    model: DensityModel,
    energy_fn: EnergyModel | EnergyFn,
    x_data: Tensor | None,
    n_samples: int,
    device: torch.device | str,
    w_ml: float = 1.0,
    w_kl: float = 1.0,
    energy_max: float | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Combined loss. If `x_data` is None or `w_ml`=0, pure KL_z."""
    return MixedLossStrategy(w_ml=w_ml, w_kl=w_kl, energy_max=energy_max).compute(
        model, energy_fn, x_data, n_samples=n_samples, device=device
    )
