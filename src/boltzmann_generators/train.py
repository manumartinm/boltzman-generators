"""Training loops for flow-based Boltzmann Generators and CNF models."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import torch
from torch import Tensor

from .base.density import BaseDensityModel
from .base.energy import EnergyModel
from .training import TrainCallback, TrainConfig, Trainer

__all__ = ["TrainCallback", "TrainConfig", "Trainer", "train_bg"]


def train_bg(
    model: BaseDensityModel,
    energy_fn: EnergyModel | Callable[[Tensor], Tensor],
    x_data: Tensor | None,
    config: TrainConfig,
    *,
    device: torch.device | str = "cpu",
    callback: TrainCallback | None = None,
) -> list[dict[str, float]]:
    """Train a BG model with mixed loss.

    .. deprecated::
        Use :class:`boltzmann_generators.training.Trainer` instead.

    If ``x_data`` is None, this falls back to pure KL_z training.
    """
    warnings.warn(
        "train_bg() is deprecated; use boltzmann_generators.training.Trainer instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    trainer = Trainer(model, energy_fn, config, device=device)
    return trainer.fit(x_data, callback=callback)
