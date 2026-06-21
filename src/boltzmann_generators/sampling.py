"""Sampling utilities: i.i.d. samples from a trained flow + importance weights."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from .base.density import BaseDensityModel, DensityModel
from .base.energy import EnergyModel
from .services.sampling import SamplingEngine

EnergyFn = Callable[[Tensor], Tensor]

__all__ = [
    "SamplingEngine",
    "effective_sample_size",
    "free_energy_diff",
    "normalized_weights",
    "sample_with_weights",
]


def sample_with_weights(
    model: BaseDensityModel | DensityModel,
    energy_fn: EnergyModel | EnergyFn,
    n: int,
    device: torch.device | str = "cpu",
    chunk: int = 4096,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample x ~ q_model and compute log importance weights log w = -u(x) - log q(x)."""
    engine = SamplingEngine(model, energy_fn)
    return engine.sample_with_weights(n, device=device, chunk=chunk)


def effective_sample_size(log_w: Tensor) -> float:
    """ESS = (sum w)^2 / sum w^2, numerically stable via log-sum-exp."""
    return SamplingEngine.effective_sample_size(log_w)


def normalized_weights(log_w: Tensor) -> Tensor:
    """Self-normalized weights summing to 1."""
    return SamplingEngine.normalized_weights(log_w)


def free_energy_diff(
    x: Tensor,
    log_w: Tensor,
    region_A: Callable[[Tensor], Tensor],
    region_B: Callable[[Tensor], Tensor],
) -> float:
    """ΔF_AB = -log(Z_B/Z_A) in units of kT. Region indicators are bool masks."""
    return SamplingEngine.free_energy_diff(x, log_w, region_A, region_B)
