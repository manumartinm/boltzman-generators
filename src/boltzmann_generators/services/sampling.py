"""Importance-sampling engine for trained density models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ..base.density import BaseDensityModel, DensityModel
from ..base.energy import EnergyModel

EnergyFn = Callable[[Tensor], Tensor]


class SamplingEngine:
    """Draw weighted samples from a trained approximate Boltzmann density."""

    def __init__(
        self,
        model: BaseDensityModel | DensityModel,
        energy: EnergyModel | EnergyFn,
    ) -> None:
        self.model = model
        self.energy = energy

    def sample_with_weights(
        self,
        n: int,
        device: torch.device | str = "cpu",
        chunk: int = 4096,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample ``x ~ q`` and compute log importance weights ``log w = -u(x) - log q(x)``."""
        xs, lws, lqs = [], [], []
        remaining = n
        while remaining > 0:
            k = min(chunk, remaining)
            with torch.no_grad():
                x, log_q = self.model.sample(k, device=device)
                u = self.energy(x)
                log_w = -u - log_q
            xs.append(x.cpu())
            lws.append(log_w.cpu())
            lqs.append(log_q.cpu())
            remaining -= k
        return torch.cat(xs), torch.cat(lws), torch.cat(lqs)

    @staticmethod
    def effective_sample_size(log_w: Tensor) -> float:
        """ESS = (sum w)^2 / sum w^2, numerically stable via log-sum-exp."""
        log_sum_w = torch.logsumexp(log_w, dim=0)
        log_sum_w2 = torch.logsumexp(2 * log_w, dim=0)
        return float(torch.exp(2 * log_sum_w - log_sum_w2))

    @staticmethod
    def normalized_weights(log_w: Tensor) -> Tensor:
        """Self-normalized weights summing to 1."""
        log_w_shifted = log_w - log_w.max()
        w = torch.exp(log_w_shifted)
        return w / w.sum()

    @staticmethod
    def free_energy_diff(
        x: Tensor,
        log_w: Tensor,
        region_a: Callable[[Tensor], Tensor],
        region_b: Callable[[Tensor], Tensor],
    ) -> float:
        """ΔF_AB = -log(Z_B/Z_A) in units of kT."""
        in_a = region_a(x).bool()
        in_b = region_b(x).bool()
        log_za = torch.logsumexp(log_w[in_a], dim=0)
        log_zb = torch.logsumexp(log_w[in_b], dim=0)
        return float(-(log_zb - log_za))
