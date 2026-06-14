"""Sampling utilities: i.i.d. samples from a trained flow + importance weights."""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .flows.base import FlowModel

EnergyFn = Callable[[Tensor], Tensor]


def sample_with_weights(
    model: FlowModel,
    energy_fn: EnergyFn,
    n: int,
    device: torch.device | str = "cpu",
    chunk: int = 4096,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample x ~ q_model and compute log importance weights log w = -u(x) - log q(x).

    Returns (x, log_w_unnormalized, log_q). Constants do not matter for
    self-normalized estimators.
    """
    xs, lws, lqs = [], [], []
    remaining = n
    while remaining > 0:
        k = min(chunk, remaining)
        with torch.no_grad():
            x, log_q = model.sample(k, device=device)
            u = energy_fn(x)
            log_w = -u - log_q
        xs.append(x.cpu())
        lws.append(log_w.cpu())
        lqs.append(log_q.cpu())
        remaining -= k
    return torch.cat(xs), torch.cat(lws), torch.cat(lqs)


def effective_sample_size(log_w: Tensor) -> float:
    """ESS = (sum w)^2 / sum w^2, numerically stable via log-sum-exp."""
    log_sum_w = torch.logsumexp(log_w, dim=0)
    log_sum_w2 = torch.logsumexp(2 * log_w, dim=0)
    return float(torch.exp(2 * log_sum_w - log_sum_w2))


def normalized_weights(log_w: Tensor) -> Tensor:
    """Self-normalized weights summing to 1."""
    log_w_shifted = log_w - log_w.max()
    w = torch.exp(log_w_shifted)
    return w / w.sum()


def free_energy_diff(
    x: Tensor,
    log_w: Tensor,
    region_A: Callable[[Tensor], Tensor],
    region_B: Callable[[Tensor], Tensor],
) -> float:
    """ΔF_AB = -log(Z_B/Z_A) in units of kT. Region indicators are bool masks."""
    in_A = region_A(x).bool()
    in_B = region_B(x).bool()
    log_ZA = torch.logsumexp(log_w[in_A], dim=0)
    log_ZB = torch.logsumexp(log_w[in_B], dim=0)
    return float(-(log_ZB - log_ZA))
