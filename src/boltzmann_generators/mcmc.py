"""Metropolis-Hastings sampling helpers used across notebooks."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

EnergyFn = Callable[[Tensor], Tensor]


def mcmc(
    energy_fn: EnergyFn,
    n_steps: int,
    *,
    x0: Tensor | None = None,
    sigma: float = 0.5,
    n_chains: int = 1,
    dim: int = 2,
    device: torch.device | str = "cpu",
    return_acceptance: bool = False,
) -> Tensor | tuple[Tensor, float]:
    """Run a random-walk Metropolis sampler.

    Args:
        energy_fn: Potential energy function in units of kT.
        n_steps: Number of MCMC steps per chain.
        x0: Optional initial states of shape (n_chains, dim).
        sigma: Proposal standard deviation.
        n_chains: Number of parallel chains if ``x0`` is not provided.
        dim: State dimension if ``x0`` is not provided.
        device: Device where sampling runs.
        return_acceptance: If True, also return acceptance ratio.

    Returns:
        Flattened samples of shape (n_steps * n_chains, dim), and optionally
        acceptance ratio.
    """
    if x0 is None:
        x = torch.zeros(n_chains, dim, device=device)
    else:
        x = x0.to(device=device).clone()
        if x.ndim != 2:
            raise ValueError("x0 must have shape (n_chains, dim)")
        n_chains = x.shape[0]
        dim = x.shape[1]

    u = energy_fn(x)
    samples = torch.empty(n_steps, n_chains, dim, device=device, dtype=x.dtype)
    accepted = 0

    for step in range(n_steps):
        proposal = x + sigma * torch.randn_like(x)
        u_prop = energy_fn(proposal)
        log_alpha = -(u_prop - u)
        alpha = torch.exp(torch.clamp(log_alpha, max=0.0))
        accept = torch.rand(n_chains, device=device) < alpha
        x = torch.where(accept[:, None], proposal, x)
        u = torch.where(accept, u_prop, u)
        samples[step] = x
        accepted += int(accept.sum().item())

    flat = samples.reshape(-1, dim)
    if return_acceptance:
        ratio = accepted / float(n_steps * n_chains)
        return flat, ratio
    return flat
