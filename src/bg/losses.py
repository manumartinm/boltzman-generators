"""Boltzmann Generator losses.

`kl_by_example` = NLL on samples (forward KL, mode-covering).
`kl_by_energy` = reverse KL using only the energy function (mode-seeking).
`mixed_loss` = weighted combination, optionally with energy clipping for
stability in early training.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .flows.base import FlowModel

EnergyFn = Callable[[Tensor], Tensor]


def kl_by_example(model: FlowModel, x: Tensor) -> Tensor:
    """Forward-KL training (= MLE up to constant). Needs target samples."""
    return -model.log_prob(x).mean()


def kl_by_energy(
    model: FlowModel,
    energy_fn: EnergyFn,
    n_samples: int,
    device: torch.device | str,
    energy_max: float | None = None,
) -> Tensor:
    """Reverse-KL training. Needs only the energy function.

    `energy_max`: if set, clamp `u(x)` from above. Useful early in training
    when the flow may sample very high-energy regions and produce exploding
    gradients (paper Sec. S4).
    """
    z = model.prior.sample(n_samples, device=device)
    x, log_det_fwd = model.flow.forward(z)
    u = energy_fn(x)
    if energy_max is not None:
        u = torch.clamp(u, max=energy_max)
    # KL(q || p) = E_q[log q - log p] = E_q[log q + u] + const
    # log q(x) = log q_Z(z) - log_det_fwd. log q_Z(z) is const wrt theta.
    # So gradient-relevant loss: E[u(x) - log_det_fwd].
    return (u - log_det_fwd).mean()


def mixed_loss(
    model: FlowModel,
    energy_fn: EnergyFn,
    x_data: Tensor | None,
    n_samples: int,
    device: torch.device | str,
    w_ml: float = 1.0,
    w_kl: float = 1.0,
    energy_max: float | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Combined loss. If `x_data` is None or `w_ml`=0, pure KL_z."""
    parts: dict[str, float] = {}
    loss = torch.zeros((), device=device)
    if w_ml > 0 and x_data is not None:
        l_ml = kl_by_example(model, x_data)
        loss = loss + w_ml * l_ml
        parts["kl_x"] = l_ml.item()
    if w_kl > 0:
        l_kl = kl_by_energy(model, energy_fn, n_samples, device=device, energy_max=energy_max)
        loss = loss + w_kl * l_kl
        parts["kl_z"] = l_kl.item()
    return loss, parts
