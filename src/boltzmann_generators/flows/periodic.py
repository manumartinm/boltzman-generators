"""Periodic-coordinate helpers for angular systems."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PeriodicEmbedding(nn.Module):
    """Map angles in radians to cosine/sine features.

    Input shape: (..., d) with angles in [-pi, pi].
    Output shape: (..., 2d) as [cos(theta), sin(theta)] concatenation.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def periodic_inverse(x_embedded: Tensor) -> Tensor:
    """Recover angles from a cosine/sine embedding using atan2."""
    if x_embedded.shape[-1] % 2 != 0:
        raise ValueError("Embedded periodic tensor must have even feature dimension.")
    half = x_embedded.shape[-1] // 2
    cos_part = x_embedded[..., :half]
    sin_part = x_embedded[..., half:]
    return torch.atan2(sin_part, cos_part)
