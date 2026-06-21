"""Double-well potentials in 1D and 2D.

All energies are returned as reduced (unitless) energies u(x) = U(x)/(kT).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base.energy import EnergyModel


class DoubleWell2D(EnergyModel):
    """u(x, y) = a*(x^2 - 1)^2 + 0.5/sigma^2 * y^2.

    Two minima at (±1, 0). The y direction is a harmonic well of width sigma.
    Parameter `a` controls the barrier height: barrier ≈ a (in kT).
    """

    def __init__(self, a: float = 4.0, sigma_y: float = 0.5) -> None:
        self.a = a
        self.sigma_y = sigma_y
        self.dim = 2

    def __call__(self, x: Tensor) -> Tensor:
        return self.energy(x)

    def energy(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 2
        xx = x[..., 0]
        yy = x[..., 1]
        return self.a * (xx.pow(2) - 1.0).pow(2) + 0.5 * (yy / self.sigma_y).pow(2)

    def grid(self, n: int = 200, span: float = 2.5) -> tuple[Tensor, Tensor, Tensor]:
        xs = torch.linspace(-span, span, n)
        ys = torch.linspace(-span, span, n)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
        u = self.energy(grid).reshape(n, n)
        return gx, gy, u


class DoubleWell1D(EnergyModel):
    """u(x) = a*(x^2 - 1)^2. Minima at ±1, barrier height = a (in kT)."""

    def __init__(self, a: float = 4.0) -> None:
        self.a = a
        self.dim = 1

    def __call__(self, x: Tensor) -> Tensor:
        return self.energy(x)

    def energy(self, x: Tensor) -> Tensor:
        xx = x[..., 0] if x.ndim > 1 else x
        return self.a * (xx.pow(2) - 1.0).pow(2)
