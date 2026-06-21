"""Müller-Brown potential.

Standard 2D benchmark for rare-event sampling. Three minima with two saddle
points connecting them. Parameters from Müller & Brown (1979).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base.energy import EnergyModel

_A = torch.tensor([-200.0, -100.0, -170.0, 15.0])
_a = torch.tensor([-1.0, -1.0, -6.5, 0.7])
_b = torch.tensor([0.0, 0.0, 11.0, 0.6])
_c = torch.tensor([-10.0, -10.0, -6.5, 0.7])
_x0 = torch.tensor([1.0, 0.0, -0.5, -1.0])
_y0 = torch.tensor([0.0, 0.5, 1.5, 1.0])


class MullerBrown(EnergyModel):
    """Reduced energy u(x) = U(x) / scale, with scale tuning barrier heights.

    Native U has barriers ~100-200 (arbitrary units). For BG training we need
    barriers of moderate height (a few kT), so we rescale by `scale`. Default
    scale=20 yields barriers ~5-10 kT, reasonable for training.
    """

    def __init__(self, scale: float = 20.0) -> None:
        self.scale = scale
        self.dim = 2

    def __call__(self, x: Tensor) -> Tensor:
        return self.energy(x)

    def energy(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 2
        xx = x[..., 0:1]  # (..., 1)
        yy = x[..., 1:2]
        device = x.device
        A = _A.to(device)
        a = _a.to(device)
        b = _b.to(device)
        c = _c.to(device)
        x0 = _x0.to(device)
        y0 = _y0.to(device)
        dx = xx - x0
        dy = yy - y0
        terms = A * torch.exp(a * dx.pow(2) + b * dx * dy + c * dy.pow(2))
        U = terms.sum(dim=-1)
        return U / self.scale

    def grid(
        self,
        n: int = 200,
        x_span: tuple[float, float] = (-1.7, 1.2),
        y_span: tuple[float, float] = (-0.4, 2.1),
    ) -> tuple[Tensor, Tensor, Tensor]:
        xs = torch.linspace(*x_span, n)
        ys = torch.linspace(*y_span, n)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
        u = self.energy(grid).reshape(n, n)
        return gx, gy, u
