"""Synthetic Ramachandran-like potential — a stand-in for alanine dipeptide.

Real alanine dipeptide has a ~50-atom Cartesian description and a CHARMM/AMBER
force field. Implementing that here would pull in OpenMM or a full from-scratch
MM energy, both of which are out of scope.

Instead we model the *free-energy surface* in dihedral space (phi, psi) directly,
using a sum of Gaussian wells at the canonical alanine dipeptide minima. This
gives a 2D periodic potential with the right qualitative structure (alpha_R,
alpha_L, beta/PPII, C5/C7eq) so we can demonstrate BG/CFM training on a
molecular-flavored target without molecular machinery.

Coordinates: (phi, psi) in degrees, periodic in [-180, 180].
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# Approximate alanine dipeptide minima (degrees), depths in kT
_MINIMA = torch.tensor([
    [ -65.0,  -40.0],   # alpha_R    (deepest)
    [-150.0,  155.0],   # C5 / beta
    [ -80.0,   80.0],   # PPII / C7eq
    [  65.0,   40.0],   # alpha_L    (shallow)
])
_DEPTHS = torch.tensor([6.0, 5.0, 4.5, 2.5])     # in kT
_WIDTHS = torch.tensor([22.0, 28.0, 30.0, 30.0])  # degrees


def _wrap_deg(d: Tensor) -> Tensor:
    """Wrap angle differences to (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


class RamachandranDipeptide:
    """Synthetic 2D dipeptide free-energy surface in (phi, psi) degrees.

    u(phi, psi) = -log sum_k exp(-d_k(phi, psi) / w_k^2 + log depth_k) + const

    Each well is a Gaussian in periodic-angle distance. Total potential is a
    smooth log-sum-exp combination so derivatives are well-defined.
    """

    def __init__(self) -> None:
        self.dim = 2

    def __call__(self, x: Tensor) -> Tensor:
        return self.energy(x)

    def energy(self, x: Tensor) -> Tensor:
        """x: (..., 2) in degrees. Returns reduced energy (kT units)."""
        device = x.device
        minima = _MINIMA.to(device)
        depths = _DEPTHS.to(device)
        widths = _WIDTHS.to(device)
        # Periodic squared distance to each minimum
        dphi = _wrap_deg(x[..., 0:1] - minima[:, 0])  # (..., K)
        dpsi = _wrap_deg(x[..., 1:2] - minima[:, 1])
        d2 = (dphi.pow(2) + dpsi.pow(2)) / widths.pow(2)
        # log-sum-exp combination: u = -log sum exp(depth - d2)
        logits = depths - d2
        u = -torch.logsumexp(logits, dim=-1)
        return u

    def grid(self, n: int = 200) -> tuple[Tensor, Tensor, Tensor]:
        xs = torch.linspace(-180, 180, n)
        ys = torch.linspace(-180, 180, n)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
        u = self.energy(grid).reshape(n, n)
        return gx, gy, u

    @property
    def minima(self) -> Tensor:
        return _MINIMA.clone()
